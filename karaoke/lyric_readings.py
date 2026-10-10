"""Source-bound Japanese readings and cached Cantonese dictionary readings."""
from difflib import SequenceMatcher
import hashlib
import inspect
from html.parser import HTMLParser
import json
import re
import time
import unicodedata
from urllib.parse import urlencode


def hiragana(text):
    return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c
                   for c in unicodedata.normalize('NFKC', text))


def separate_inline_readings(lines, evidence):
    """Separate quoted ruby only when a publisher confirms the exact pair."""
    from .lyric_lookup import comparison_key
    pairs = {(comparison_key(p['text']), hiragana(p['reading'])): source['url']
             for source in evidence for p in source['pairs']}
    if not pairs:
        return lines, []
    pattern = re.compile(r'([《≪「『])([^《》≪≫「」『』\n]+)([》≫」』])\s*[（(]([ぁ-ゖァ-ヶー]+)[）)]')
    cleaned, annotations = [], []
    for line in lines:
        parts, spans, cursor = [], [], 0
        for match in pattern.finditer(line):
            base, reading = match[2], hiragana(match[4])
            source = pairs.get((comparison_key(base), reading))
            if (dict(zip('《≪「『', '》≫」』'))[match[1]] != match[3]
                    or not re.search('[\u3400-\u9fff]', base) or not source):
                continue
            parts.append(line[cursor:match.start(2)])
            first = sum(map(len, parts))
            parts.append(base + match[3])
            spans.append(dict(first=first, stop=first+len(base), text=base, reading=reading, url=source))
            cursor = match.end()
        parts.append(line[cursor:])
        cleaned.append(''.join(parts))
        if spans:
            annotations.append(dict(text=cleaned[-1], source_text=line, spans=spans))
    return cleaned, annotations


class ReadingPage(HTMLParser):
    """Read UtaTen's ruby body; navigation, romanization and comments are excluded."""
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.text, self.spans, self.links, self.identity = '', [], [], {}
        self.depth, self.stack, self.ruby, self.in_reading = 0, [], None, False
        self.page_url, self.script, self.compositions = None, None, []
        self.search_depth = 0
        self.includes_translation = False
        self.feed(html)
        # The display title mixes artist and soundtrack credits. Use the
        # composition explicitly attached to this lyric page when available.
        for record in self.compositions:
            lyric = record.get('Lyrics', record.get('lyrics'))
            recording = record.get('recordedAs')
            if not isinstance(lyric, dict) or not isinstance(recording, dict):
                continue
            page = lyric.get('mainEntityOfPage')
            if not self.page_url or not isinstance(page, dict) or page.get('url') != self.page_url:
                continue
            artist = recording.get('byArtist')
            if not isinstance(artist, dict) or not all(isinstance(v, str) and v.strip()
                    for v in (record.get('name'), artist.get('name'))):
                continue
            aliases = lambda value: [s for s in (value if isinstance(value, list) else [value])
                                     if isinstance(s, str) and s.strip()]
            self.identity = dict(title=record['name'], artist=artist['name'],
                title_aliases=aliases(record.get('alternateName')),
                artist_aliases=aliases(artist.get('alternateName')))
            break

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'table' and (self.search_depth or 'searchResult' in classes):
            self.search_depth += 1
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.script = ''
        if tag == 'meta' and attrs.get('property') == 'og:url':
            self.page_url = attrs.get('content')
        if tag == 'meta' and attrs.get('property') == 'og:title':
            self.includes_translation = bool(re.search(r'【[^】]*和訳[^】]*】\s*ふりがな付', attrs.get('content', '')))
            match = re.fullmatch(r'(.+?) 歌詞 (.+?) ふりがな付(?: - .+)?', attrs.get('content', ''))
            if match:
                self.identity = dict(title=match[1].strip(), artist=match[2].strip())
        if self.search_depth and tag == 'a' and re.fullmatch(r'/lyric/[A-Za-z0-9]+/', attrs.get('href', '')):
            if attrs['href'] not in self.links:
                self.links.append(attrs['href'])
        if tag == 'div':
            if self.depth:
                self.depth += 1
            elif 'hiragana' in classes and not self.text:
                self.depth = 1
        if not self.depth:
            return
        if tag == 'br':
            self.text += '\n'
        elif tag == 'span':
            role = next((c for c in classes if c in ('ruby', 'rb', 'rt')), '')
            self.stack.append(role)
            if role == 'ruby':
                self.ruby = dict(first=len(self.text), reading='')
            elif role == 'rt':
                self.in_reading = True

    def handle_endtag(self, tag):
        if tag == 'table' and self.search_depth:
            self.search_depth -= 1
        if tag == 'script' and self.script is not None:
            try:
                data = json.loads(self.script)
                records = data if isinstance(data, list) else data.get('@graph', [data])
                self.compositions.extend(r for r in records
                    if isinstance(r, dict) and r.get('@type') == 'MusicComposition')
            except (ValueError, AttributeError, TypeError):
                pass
            self.script = None
        if not self.depth:
            return
        if tag == 'div':
            self.depth -= 1
        elif tag == 'span' and self.stack:
            role = self.stack.pop()
            if role == 'rt':
                self.in_reading = False
            elif role == 'ruby' and self.ruby is not None:
                self.spans.append(dict(self.ruby, stop=len(self.text)))
                self.ruby = None

    def handle_data(self, text):
        if self.script is not None:
            self.script += text
        if self.depth:
            if self.in_reading and self.ruby is not None:
                self.ruby['reading'] += text
            else:
                self.text += text


def text_positions(text):
    """Match typography without losing the original character boundaries."""
    normalized, positions = '', []
    for i, character in enumerate(text):
        for c in unicodedata.normalize('NFKC', character):
            if c.isalnum():
                normalized += c
                positions.append(i)
    return normalized, positions


def match_readings(lines, page):
    expected, positions = text_positions('\n'.join(lines))
    published, source_positions = text_positions(page.text)
    blocks = SequenceMatcher(None, published, expected, autojunk=False).get_matching_blocks()
    matched = sum(b.size for b in blocks)
    coverage = matched / max(1, len(expected), len(published))
    # A shared short phrase cannot establish which song these readings belong to.
    if not expected or coverage < .9:
        return None, coverage
    mapping = {b.a+i: b.b+i for b in blocks for i in range(b.size)}
    by_source = {raw: i for i, raw in enumerate(source_positions)}
    rows, owners = [], []
    for line in lines:
        rows.append(dict(text=line, spans=[]))
        owners.extend((len(rows)-1, i) for i in range(len(line)+1))
    for span in page.spans:
        reading = hiragana(span['reading'].strip())
        indexes = [by_source[i] for i in range(span['first'], span['stop']) if i in by_source]
        target = [mapping.get(i) for i in indexes]
        if (not target or None in target or target != list(range(target[0], target[0]+len(target)))
                or not re.fullmatch('[ぁ-ゖー]+', reading)):
            continue
        line, first = owners[positions[target[0]]]
        last_line, stop = owners[positions[target[-1]]]
        if line == last_line:
            rows[line]['spans'].append(dict(first=first, stop=stop+1,
                text=lines[line][first:stop+1], reading=reading))
    return rows, coverage


def read_page(url):
    import httpx
    with httpx.stream('GET', url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'}) as response:
        response.raise_for_status()
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > 2*1024*1024:
                raise ValueError('Reading page exceeds 2 MiB')
    return body.decode('utf-8')


def prepare_readings(lines, source_url, cache, language='Japanese'):
    """Optional, cached presentation evidence; unavailable readings never lose lyrics."""
    from .lyric_lookup import cached_lyric_request, lyric_request_error
    path = cache.root/'lyric-readings.json'
    identity = dict(lines=lines, source_url=source_url)
    inline = []
    if language == 'Japanese':
        try:
            lyrics = json.loads((cache.root/'lyrics.json').read_text())
            if lyrics.get('source_url') == source_url:
                inline = lyrics.get('provenance', {}).get('lyric_lookup', {}).get('inline_readings', [])
        except (OSError, ValueError):
            pass
        if inline:
            identity['inline_readings'] = inline
    if language == 'Japanese':
        identity['parser_sha256'] = hashlib.sha256(inspect.getsource(ReadingPage).encode()).hexdigest()
    if language == 'Cantonese':
        from subtitle_style import cantonese_readings
        from importlib.metadata import PackageNotFoundError, version
        try:
            installed = version('pycantonese')
        except PackageNotFoundError:
            installed = None
        identity.update(language=language, dictionary='pycantonese', version=installed,
            mapper_sha256=hashlib.sha256(inspect.getsource(cantonese_readings).encode()).hexdigest())
    try:
        old = json.loads(path.read_text())
        if old['identity'] == identity and (old.get('error', {}).get('retry_at') or float('inf')) > time.time():
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    packet = dict(identity=identity, status='unavailable', lines=[], attempts=[])
    try:
        if language == 'Cantonese':
            for text in lines:
                spans, warnings = cantonese_readings(text)
                packet['lines'].append(dict(text=text, spans=[dict(first=a, stop=b, text=text[a:b], reading=r)
                    for a, b, r in spans], warnings=list(warnings)))
            packet['status'] = 'dictionary'
        query = next((line.strip() for line in lines if re.search('[\u3400-\u9fff]', line)), '') if language == 'Japanese' else ''
        if query:
            url = 'https://utaten.com/search?'+urlencode(dict(body=query))
            fetch = lambda url: cached_lyric_request('utaten_readings', url, cache, lambda: read_page(url))
            listing = ReadingPage(fetch(url))
            for link in listing.links[:3]:
                url = 'https://utaten.com'+link
                html = fetch(url)
                rows, coverage = match_readings(lines, ReadingPage(html))
                packet['attempts'].append(dict(url=url, text_coverage=coverage))
                if rows and any(row['spans'] for row in rows):
                    packet.update(status='published', lines=rows, url=url,
                                  page_sha256=hashlib.sha256(html.encode()).hexdigest())
                    break
    except Exception as error:
        packet['error'] = lyric_request_error(error)
    if inline:
        supplied = {row['text']: row['spans'] for row in inline}
        if not packet['lines']:
            packet['lines'] = [dict(text=text, spans=[]) for text in lines]
        for row in packet['lines']:
            additions = supplied.get(row['text'], [])
            row['spans'] = sorted([*additions, *(s for s in row['spans'] if not any(
                s['first'] < a['stop'] and a['first'] < s['stop'] for a in additions))], key=lambda s:s['first'])
        if any(row['spans'] for row in packet['lines']):
            packet['status'] = 'published'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    return path


def resolve_readings(text, annotations=(), language='Japanese'):
    from subtitle_style import japanese_readings, cantonese_readings
    if language not in ('Japanese', 'Cantonese'):
        return (), ()
    fallback, warnings = (japanese_readings if language == 'Japanese' else cantonese_readings)(text)
    supplied = []
    for item in annotations:
        try:
            first, stop = item['first'], item['stop']
            reading = hiragana(item['reading']) if language == 'Japanese' else item['reading']
            pattern = '[ぁ-ゖー]+' if language == 'Japanese' else r'[a-z]+[1-6](?: [a-z]+[1-6])*'
            if (not isinstance(first, int) or not isinstance(stop, int) or not 0 <= first < stop <= len(text)
                    or text[first:stop] != item['text'] or not re.fullmatch(pattern, reading)
                    or supplied and first < supplied[-1][1]):
                raise ValueError('Reading does not match lyric span')
            supplied.append((first, stop, reading))
        except (KeyError, TypeError, ValueError):
            warnings += ('Ignored a published reading that does not match this lyric span',)
    return tuple(sorted([*supplied, *(s for s in fallback if not any(
        s[0] < stop and first < s[1] for first, stop, _ in supplied))])), warnings


def load_readings(path, cues, language='Japanese'):
    """Refuse stale annotations without changing any lyric or timing data."""
    try:
        packet = json.loads(path.read_text())
        if (packet['identity'].get('language', 'Japanese') == language
                and packet['identity']['lines'] == [cue['text'] for cue in cues] and isinstance(packet['lines'], list) and all(
                isinstance(row, dict) and isinstance(row.get('text'), str) and isinstance(row.get('spans'), list)
                for row in packet['lines'])):
            return packet['lines'], packet
        return [], dict(status='stale_lyric_body')
    except (OSError, ValueError, KeyError, TypeError):
        return [], dict(status='unavailable')
