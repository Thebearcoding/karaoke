"""Shared, transparent subtitle typography for previews and exported videos."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import unicodedata

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont


TOKENS = json.loads(Path(__file__).with_name('subtitle-styles.json').read_text())
STYLES = TOKENS['styles']
DEFAULT_STYLE = TOKENS['default_style']
KANJI = r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff々〆]+'


class CaptionOverflow(ValueError):
    pass


@lru_cache(maxsize=4096)
def japanese_readings(text):
    """Dictionary readings of whole words, anchored to the unchanged lyric text."""
    if not re.search(KANJI, text):
        return (), ()
    # Retain original offsets even when the tokenizer normalizes width or dakuten.
    normalized, positions = '', []
    for cluster in re.finditer(r'.[\u0300-\u036f\u3099\u309a\uff9e\uff9f]*', text, re.S):
        value = unicodedata.normalize('NFKC', cluster[0])
        normalized += value
        positions.extend([(cluster.start(), cluster.end())]*len(value))

    def hira(value):
        return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c
                       for c in unicodedata.normalize('NFKC', value))

    try:
        import pyopenjtalk
        words = pyopenjtalk.run_frontend(normalized)
    except Exception as error:
        # A missing optional reading must never discard valid, timed lyrics.
        return (), (f'{type(error).__name__}: {error}',)
    spans, cursor = [], 0
    for word in words:
        written = unicodedata.normalize('NFKC', word['string'])
        first = normalized.find(written, cursor)
        if first < 0 or not written:
            continue
        cursor = first+len(written)
        if not re.search(KANJI, written):
            continue
        parts = list(re.finditer(KANJI+r'|[^\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff々〆]+', written))
        pattern = ''.join('([ぁ-ゖー]+?)' if re.fullmatch(KANJI, p[0]) else re.escape(hira(p[0]))
                          for p in parts)
        # "read" preserves spelling (きょう); "pron" can replace it with きょー.
        matched = re.fullmatch(pattern, hira(word.get('read', '')))
        if not matched:
            continue
        readings = iter(matched.groups())
        for part in parts:
            if re.fullmatch(KANJI, part[0]):
                a, b = positions[first+part.start()][0], positions[first+part.end()-1][1]
                spans.append((a, b, next(readings)))
    unresolved = tuple(f'No dictionary reading for {m[0]!r} at {m.start()}'
                       for m in re.finditer(KANJI, text)
                       if any(not any(a <= i < b for a, b, _ in spans)
                              for i in range(m.start(), m.end())))
    return tuple(spans), unresolved


def caption_height(language):
    return TOKENS['strip_height']+(TOKENS['furigana']['extra_height'] if language == 'Japanese' else 0)


def annotated_lines(text, readings, font, ruby_font, tracking, maximum, balanced):
    """Wrap whole reading groups, reserving their width instead of squeezing kana."""
    spans = {first: (stop, reading) for first, stop, reading in readings}
    items, cursor = [], 0
    while cursor < len(text):
        stop, reading = spans.get(cursor, (None, None))
        if stop is None:
            token = re.match(r"[A-Za-z0-9’'\-]+|[^\S\n]+|[^\n]|\n", text[cursor:])[0]
            stop = cursor+len(token)
        token = text[cursor:stop]
        width = max(line_width(token, font, tracking), ruby_font.getlength(reading) if reading else 0)
        items.append(dict(text=token, first=cursor, stop=stop, reading=reading, width=width))
        cursor = stop

    def trim(row):
        while row and row[-1]['text'].isspace():
            row = row[:-1]
        while row and row[0]['text'].isspace():
            row = row[1:]
        return row

    def width(row):
        return sum(item['width'] for item in row)+max(0, len(row)-1)*tracking

    rows, row = [], []
    for item in items:
        if item['text'] == '\n':
            rows.append(trim(row))
            row = []
        elif row and width(row+[item]) > maximum:
            rows.append(trim(row))
            row = [] if item['text'].isspace() else [item]
        else:
            row.append(item)
    rows.append(trim(row))
    if balanced and len(rows) == 2 and '\n' not in text:
        candidates = []
        for i in range(1, len(items)):
            left, right = trim(items[:i]), trim(items[i:])
            if not left or not right or right[0]['text'][0] in '、。，．？！!?：:；;）)]」』ぁぃぅぇぉっゃゅょァィゥェォッャュョー':
                continue
            a, b = width(left), width(right)
            if max(a, b) <= maximum:
                semantic = items[i-1]['text'].isspace() or items[i]['text'].isspace() or items[i-1]['text'][-1] in '、，,。.!?？！;；'
                candidates.append((abs(a-b)+(0 if semantic else maximum*.15), i, left, right))
        if candidates:
            rows = list(min(candidates)[2:])
    return rows


@lru_cache(maxsize=32)
def system_font(name):
    path = Path('/System/Library/Fonts')/name
    if path.exists():
        return path
    # macOS stores some Japanese font filenames in decomposed Unicode.
    for candidate in path.parent.iterdir():
        if unicodedata.normalize('NFC', candidate.name) == path.name:
            return candidate
    raise FileNotFoundError(path)


def script_for(text, language='auto'):
    if language != 'auto':
        return language
    if re.search('[\u3040-\u30ff]', text):
        return 'Japanese'
    if re.search('[\u3400-\u9fff]', text):
        return 'Chinese'
    return 'Latin'


@lru_cache(maxsize=32)
def coverage(path, index):
    with TTFont(str(path), fontNumber=index, lazy=True) as font:
        return set(font.getBestCmap())


def resolve_font(text, style=DEFAULT_STYLE, language='auto', font_file=None):
    font_file = font_file or os.getenv('KARAOKE_FONT_FILE')
    script = script_for(text, language)
    name, index = STYLES[style]['fonts'][script]
    path = Path(font_file) if font_file else system_font(name)
    if font_file:
        index = 0
    missing = {c for c in text if not c.isspace() and ord(c) not in coverage(path, index)}
    if missing:
        raise ValueError(f'Font {path.name} lacks glyphs: {"".join(sorted(missing))}')
    return path, index, script


def line_width(text, font, tracking):
    return sum(font.getlength(c) for c in text)+max(0, len(text)-1)*tracking


def wrap_lines(text, font, tracking, maximum):
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        # Keep Latin words intact; CJK can wrap between characters.
        for token in re.findall(r"[A-Za-z0-9’'\-]+|[^\S\n]+|[^\n]", paragraph):
            if line and line_width(line+token, font, tracking) > maximum:
                lines.append(line.rstrip())
                line = token.lstrip()
            else:
                line += token
        lines.append(line.rstrip())
    return lines


def balanced_lines(text, font, tracking, maximum):
    lines = wrap_lines(text, font, tracking, maximum)
    if '\n' in text or len(lines) != 2:
        return lines
    candidates = []
    for index in range(1, len(text)):
        left, right = text[:index].rstrip(), text[index:].lstrip()
        if not left or not right or right[0] in '、。，．？！!?：:；;）)]」』ぁぃぅぇぉっゃゅょァィゥェォッャュョー':
            continue
        if text[index-1].isascii() and text[index-1].isalnum() and text[index].isascii() and text[index].isalnum():
            continue
        a, b = line_width(left, font, tracking), line_width(right, font, tracking)
        if max(a, b) <= maximum:
            semantic_break = text[index-1].isspace() or text[index].isspace() or text[index-1] in '、，,。.!?？！;；'
            candidates.append((abs(a-b)+(0 if semantic_break else maximum*.15), left, right))
    return list(min(candidates)[1:]) if candidates else lines


def outline_mask(mask, radius):
    """Dilate only the occupied region, including the filter's full halo."""
    outline = Image.new('L', mask.size)
    bounds = mask.getbbox()
    if bounds is not None:
        left, top, right, bottom = bounds
        bounds = (max(0, left-radius), max(0, top-radius),
                  min(mask.width, right+radius), min(mask.height, bottom+radius))
        outline.paste(mask.crop(bounds).filter(ImageFilter.MaxFilter(2*radius+1)), bounds)
    return outline


def caption_image(text, width=1280, height=None, font_file=None,
                  style=DEFAULT_STYLE, tone='light', language='auto', layout_out=None,
                  style_overrides=None, readings=()):
    """Return a transparent strip. No box, gradient, or background repainting."""
    scale = width/TOKENS['reference_width']
    height = height or round(TOKENS['strip_height']*scale)
    supersample = 2
    image = Image.new('RGBA', (width*supersample, height*supersample))
    if not text.strip():
        return image.resize((width, height))
    spec = {**STYLES[style], **(style_overrides or {})}
    path, index, script = resolve_font(text, style, language, font_file)
    font_scale = scale*supersample
    tracking_em = spec['latin_tracking_em'] if script == 'Latin' else spec['tracking_em']
    maximum = width*(1-2*TOKENS['safe_x'])*supersample
    one_line = [(size, True) for size in range(spec['size'], spec.get('single_line_floor', spec['size']+1)-1, -2)]
    sizes = one_line+[(size, False) for size in range(spec['size'], spec.get('minimum_size', TOKENS['minimum_font_size'])-1, -2)]
    for size, require_single in sizes:
        font = ImageFont.truetype(str(path), round(size*font_scale), index=index)
        tracking = size*font_scale*tracking_em
        if readings:
            ruby_font = ImageFont.truetype(str(path), max(1, round(size*font_scale*TOKENS['furigana']['size_em'])), index=index)
            rows = annotated_lines(text, readings, font, ruby_font, tracking, maximum, spec.get('balanced_wrap'))
            lines = [''.join(item['text'] for item in row) for row in rows]
            widths = [sum(item['width'] for item in row)+max(0, len(row)-1)*tracking for row in rows]
        else:
            lines = (balanced_lines if spec.get('balanced_wrap') else wrap_lines)(text, font, tracking, maximum)
            widths = [line_width(line, font, tracking) for line in lines]
        if require_single and len(lines) > 1:
            continue
        if len(lines) <= spec.get('maximum_lines', 2) and all(length <= maximum for length in widths):
            break
    else:
        raise CaptionOverflow('Caption exceeds the readable line limit; split the lyric cue')
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)
    ruby_mask = Image.new('L', image.size)
    ruby_draw = ImageDraw.Draw(ruby_mask)
    baseline = (height-spec.get('baseline_inset', 36)*scale)*supersample
    characters, ruby_boxes = [], []
    cursor = 0
    for i, line in enumerate(lines):
        length = widths[i]
        x = ((width*supersample-length)/2 if spec['alignment'] == 'center'
             else width*TOKENS['safe_x']*supersample)
        line_height = size*font_scale*spec['line_height']
        if readings:
            line_height += ruby_font.size+TOKENS['furigana']['line_gap']*font_scale
        y = baseline-(len(lines)-i-1)*line_height
        offset = text.find(line, cursor)
        if offset < 0:
            raise ValueError('Wrapped caption no longer matches its source text')
        row = rows[i] if readings else [dict(text=line, first=offset, width=length, reading=None)]
        for item in row:
            glyph_x = x+(item['width']-line_width(item['text'], font, tracking))/2
            for j, character in enumerate(item['text']):
                draw.text((round(glyph_x), round(y)), character, font=font, fill=255, anchor='ls')
                characters.append({'index': item['first']+j, 'line': i, 'character': character,
                    'left': glyph_x/supersample, 'right': (glyph_x+font.getlength(character))/supersample,
                    'top': (y-size*font_scale)/supersample, 'baseline': y/supersample})
                glyph_x += font.getlength(character)+tracking
            if item['reading']:
                position = (round(x+(item['width']-ruby_font.getlength(item['reading']))/2),
                            round(y-size*font_scale-TOKENS['furigana']['gap']*font_scale))
                ruby_draw.text(position, item['reading'], font=ruby_font, fill=255, anchor='ls')
                box = ruby_draw.textbbox(position, item['reading'], font=ruby_font, anchor='ls')
                ruby_boxes.append(dict(first=item['first'], stop=item['stop'], reading=item['reading'], line=i,
                                       bounds=tuple(value/supersample for value in box)))
            x += item['width']+tracking
        cursor = offset+len(line)
    visible_mask = ImageChops.lighter(mask, ruby_mask)
    if tone not in ('light', 'dark'):
        raise ValueError('tone must be light or dark')
    color = spec['color'] if tone == 'light' else '#172028'
    shadow_rgb = (0, 0, 0) if tone == 'light' else (255, 255, 255)
    # A fine local edge and soft shadow protect glyphs without a heavy outline.
    edge_radius = max(1, round(spec.get('outline_width', .5)*font_scale))
    for alpha in [visible_mask.filter(ImageFilter.GaussianBlur(spec.get('shadow_blur', 2.4)*font_scale))
                       .point(lambda a: round(a*spec.get('shadow_alpha', .6))),
                  outline_mask(visible_mask, edge_radius)
                       .point(lambda a: round(a*spec.get('outline_alpha', .34)))]:
        shadow = Image.new('RGBA', image.size, shadow_rgb+(0,))
        shadow.putalpha(alpha)
        image.alpha_composite(shadow)
    foreground = Image.new('RGBA', image.size, ImageColor.getrgb(color)+(0,))
    foreground.putalpha(visible_mask)
    image.alpha_composite(foreground)
    if layout_out is not None:
        layout_out.update(mask=mask.resize((width, height), Image.Resampling.LANCZOS),
                          characters=characters, font_size=size, lines=lines, ruby=ruby_boxes,
                          content_top=min([c['top'] for c in characters]+[r['bounds'][1] for r in ruby_boxes]))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def with_opacity(image, opacity):
    result = image.copy()
    result.putalpha(image.getchannel('A').point(lambda a: round(a*opacity)))
    return result


def compact_caption_image(text, **options):
    """Fit auxiliary text; the current lyric caption always uses caption_image."""
    try:
        return caption_image(text, **options)
    except CaptionOverflow:
        low, high = 0, len(text)-1
        row = caption_image('…', **dict(options, readings=()))
        while low <= high:
            middle = (low+high)//2
            candidate_text = text[:middle].rstrip()+'…'
            candidate_options = dict(options)
            if options.get('readings'):
                candidate_options['readings'] = tuple(
                    (a,b,kana) for a,b,kana in options['readings'] if b <= len(candidate_text)-1)
            try:
                candidate = caption_image(candidate_text, **candidate_options)
            except CaptionOverflow:
                high = middle-1
            else:
                row, low = candidate, middle+1
        return row


def song_intro_image(info, width=1280, style=DEFAULT_STYLE):
    """Transparent song credits, using the same typefaces as the lyrics."""
    title = info.get('title', '').strip()
    if not title:
        raise ValueError('Song information requires a title')
    spec = TOKENS['intro']
    scale = width/TOKENS['reference_width']
    image = Image.new('RGBA', (width, round(spec['height']*scale)))
    rows = [(title, 0, 240, spec['title_size'], spec['title_minimum_size'], 40, 2)]
    artist = (info.get('artist') or info.get('original_artist') or '').strip()
    if artist:
        rows.append((f'演唱 · {artist}', 242, 72, spec['artist_size'], 24, 22, 1))
    version = info.get('version', '').strip()
    if version:
        rows.append((version, 310, 52, spec['version_size'], 20, 18, 1))
    for text, top, height, size, minimum, inset, lines in rows:
        row = compact_caption_image(text, width=width, height=round(height*scale), style=style,
            style_overrides={'size': size, 'minimum_size': minimum, 'single_line_floor': minimum,
                'maximum_lines': lines, 'baseline_inset': inset, 'alignment': 'center',
                'balanced_wrap': True})
        image.alpha_composite(row, (0, round(top*scale)))
    color = ImageColor.getrgb(STYLES[style].get('progress_color', '#e59a24'))+(255,)
    ImageDraw.Draw(image).rectangle((round(width/2-24*scale), round(229*scale),
                                    round(width/2+24*scale), round(230*scale)), fill=color)
    return image


def cue_opacity(elapsed, duration, style=DEFAULT_STYLE):
    if not 0 <= elapsed < duration:
        return 0.
    spec = STYLES[style]
    ingress = min(spec['fade_in_ms']/1000, duration/4)
    egress = min(spec['fade_out_ms']/1000, duration/4)
    return max(0., min(1., elapsed/ingress if ingress else 1.,
                       (duration-elapsed)/egress if egress else 1.))
