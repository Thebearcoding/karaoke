"""Map displayed Chinese/Japanese lyric spans to sung phoneme sequences."""
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from pypinyin import lazy_pinyin
import pykakasi
from opencc import OpenCC

from lyric_progress import canonical_positions, japanese_tokenization

ROOT = Path(__file__).resolve().parent
KAKASI = pykakasi.kakasi()
SPOKEN_CHINESE = OpenCC('tw2s')
TRADITIONAL_CHINESE = OpenCC('s2t')
KANJI = r'[\u3400-\u9fff々〆]+'
SMALL = 'ゃゅょぁぃぅぇぉゎ'


def hiragana(text):
    text = unicodedata.normalize('NFKC', text)
    return ''.join(chr(ord(c)-0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)


def moras(text):
    result = []
    for i, character in enumerate(hiragana(text)):
        if character in SMALL and result:
            result[-1][1] = i+1
            result[-1][2] += character
        else:
            result.append([i,i+1,character])
    return result


@lru_cache(maxsize=4096)
def reading(text):
    return ''.join(part['hira'] for part in KAKASI.convert(text))


def reading_parts(text, kana):
    """Use written kana as anchors; retain ambiguous kanji compounds together."""
    parts = list(re.finditer(KANJI+r'|[^\u3400-\u9fff々〆]+', text))
    pattern = ''.join('(.+?)' if re.fullmatch(KANJI,p[0]) else re.escape(hiragana(p[0])) for p in parts)
    match = re.fullmatch(pattern,kana)
    if not match:
        return [(0,len(text),kana)]
    groups, output = iter(match.groups()), []
    for part in parts:
        written = part[0]
        if re.fullmatch(KANJI,written):
            spoken = next(groups)
            single = [reading(c) for c in written]
            if ''.join(single) == spoken:
                output.extend((part.start()+i,part.start()+i+1,k) for i,k in enumerate(single))
            else:
                output.append((part.start(),part.end(),spoken))
        else:
            output.extend((part.start()+a,part.start()+b,k) for a,b,k in moras(written))
    return output


def japanese_phone_groups(kana, previous_vowel=None):
    groups, vowel = [], previous_vowel
    for _,_,mora in moras(kana.replace('’','').replace("'",'')):
        if mora == 'ん':
            phones = ['N']
        elif mora == 'っ':
            phones = ['cl']
        elif mora == 'ー':
            if vowel is None:
                raise ValueError('Long vowel without a preceding vowel')
            phones = [vowel]
        else:
            roman = ''.join(p['hepburn'] for p in KAKASI.convert(mora))
            if not re.fullmatch('[a-z]*[aiueo]',roman):
                raise ValueError(f'Unsupported Japanese mora: {mora}')
            consonant, vowel = roman[:-1], roman[-1]
            phones = ([consonant] if consonant else [])+[vowel]
        groups.append(phones)
    return groups


def japanese_units(text, isolated=False):
    import pyopenjtalk
    normalized, positions, words = japanese_tokenization(text)
    output, cursor, previous_vowel = [], 0, None
    for word in words:
        written = word['string']
        if not any(c.isalnum() for c in written):
            continue
        first = normalized.find(written,cursor)
        if first < 0:
            raise ValueError(f'Japanese tokenizer changed display text: {written}')
        cursor = first+len(written)
        kana = reading(written)
        pronunciation = word['pron']
        if isolated and re.search(KANJI,written):
            pronunciation = ''.join(w['pron'] for w in pyopenjtalk.run_frontend(written))
        phone_groups = japanese_phone_groups(pronunciation,previous_vowel)
        previous_vowel = next((p for g in reversed(phone_groups) for p in reversed(g) if p in 'aiueo'),previous_vowel)
        parts = reading_parts(written,kana)
        if sum(len(moras(k)) for _,_,k in parts) != len(phone_groups):
            parts = [(0,len(written),kana)]
        offset = 0
        for a,b,spoken in parts:
            count = len(moras(spoken)) if len(parts)>1 else len(phone_groups)
            phones = [p for group in phone_groups[offset:offset+count] for p in group]
            offset += count
            if not phones:
                raise ValueError(f'Empty phoneme sequence for {written[a:b]}')
            begin, end = positions[first+a][0], positions[first+b-1][1]
            if output and begin < output[-1]['stop']:
                # A number such as 12 may yield separate 十/二 tokens. It is
                # still one original span, not two overlapping highlights.
                previous = output.pop()
                begin = previous['first']
                phones = previous['phones']+phones
                spoken = previous['reading']+spoken
            output.append(dict(first=begin,stop=end,text=text[begin:end],reading=spoken,phones=phones))
    return output


@lru_cache(maxsize=1)
def mandarin_dictionary():
    path = ROOT/'research/subtitles-2026-09-18/sofa/dictionary/opencpop-extension.txt'
    return {key:phones.split() for key,phones in (line.split('\t') for line in path.read_text().splitlines() if line.strip())}


def chinese_units(text):
    normalized, positions = canonical_positions(text,'Chinese')
    # Taiwan's particle 著 is 着 (zhe), while 著作 remains 著 (zhu).
    # Conversion needs the whole phrase; converting characters separately
    # discards the evidence used by OpenCC's regional phrase dictionary.
    spoken, _ = canonical_positions(chinese_pronunciation_text(text), 'Chinese')
    if len(spoken) != len(normalized):
        raise ValueError('Chinese pronunciation conversion changes display alignment')
    normalized = spoken
    pinyins = lazy_pinyin(normalized,errors=lambda value:list(value))
    if len(pinyins) != len(positions):
        raise ValueError('Chinese pronunciation does not map to displayed characters')
    dictionary = mandarin_dictionary()
    result = []
    for pinyin, position in zip(pinyins,positions):
        if not '\u3400' <= normalized[len(result)] <= '\u9fff' or pinyin not in dictionary:
            raise ValueError(f'Unsupported Chinese pronunciation: {text[position]} / {pinyin}')
        result.append(dict(first=position,stop=position+1,text=text[position],reading=pinyin,phones=dictionary[pinyin]))
    return result


def chinese_pronunciation_text(text):
    # Canonicalize mixed scripts first: tw2s treats a simplified 么 as the
    # traditional form of 幺. Genuine 幺 remains unchanged through s2t.
    return SPOKEN_CHINESE.convert(TRADITIONAL_CHINESE.convert(text))


def spoken_units(text,language,isolated=False):
    result = japanese_units(text,isolated) if language == 'Japanese' else chinese_units(text)
    if canonical_positions(''.join(u['text'] for u in result),language)[0] != canonical_positions(text,language)[0]:
        raise ValueError('Phonetic mapping does not cover the displayed lyric')
    return result


def phoneme_sequence(units):
    phones, owners = ['SP'], [-1]
    for i,unit in enumerate(units):
        phones.extend(unit['phones']); owners.extend([i]*len(unit['phones']))
        phones.append('SP'); owners.append(-1)
    return phones, [str(i) for i in range(len(units))], owners
