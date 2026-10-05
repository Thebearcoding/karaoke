"""Shared, transparent subtitle typography for previews and exported videos."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import unicodedata

from fontTools.ttLib import TTFont
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


TOKENS = json.loads(Path(__file__).with_name('subtitle-styles.json').read_text())
STYLES = TOKENS['styles']
DEFAULT_STYLE = TOKENS['default_style']


@lru_cache(maxsize=32)
def system_font(name):
    path = Path('/System/Library/Fonts')/name
    if path.exists():
        return path
    # macOS stores some Japanese font filenames in decomposed Unicode.
    return next(p for p in path.parent.iterdir()
                if unicodedata.normalize('NFC', p.name) == path.name)


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
    if '\n' in text or len(lines) < 2:
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


def caption_image(text, width=1280, height=None, font_file=None,
                  style=DEFAULT_STYLE, tone='light', language='auto', layout_out=None,
                  style_overrides=None):
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
        lines = (balanced_lines if spec.get('balanced_wrap') else wrap_lines)(text, font, tracking, maximum)
        if require_single and len(lines) > 1:
            continue
        if len(lines) <= spec.get('maximum_lines', 2) and all(line_width(line, font, tracking) <= maximum for line in lines):
            break
    else:
        raise ValueError('Caption exceeds the readable line limit; split the lyric cue')
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)
    baseline = (height-spec.get('baseline_inset', 36)*scale)*supersample
    characters = []
    cursor = 0
    for i, line in enumerate(lines):
        length = line_width(line, font, tracking)
        x = ((width*supersample-length)/2 if spec['alignment'] == 'center'
             else width*TOKENS['safe_x']*supersample)
        y = baseline-(len(lines)-i-1)*size*font_scale*spec['line_height']
        offset = text.find(line, cursor)
        if offset < 0:
            raise ValueError('Wrapped caption no longer matches its source text')
        for j, character in enumerate(line):
            draw.text((round(x), round(y)), character, font=font, fill=255, anchor='ls')
            characters.append({'index': offset+j, 'line': i, 'character': character,
                'left': x/supersample, 'right': (x+font.getlength(character))/supersample,
                'top': (y-size*font_scale)/supersample, 'baseline': y/supersample})
            x += font.getlength(character)+tracking
        cursor = offset+len(line)
    if tone not in ('light', 'dark'):
        raise ValueError('tone must be light or dark')
    color = spec['color'] if tone == 'light' else '#172028'
    shadow_rgb = (0, 0, 0) if tone == 'light' else (255, 255, 255)
    # A fine local edge and soft shadow protect glyphs without a heavy outline.
    edge_radius = max(1, round(spec.get('outline_width', .5)*font_scale))
    for alpha in [mask.filter(ImageFilter.GaussianBlur(spec.get('shadow_blur', 2.4)*font_scale))
                       .point(lambda a: round(a*spec.get('shadow_alpha', .6))),
                  mask.filter(ImageFilter.MaxFilter(2*edge_radius+1))
                       .point(lambda a: round(a*spec.get('outline_alpha', .34)))]:
        shadow = Image.new('RGBA', image.size, shadow_rgb+(0,))
        shadow.putalpha(alpha)
        image.alpha_composite(shadow)
    foreground = Image.new('RGBA', image.size, ImageColor.getrgb(color)+(0,))
    foreground.putalpha(mask)
    image.alpha_composite(foreground)
    if layout_out is not None:
        layout_out.update(mask=mask.resize((width, height), Image.Resampling.LANCZOS),
                          characters=characters, font_size=size, lines=lines)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def with_opacity(image, opacity):
    result = image.copy()
    result.putalpha(image.getchannel('A').point(lambda a: round(a*opacity)))
    return result


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
        row = caption_image(text, width=width, height=round(height*scale), style=style,
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
