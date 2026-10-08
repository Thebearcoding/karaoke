"""Render stationary lyrics with a left-to-right word sweep or an underline."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageChops, ImageColor, ImageDraw

from subtitle_style import TOKENS, STYLES, CaptionOverflow, caption_image, compact_caption_image, song_intro_image
from subtitles import countdown_events, countdown_state, normalized, read_ahead_state
from stage_progress import progress_items, report_progress


class Caption:
    def __init__(self, cue, style, language):
        self.cue = cue
        self.layout = {}
        self.color = ImageColor.getrgb(STYLES[style].get('progress_color', '#efd29a'))
        self.pages = []
        try:
            self.base = caption_image(cue['text'], style=style, language=language, layout_out=self.layout)
        except CaptionOverflow:
            self.pages = self.paginate(style, language)
            self.base = self.pages[0].base
            self.starts = [page.cue['units'][0]['start'] for page in self.pages]
            self.countdown_geometry = self.pages[0].countdown_geometry
            return
        self.starts = [u['start'] for u in cue.get('units', [])]
        self.cache = {}
        spec = TOKENS['countdown']
        self.countdown_geometry = dict(radius=spec['radius'], spacing=spec['spacing'],
            color=STYLES[style].get('progress_color', '#efd29a'),
            y=max(spec['radius']+3, min(c['top'] for c in self.layout['characters'])-spec['above_text']))

    def paginate(self, style, language):
        """Change only the display pages, at existing measured unit boundaries."""
        text, units = self.cue['text'], self.cue.get('units', [])
        if not units or normalized(text[:units[0]['first']]) or normalized(text[units[-1]['stop']:]) or any(
                a['stop'] > b['first'] or normalized(text[a['stop']:b['first']]) or a['start'] >= b['start']
                for a,b in zip(units, units[1:])):
            raise CaptionOverflow('Long captions require complete, ordered timing units')
        pages, first = [], 0
        while first < len(units):
            low, high, fit = first+1, len(units), None
            begin = units[first]['first'] if first else 0
            while low <= high:
                stop = (low+high)//2
                end = units[stop]['first'] if stop < len(units) else len(text)
                try:
                    caption_image(text[begin:end], style=style, language=language)
                except CaptionOverflow:
                    high = stop-1
                else:
                    fit, low = stop, stop+1
            if fit is None:
                raise CaptionOverflow('One timing unit exceeds a readable page; finer acoustic units are needed')
            end = units[fit]['first'] if fit < len(units) else len(text)
            page = dict(self.cue, text=text[begin:end], units=[
                dict(unit, first=unit['first']-begin, stop=unit['stop']-begin) for unit in units[first:fit]])
            pages.append(Caption(page, style, language))
            first = fit
        return pages

    def countdown(self, remaining, total):
        """Fixed dot positions above the lyric; exhausted dots become faint rings."""
        spec = self.countdown_geometry
        image = Image.new('RGBA', self.base.size)
        draw = ImageDraw.Draw(image)
        y = spec['y']
        for index in range(total):
            x = image.width/2+(index-(total-1)/2)*spec['spacing']
            radius = spec['radius']
            bounds = (x-radius, y-radius, x+radius, y+radius)
            active = index >= total-remaining
            draw.ellipse((bounds[0]-1, bounds[1]-1, bounds[2]+1, bounds[3]+1), fill=(0, 0, 0, 190))
            draw.ellipse(bounds, fill=self.color+(255,) if active else (0, 0, 0, 0),
                         outline=self.color+(255,) if active else (246, 244, 236, 160), width=2)
        return image

    def frame(self, instant, mode, *, page_at=None):
        if self.pages:
            page = max(0, bisect_right(self.starts, instant if page_at is None else page_at)-1)
            return self.pages[page].frame(instant, mode)
        index = bisect_right(self.starts, instant)-1
        if mode == 'static' or index < 0:
            return self.base
        unit = self.cue['units'][index]
        if mode == 'underline':
            next_start = self.starts[index+1] if index+1 < len(self.starts) else unit['end']+.2
            hold_end = next_start if next_start-unit['end'] <= .6 else unit['end']
            if instant >= hold_end:
                return self.base
        key = (mode, index)
        sweeping = [u for u in self.cue['units'][:index+1] if instant < u['end']] if mode == 'word' else []
        cacheable = not sweeping
        if cacheable and key in self.cache:
            return self.cache[key]
        result = self.base.copy()
        chars = self.layout['characters']
        if mode == 'word':
            area = Image.new('L', result.size)
            draw = ImageDraw.Draw(area)
            right_edges = {}
            for active in sweeping:
                selected = [c for c in chars if active['first'] <= c['index'] < active['stop']]
                fraction = (instant-active['start'])/(active['end']-active['start'])
                remaining = sum(c['right']-c['left']+4 for c in selected) * fraction
                for char in selected:
                    width = char['right']-char['left']+4
                    right_edges[char['index']] = char['left']-2+min(width, remaining)
                    remaining = max(0., remaining-width)
            for char in chars:
                right = right_edges.get(char['index'], char['right']+2)
                if char['index'] < unit['stop'] and right > char['left']-2:
                    draw.rectangle((char['left']-2, char['top']-2, right, char['baseline']+8), fill=255)
            tint = Image.new('RGBA', result.size, self.color+(0,))
            tint.putalpha(ImageChops.multiply(area, self.layout['mask']))
            result.alpha_composite(tint)
        elif mode == 'underline':
            selected = [c for c in chars if unit['first'] <= c['index'] < unit['stop'] and not c['character'].isspace()]
            draw = ImageDraw.Draw(result)
            for line in sorted({c['line'] for c in selected}):
                group = [c for c in selected if c['line'] == line]
                left, right = min(c['left'] for c in group), max(c['right'] for c in group)
                y = group[0]['baseline']+10
                draw.rounded_rectangle((left-1, y-1, right+1, y+3), radius=2, fill=(13, 17, 21, 220))
                draw.rounded_rectangle((left, y, right, y+2), radius=1, fill=self.color+(255,))
        else:
            raise ValueError(mode)
        if cacheable:
            self.cache[key] = result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--video-only', action='store_true', help='Render the picture before final audio is ready')
    parser.add_argument('--cues', type=Path, required=True)
    parser.add_argument('--song-info', type=Path,
                        help='JSON with title, optional artist and version for opening credits')
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--mode', choices=['word', 'underline', 'static'], default='word')
    parser.add_argument('--style', choices=STYLES, default='cinema_focus')
    parser.add_argument('--language', choices=['Chinese', 'Japanese', 'Latin'], required=True)
    parser.add_argument('--bottom-margin', type=int, default=64)
    parser.add_argument('--next-line', action=argparse.BooleanOptionalAction, default=True,
                        help='Always show the next lyric line in white; advance at its first sung word')
    parser.add_argument('--countdown', action=argparse.BooleanOptionalAction, default=True,
                        help='Three-dot countdown before vocal entries following long gaps')
    parser.add_argument('--gain', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cues = json.loads(args.cues.read_text())
    if any(c['end'] <= c['start'] for c in cues) or any(a['end'] > b['start']+1e-5 for a, b in zip(cues, cues[1:])):
        raise ValueError('Invalid display intervals')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main_height = TOKENS['strip_height']
    preview_height = 64 if args.next_line else 0
    args.bottom_margin = max(args.bottom_margin, preview_height)
    width, height, fps = 1280, main_height+preview_height, 30
    intro = None
    intro_path = None
    if args.song_info:
        try:
            info = json.loads(args.song_info.read_text())
            first_onset = (cues[0]['units'][0]['start'] if cues and cues[0].get('units')
                           else cues[0].get('aligned_start', cues[0]['start']) if cues else float('inf'))
            end = min(TOKENS['intro']['maximum_seconds'], max(TOKENS['intro']['minimum_seconds'], first_onset))
            intro = {'information': info, 'start': 0., 'end': end, 'top': TOKENS['intro']['top'],
                     'policy': 'Credits above the always-visible lyrics; no audio or lyric timing offset',
                     'visible_in_this_export': args.start < end}
            if intro['visible_in_this_export']:
                intro_path = args.output.with_suffix('.intro.png')
                song_intro_image(info, width=width, style=args.style).save(intro_path)
        except Exception as error:
            intro_path = None
            intro = dict(status='omitted', error=type(error).__name__, reason=str(error),
                         visible_in_this_export=False)
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(args.start), '-i', str(args.video),
        '-f', 'rawvideo', '-pixel_format', 'rgba', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', 'pipe:0']
    if args.audio and not args.video_only:
        command += ['-ss', str(args.start), '-i', str(args.audio)]
    audio_index = 2 if args.audio and not args.video_only else 0
    if intro_path:
        intro_index = 3 if args.audio and not args.video_only else 2
        command += ['-loop', '1', '-framerate', str(fps), '-i', str(intro_path)]
    filters = (
        '[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,'
        'fps=30,setpts=PTS-STARTPTS[base];[1:v]setpts=PTS-STARTPTS[text];'
        f'[base][text]overlay=0:H-{main_height}-{args.bottom_margin}:format=auto')
    if intro_path:
        filters += (f'[captioned];[captioned][{intro_index}:v]overlay=0:{intro["top"]}:format=auto:'
                    f'enable=\'lt(t,{intro["end"]-args.start:.9f})\'')
    command += ['-filter_complex', filters+',format=yuv420p[v]', '-map', '[v]']
    if args.video_only:
        command += ['-an']
    else:
        command += ['-map', f'{audio_index}:a:0', '-af', f'volume={args.gain:.17g},asetpts=PTS-STARTPTS',
                    '-c:a', 'aac', '-b:a', '192k']
    command += [
        '-t', str(args.duration), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19',
        '-threads', '4', '-movflags', '+faststart', str(args.output)]
    started = time.perf_counter()
    blank = Image.new('RGBA', (width, height))
    cue_starts = [c['start'] for c in cues]
    countdown = countdown_events(cues, minimum_silence=TOKENS['countdown']['minimum_silence_seconds'],
        dots=TOKENS['countdown']['dots'], step_seconds=TOKENS['countdown']['step_seconds'],
        settle_seconds=TOKENS['countdown']['settle_seconds']) if args.countdown else []
    previous, caption = None, None
    preview_previous, preview_image = None, None
    shown, pagination = set(), {}
    log = args.output.with_suffix('.ffmpeg.log')
    with log.open('w') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in progress_items(range(round(args.duration*fps)), 'frames'):
                instant = args.start+frame/fps
                preview = None
                if args.next_line:
                    index, preview = read_ahead_state(cues, instant)
                else:
                    index = bisect_right(cue_starts, instant)-1
                    if index < 0 or instant >= cues[index]['end']:
                        index = None
                entry, remaining = countdown_state(countdown, instant)
                if entry is not None:
                    # The previous singing has ended; prepare the next lyric without highlighting it early.
                    index = entry['line']
                    preview = index+1 if args.next_line and index+1 < len(cues) else None
                picture = blank.copy() if preview_height else blank
                if index is not None:
                    if previous != index:
                        caption = Caption(cues[index], args.style, args.language)
                        if caption.pages:
                            pagination[index+1] = [dict(text=p.cue['text'], start=p.starts[0]) for p in caption.pages]
                        previous = index
                    current = caption.frame(instant, args.mode, page_at=entry['onset'] if entry is not None else None)
                    if preview_height:
                        picture.alpha_composite(current)
                    else:
                        picture = current
                    if entry is not None:
                        # Copy also in single-line mode so the cached caption remains unchanged.
                        picture = picture.copy()
                        picture.alpha_composite(caption.countdown(remaining, entry['dots']))
                    shown.add(index)
                if preview is not None:
                    if preview_previous != preview:
                        preview_image = compact_caption_image(cues[preview]['text'], height=preview_height,
                            style=args.style, language=args.language, style_overrides={
                                'size': 34, 'minimum_size': 26, 'single_line_floor': 34,
                                'maximum_lines': 1, 'balanced_wrap': False, 'baseline_inset': 22})
                        preview_previous = preview
                    picture.alpha_composite(preview_image, (0, main_height))
                process.stdin.write(picture.tobytes())
            process.stdin.close()
            report_progress(detail='封装视频')
            if process.wait() != 0:
                raise RuntimeError(log.read_text())
        except Exception:
            process.kill()
            process.wait()
            raise
    report_progress(detail='校验视频')
    if not args.video_only:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(args.output), '-f', 'null', '-'], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height', '-of', 'json', str(args.output)]))
    if abs(float(probe['format']['duration'])-args.duration) > .1:
        raise ValueError('Incorrect rendered duration')
    opening = None
    if countdown and countdown[0]['start'] < 0:
        geometry = Caption(cues[0], args.style, args.language).countdown_geometry
        opening = dict(countdown[0], **geometry, width=width, height=720)
        opening['y'] += 720-main_height-args.bottom_margin
    report = {'command': command, 'seconds': time.perf_counter()-started, 'start': args.start,
        'duration': args.duration, 'mode': args.mode, 'style': args.style, 'style_version': TOKENS['version'],
        'style_parameters': STYLES[args.style],
        'font': 'Bold Songti / Hiragino Mincho W6; source glyphs verified',
        'shown_lines': sorted(i+1 for i in shown),
        'display_pages': pagination,
        'decode_check': 'pending_final_mux' if args.video_only else 'passed', 'probe': probe,
        'motion': 'Fixed text positions; hard line cuts; linear left-to-right fill within each existing timing unit',
        'next_line_enabled': args.next_line,
        'read_ahead_strategy': 'Current and next lines visible; advance at sung onsets, with white preparation during long-gap countdowns',
        'song_intro': intro,
        'opening_countdown': opening,
        'countdown': {'enabled': args.countdown, 'events': countdown,
                      'minimum_silence_seconds': TOKENS['countdown']['minimum_silence_seconds'],
                      'policy': 'Three one-second dot steps, then empty rings held for one second before the acoustic onset'},
        'fallback_lines': [i+1 for i in sorted(shown) if not cues[i].get('units')],
        'estimated_lines': [i+1 for i in sorted(shown) if any(
            u.get('timing_source') == 'linear_fallback' for u in cues[i].get('units', []))]}
    args.output.with_suffix('.render.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
