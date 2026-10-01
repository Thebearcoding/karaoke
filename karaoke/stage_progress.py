"""Measured work counters shared by the pipeline's command-line stages."""
import json
import math
import os
import time

PREFIX = 'KARAOKE_STAGE_PROGRESS '
_last_time, _last_kind = 0., None


def report_progress(completed=None, total=None, unit=None, *, detail=None):
    global _last_time, _last_kind
    if os.getenv('KARAOKE_EVENTS_STDOUT') != '1':
        return
    instant, kind = time.monotonic(), (total, unit, detail)
    if kind == _last_kind and completed not in (0, total) and instant-_last_time < .5:
        return
    _last_time, _last_kind = instant, kind
    print(PREFIX + json.dumps(dict(completed=completed, total=total, unit=unit, detail=detail)), flush=True)


def progress_items(items, unit='chunks'):
    """Advance only after the caller has finished processing each item."""
    if os.getenv('KARAOKE_EVENTS_STDOUT') != '1':
        yield from items
        return
    total = len(items)
    report_progress(0, total, unit)
    for completed, item in enumerate(items, 1):
        yield item
        report_progress(completed, total, unit)


def parse_progress(line):
    if not line.startswith(PREFIX):
        return None
    try:
        value = json.loads(line[len(PREFIX):])
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    def number(key):
        candidate = value.get(key)
        return candidate if type(candidate) in (int, float) and math.isfinite(candidate) and candidate >= 0 else None
    completed, total = number('completed'), number('total')
    unit = value.get('unit')
    if unit not in ('bytes', 'chunks', 'blocks', 'lines', 'frames'):
        completed, total, unit = None, None, None
    if not total or completed is None or completed > total:
        total = None
    detail = value.get('detail')
    # yt-dlp downloads video and audio separately; their counters must not look
    # like a single percentage that jumps backwards.
    if value.get('vcodec'):
        detail = '音频' if value['vcodec'] == 'none' else '视频'
    detail = detail[:40] if isinstance(detail, str) else None
    return dict(completed=completed, total=total, unit=unit, detail=detail)
