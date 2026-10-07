"""Public job inputs, safe provider errors and single-video platform URLs."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
from typing import Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


def retry_after_seconds(value):
    """Retain a server's retry hint without exposing response headers or body."""
    hints = []
    headers = getattr(getattr(value, 'response', None), 'headers', {})
    for name, scale in (('retry-after-ms', .001), ('retry-after', 1.)):
        if name not in headers:
            continue
        try:
            hints.append(float(headers[name])*scale)
        except ValueError:
            try:
                hints.append((parsedate_to_datetime(headers[name])-datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    for amount, unit in re.findall(r'try again in\s+(\d+(?:\.\d+)?)\s*(ms|seconds?|s|minutes?|m)\b', str(value), re.I):
        hints.append(float(amount)*(.001 if unit.lower() == 'ms' else 60 if unit.lower().startswith('m') else 1))
    return max((v for v in hints if math.isfinite(v) and v >= 0), default=None)


def provider_error(value):
    """Classify SDK exceptions (or their log lines) without exposing response data."""
    if isinstance(value, BaseException):
        names = [type(value).__name__] if type(value).__module__.startswith('openai') else []
    else:
        names = re.findall(r'^openai\.(\w+Error):', str(value), re.M)
    known = {
        'APITimeoutError': ('timeout', '语音或歌词服务响应超时，请稍后重试', True),
        'APIConnectionError': ('connection', '无法连接语音或歌词服务，请稍后重试', True),
        'RateLimitError': ('rate_limit', '语音或歌词服务额度或频率受限，请稍后重试', True),
        'ContentFilterFinishReasonError': ('content_rejected', '歌词服务拒绝处理本次内容，需要人工检查', False),
        'AuthenticationError': ('authentication', '语音或歌词服务认证失败，请检查服务配置', False),
        'PermissionDeniedError': ('permission', '语音或歌词服务拒绝访问，请检查服务权限', False),
    }
    for name in reversed(names):
        if name in known:
            code, message, retryable = known[name]
            text = str(value)
            if name == 'RateLimitError' and 'request too large' in text.lower():
                code, message, retryable = 'request_too_large', '请求超过模型或项目限制，需要调整请求或配置', False
            elif name == 'RateLimitError' and 'insufficient_quota' in text.lower():
                code, message, retryable = 'quota', '语音或歌词服务额度不足，需要检查项目额度', False
            result = dict(code='openai_'+code, message=message, provider='openai', retryable=retryable)
            delay = retry_after_seconds(value)
            if delay is not None:
                result['retry_after_seconds'] = delay
            model = re.search(r'for ([\w.-]+) in (?:project|organization)', text)
            if model:
                result['model'] = model.group(1)
            return result
    return None


def song_language(value):
    key = str(value or '').lower().replace('_', '-').split('-')[0]
    return {'zh': 'Chinese', 'chinese': 'Chinese', 'ja': 'Japanese', 'japanese': 'Japanese',
            'en': 'English', 'english': 'English'}.get(key)


def thumbnail_url(value):
    """Only display artwork served by the supported source platforms."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme == 'https' and not parsed.username and not parsed.password
                and parsed.port in (None, 443) and any(
                    parsed.hostname == host or (parsed.hostname or '').endswith('.'+host)
                    for host in ('ytimg.com', 'googleusercontent.com', 'hdslb.com'))):
            return value
    except ValueError:
        pass
    return None


def video_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
        raise ValueError('请输入 YouTube 或 Bilibili 的 HTTP(S) 视频链接')
    if parsed.port not in (None, 80, 443):
        raise ValueError('视频链接不支持自定义端口')
    host = (parsed.hostname or '').lower()
    parts = parsed.path.strip('/').split('/')
    query = parse_qs(parsed.query)
    if host in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be'):
        identifier = (parts[0] if host == 'youtu.be' else
                      query.get('v', [''])[0] if parsed.path == '/watch' else
                      parts[1] if len(parts) == 2 and parts[0] in ('shorts', 'live', 'embed') else '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', identifier):
            raise ValueError('请提供单个 YouTube 视频链接，而非频道或播放列表')
        return 'https://www.youtube.com/watch?v=' + identifier
    if host in ('bilibili.com', 'www.bilibili.com', 'm.bilibili.com'):
        if len(parts) != 2 or parts[0] != 'video' or not re.fullmatch(r'(?:BV[0-9A-Za-z]{10}|av[0-9]+)', parts[1]):
            raise ValueError('请提供 /video/BV… 或 /video/av… 的 Bilibili 视频链接')
        page = query.get('p', ['1'])[0]
        if not page.isdigit() or not 1 <= int(page) <= 10000:
            raise ValueError('无效的 Bilibili 分 P 编号')
        return f'https://www.bilibili.com/video/{parts[1]}' + (f'?p={int(page)}' if int(page) != 1 else '')
    if host == 'b23.tv' and len(parts) == 1 and re.fullmatch(r'[0-9A-Za-z]{4,32}', parts[0]):
        return f'https://b23.tv/{parts[0]}'
    raise ValueError('当前只接受 YouTube、Bilibili 和 b23.tv 视频链接')


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(max_length=2048)
    lyrics: list[str] | None = Field(default=None, max_length=1000)
    title: str | None = Field(default=None, max_length=200)
    artist: str | None = Field(default=None, max_length=200)
    language: Literal['Chinese', 'Japanese', 'English'] | None = None
    mv_audio: Literal['instrumental_backing', 'instrumental'] = 'instrumental_backing'
    force: bool = False

    @field_validator('url')
    @classmethod
    def canonical_url(cls, value):
        return video_url(value)

    @field_validator('lyrics', mode='before')
    @classmethod
    def lyric_lines(cls, value):
        if isinstance(value, str):
            value = value.splitlines()
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(line, str) for line in value):
            raise ValueError('lyrics 应为分行文本或字符串数组')
        lines = [line.strip() for line in value if line.strip()]
        if not lines or any(len(line) > 200 for line in lines) or sum(map(len, lines)) > 50000:
            raise ValueError('歌词不能为空，每行最多 200 字符，总计最多 50000 字符')
        return lines

    @field_validator('language', mode='before')
    @classmethod
    def language_alias(cls, value):
        return song_language(value) or value
