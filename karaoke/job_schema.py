"""Public job inputs and canonical, single-video platform URLs."""
import re
from typing import Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        return {'zh': 'Chinese', 'ja': 'Japanese', 'en': 'English'}.get(value, value)
