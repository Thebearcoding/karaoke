"""Download a second stream from fresh metadata without re-querying the site."""
import json
from pathlib import Path

from yt_dlp import YoutubeDL, parse_options


def main():
    options = parse_options()
    metadata = json.loads(Path(options.options.load_info_filename).read_text())
    options.ydl_opts['ignoreerrors'] = False
    with YoutubeDL(options.ydl_opts) as downloader:
        # download_with_info_file silently re-extracts on failure, including
        # access failures. Let the caller preserve the original 429/login hold.
        downloader.process_ie_result(metadata, download=True)


if __name__ == '__main__':
    main()
