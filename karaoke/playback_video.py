"""Package the unchanged picture with silent AAC for native iPad playback.

The audible music still comes exclusively from the separate playback stems.
"""
import argparse
import subprocess


def mux_playback_video(source, output):
    # A video-only MP4 stutters on the tested iPad/TV output path. An audio
    # track lets the native media player present smoothly, even while muted.
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '2',
        '-i', str(source), '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '32k',
        '-shortest', '-movflags', '+faststart', str(output)], check=True, timeout=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    mux_playback_video(args.source, args.output)
