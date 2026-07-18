#!/usr/bin/env bash

# Converts a ZMBV AVI capture directly to H.265/HEVC in a single ffmpeg pass.
# Usage: zmbv_to_h265_direct.sh <input_file> [output_file]

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file> [output_file]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}.mp4}"

ffmpeg -y -i "$INPUT" \
    -vf "format=rgb24,scale=iw*2:ih*2:flags=neighbor,format=yuv444p10le" \
    -c:v libx265 -crf 10 -pix_fmt yuv444p10le \
    -c:a aac -b:a 192k \
    "$OUTPUT"
