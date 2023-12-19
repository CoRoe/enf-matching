#!/usr/bin/env bash

curl \
  -H "Accept: application/vnd.github.VERSION.raw" \
  -o 001_ref.wav \
  https://raw.githubusercontent.com/ghuawhu/ENF-WHU-Dataset/master/ENF-WHU-Dataset/H1_ref_one_day/001_ref.wav
curl \
  -H "Accept: application/vnd.github.VERSION.raw" \
  -o 001.wav \
  https://raw.githubusercontent.com/ghuawhu/ENF-WHU-Dataset/master/ENF-WHU-Dataset/H1/001.wav

# Create a shorter reference file that starts at sample number 71000. The
# original file caused my machine to run aout of memory during one of the
# transforms.
ffmpeg -ss 71000 -i 001_ref.wav -c copy 71000_ref.wav
