#!/bin/sh

FFMPEG=/usr/bin/ffmpeg
YTDLP=/usr/local/bin/yt-dlp
DIR=../samplemedia

mkdir -p ${DIR}

# Download file from YouTube
${YTDLP} -o ${DIR}/Kate.webm https://www.youtube.com/watch?v=n1-RCm29wrA

# Create file with ENF pattern
python3 ./fm.py --width 1920 --height 1080 --grid 50 /tmp/enf.mp4

# Add ENF to the royal video, creating with 10% and one with 20% ENF
ffmpeg -i ${DIR}/Kate.webm -i /tmp/enf.mp4 -filter_complex \
       "mix=duration=shortest:weights=0.8 0.1" \
       -c:v libx264 -c:a copy -y -v quiet ${DIR}/Kate-10.mp4
ffmpeg -i ${DIR}/Kate.webm -i /tmp/enf.mp4 -filter_complex \
       "mix=duration=shortest:weights=0.8 0.2" \
       -c:v libx264 -c:a copy -y -v quiet ${DIR}/Kate-20.mp4

# Download file from YouTube.
${YTDLP} -o ${DIR}/selenski.webm https://www.youtube.com/watch?v=BEh1Fr-BaN8

# Convert to MP4, setting the frame rate to 
${FFMPEG} -i ${DIR}/selenski.webm -filter:v fps=30 -y -v quiet \
	  ${DIR}/selenski.mp4

# Create file with ENF pattern
python3 ./fm.py --width 1280 --height 720 --grid 50 /tmp/enf.mp4

# Add ENF to Selensky video, creating with 10% and one with 20% ENF
ffmpeg -i ${DIR}/selenski.mp4 -i /tmp/enf.mp4 -filter_complex \
       "mix=duration=shortest:weights=0.8 0.1" \
       -c:v libx264 -c:a copy -y -v quiet ${DIR}/selensky-10.mp4
ffmpeg -i ${DIR}/selenski.mp4 -i /tmp/enf.mp4 -filter_complex \
       "mix=duration=shortest:weights=0.8 0.2" \
       -c:v libx264 -c:a copy -y -v quiet ${DIR}/selensky-20.mp4

rm /tmp/enf.mp4
