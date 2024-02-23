#
# Read video file, extract luminance signal.
#

import subprocess
import re
import numpy as np


# Read piped output: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command

def checkFormat(filename):
    cmd = ["/usr/bin/ffprobe", filename]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    output, errors = p.communicate()
    print("Output:", output)
    print("Errors:", errors)
    
    # Check if fps is always a float
    videoprops = re.compile(r'  Stream.+: Video:.+, (\d+)x(\d+),.+ (\d+\.\d+) fps,')
    m = videoprops.search(errors)
    if m:
        hres = int(m.group(1))
        vres = int(m.group(2))
        fps = float(m.group(3))
        return hres, vres, fps
    else:
        return None, None, None


def readYUV(filename, hres, vres):
    # fmpeg -i test.mp4 -pix_fmt yuyv422 -f rawvideo
    bytes_per_frame = hres * vres * 2
    per_frame_luminance = np.empty((0, ), dtype=np.uint16)

    cmd = ['/usr/bin/ffmpeg', '-i', filename, '-pix_fmt', 'yuyv422', '-f', 'rawvideo', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
    while True:
        data = proc.stdout.read(bytes_per_frame)
        if len(data) == 0:
            break
        frames = np.frombuffer(data, dtype=np.uint8)
        deinterleaved = [frames[idx::2] for idx in range(2)]
        # Luminance data for one frame
        lum = deinterleaved[0]
        l = np.uint16(np.average(lum) * 256)
        per_frame_luminance = np.append(per_frame_luminance, l)
    return per_frame_luminance


if __name__ == '__main__':
    filename = 'test.mp4'
    h, v, fps = checkFormat(filename)
    readYUV(filename, h, v)
    print(h, v, fps)
