# ENF Matching

## Setup

```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
Download sample files from https://github.com/ghuawhu/ENF-WHU-Dataset:
```
./bin/download-examnple-files
ffmpeg -ss 71000 -i 001_ref.wav -c copy 71000_ref.wav
```

## Run

To run the GUI version:

```
./hum.py
```

In oder to run the test case:

- Select the audio file to be analysed. Press the 'Load' button in the 'Audio'
  group and select the file '001.wav'. The file will be loaded and sample rate
  and duration be shown.

- Load the grid frequencies. Leave all settings at their default values and
  press the 'Load' button in the 'Grid' group.

- Press the 'Analyse' button.

To run the reference code from RH:

```
./main.py 001.wav -r 71000_ref.wav
```
Should output:

```
<snip>
True value is 71458
Best prediction is 460
```
