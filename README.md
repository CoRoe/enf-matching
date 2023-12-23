# Electrical Network Frequency Analysis

GUI application based on example code from Robert Heaton on
["github"](https://github.com/robert/enf-matching).

## Run

To run the application:

```
./hum.py
```

In oder to run the test case:

- Select the audio file to be analysed by pressing the 'Load' button in the
  *Audio* group; then select the file *001.wav*. The file will be loaded and
  sample rate and duration be shown in the fields below the button.

- One the file is loaded, it has to be processed to get the ENF values. For
  the test file *001.wav*, leave the settings *nominal grid frquency*,
  *frequency band size* and *harmonic* at their default values. Press
  *analyse*. A red curve will appear in the plot area, displaying the ENF
  values over time.

- Load the grid frequencies. Leave all settings at their default values and
  press the *Load* button in the *Grid* group. A blue line will appear in the
  plot area, indicating the ENF values loaded from a hard-wired test file.

- Press the *Match* button.

![Screenshot](images/screenshot.png)

# Status

The application reproduces the test case outlined in
["github"](https://github.com/robert/enf-matching). The original reference
file is very long, and my humble Linux notebook ran out of memory during the
signal processing. I had to shorted the reference file, see the call to ffmpeg
in https://github.com/CoRoe/enf-matching/blob/main/bin/download-example-files.

Using actual grid frequency data is not yet implemented.

# See also

https://robertheaton.com/enf/

https://github.com/bellingcat/open-questions
