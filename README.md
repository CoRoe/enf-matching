# Electrical Network Frequency Analysis

GUI application based on example code from Robert Heaton on
[github](https://github.com/robert/enf-matching).

## Run

To run the application:

```
./hum.py
```

In oder to run the test case:

- Select the audio file to be analysed by pressing the 'Load' button in the
  *Audio* group; then select the file *001.wav*. The file will be loaded and
  sample rate and duration be shown in the fields below the button.

- Once the file is loaded, it has to be processed to get the ENF values. For
  the test file *001.wav*, leave the settings *nominal grid frquency*,
  *frequency band size* and *harmonic* at their default values. Press
  *analyse*. A red curve will appear in the plot area, displaying the ENF
  values over time.

  ![Screenshot](images/screenshot-clip-enf.png)

  The screenshot shows that the 50 Hz component varies slowly albeit with some
  outliers.

  ![Screenshot](images/screenshot-clip-spectrum.png)

  Here, the clip spectrum is shown.

  ![Screenshot](images/screenshot-bad-param.png)

  Here, a different harmonic has been chosen. No clear 50 Hz component is
  present in the diagram.

- Load the grid frequencies. Leave all settings at their default values and
  press the *Load* button in the *Grid* group. A blue line will appear in the
  plot area, indicating the ENF values loaded from a hard-wired test file.

- Press the *Match* button.

  ![Screenshot](images/screenshot-matched.png)

  The diagram shows a good match between the ENF pattern in the audio clip and
  the (test) grid ENF.

# Testing

There a some unit test cases for `griddata.py`. To run them:

```
pytest-3 test.py
```

# Status

Getting actual ENF values from grid operator is implemented for Great Britain
and Finland.

Once downloaded from the internet, the extracted ENF series are stored in an
sqlite database; its filename can be changed in the *settings* dialog.

The application reproduces the test case outlined in
[github](https://github.com/robert/enf-matching). The original reference
file is very long, and my humble Linux notebook ran out of memory during the
signal processing. I had to shorted the reference file, see the call to ffmpeg
in https://github.com/CoRoe/enf-matching/blob/main/bin/download-example-files.

# See also

https://robertheaton.com/enf/

https://github.com/bellingcat/open-questions/tree/main/electrical-network-frequency-analysis
