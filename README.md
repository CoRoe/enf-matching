# Electrical Network Frequency Analysis

GUI application based on example code from Robert Heaton on
[github](https://github.com/robert/enf-matching).

## Run

To run the application:

```
python3 hum.py
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

  The plot shows the the clip spectrum.

  ![Screenshot](images/screenshot-bad-param.png)

  Here, a different harmonic has been chosen. No clear 50 Hz component is
  present in the diagram.

- Load the grid frequencies. Select *Test* as location and press the *Load*
  button in the *Grid* group. A blue line will appear in the plot area,
  indicating the ENF values loaded from a hard-wired test file.

- Leave the metric setting at *Euclidian* and press the *Match* button.

  ![Screenshot](images/screenshot-matched.png)

  The diagram shows a good match between the ENF pattern in the audio clip and
  the (test) grid ENF.

## Testing

There a some unit test cases for `griddata.py`. To run them:

```
pytest-3 test.py
```

## Design Details

- `hum` uses `ffmpeg` to convert media files to the WAV format. The reduce
  memory requirements, the file is downsampled to 4000 Hz. `hum` loads the
  entire WAV file into memory; very long files may crash the application.

- Grid data are handled with a granularity of one month. The original reason
  was that operator provide the data as one file per month; `hum` also stores
  ENF data as one record per month in the SQL database.

  Up to 12 months ENF data can be used. This limit is set to limit memory
  usage and processing time.

- ENF data are typically provided as CSV files in a format like:

```
Timestamp, frequency
2019-04-01 00:00:00,50.001
2019-04-01 00:00:01,59.997
2019-04-01 00:00:02,59.998
```

  The timestamps are evenly spaced at 1 second (Fingrid 100 ms). `hum` assumes
  that timestamps start at 00:00:00 of a month and are contiguous without
  gaps.

  In reality, there are a few gaps so that the time match may be off by a few
  seconds.

## Status

Getting actual ENF values from grid operator is implemented for Great Britain
and Finland. Fingrid have recently changed their API, `hum` has not yet been
updated.

Once downloaded from the internet, the extracted ENF series are stored in an
sqlite database; its filename should be set in the *settings* dialog.

Input files are always fed into `ffmpeg` for conversion to a WAV file; at the
same time the sample rate is reduced to 4000 Hz. All file type supported by
`ffmpeg` are hence also supported by hum.

The matching process takes several minutes; its time complexity is (clip
duration in seconds * number of seconds in a the month). Currently, the
operation is single-theaded.

Error handlng is only rudimentary. When the program runs out of memory, it
silently crashes. Catching `MemoryError`s does not work.

The application reproduces the test case outlined in
[github](https://github.com/robert/enf-matching). The original reference
file is very long, and my humble Linux notebook ran out of memory during the
signal processing. I had to shorted the reference file, see the call to ffmpeg
in https://github.com/CoRoe/enf-matching/blob/main/bin/download-example-files.

## See also

https://robertheaton.com/enf/

https://github.com/bellingcat/open-questions/tree/main/electrical-network-frequency-analysis
