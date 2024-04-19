Just a number of scripts, mostly in *statu nascendi*.

# fm.py

Generates a video test file. The video signal mimics a light source whose
luminosity is proportional to the rectified current grid voltage recorded by a
camera with global shutter. The simulated grid voltage is slightly shifted to
simulate grid frequency changes.

The pattern of the frequency deviation is according to the following table:

```
    # Format: Duration (seconds), frequency deviation factor
    freq_deltas = (
        (20, +0.002),
        (20, -0.002),
        (30,  0.000),
        (30, -0.003),
        (20, +0.003)
    )
```
