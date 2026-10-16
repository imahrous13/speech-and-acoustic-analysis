# Sound Level and Distance Analysis

## Overview

This notebook analyzes how measured sound pressure level changes with distance for five test frequencies: 500 Hz, 1 kHz, 2 kHz, 4 kHz, and 8 kHz. It visualizes the measurements in decibels, converts them to sound pressure in pascals, and discusses how the observed patterns compare with ideal distance-based attenuation.

The project is a compact acoustics analysis rather than a speech-recognition model. Its six measurement positions and all recorded values are embedded directly in the notebook.

## Features

- Organizes sound-level measurements from 0 to 3 meters in a pandas DataFrame.
- Plots decibel level against distance for five frequencies.
- Converts dB SPL to pressure using `p = p0 * 10^(L/20)` with `p0 = 20 µPa`.
- Plots pressure against distance and provides a combined comparison figure.
- Reports the measured change between 0 and 3 meters for each frequency.

## Tech Stack

- Python
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib

## Architecture / How It Works

The notebook follows a linear analysis workflow:

1. Construct a DataFrame from the embedded distance and frequency measurements.
2. Select all decibel columns and plot each frequency as a separate series.
3. Compare the first and last readings to summarize attenuation.
4. Convert every dB SPL value to pascals using the standard 20 µPa reference pressure.
5. Plot the converted pressure values and display both views side by side.

No external data files, APIs, trained models, or services are used.

## Technical Highlights

- Applies the logarithmic dB SPL-to-pressure conversion consistently across multiple frequency series.
- Presents the same measurements in logarithmic and linear units, making the physical relationship easier to inspect.
- Retains executed outputs and plots in the notebook for review without rerunning it.

## Setup

From this directory, create an environment and install the notebook dependencies:

```bash
python -m venv .venv
python -m pip install jupyter numpy pandas matplotlib
jupyter notebook sound-level-distance-analysis.ipynb
```

Activate the virtual environment first if your system requires it. Run the notebook cells from top to bottom.

## Usage

Open `sound-level-distance-analysis.ipynb`. To analyze another experiment, replace the values in the DataFrame construction cell while preserving the distance column and `(dB)` suffix used to select frequency columns.

## Project Structure

```text
sound-propagation-analysis/
├── README.md
└── sound-level-distance-analysis.ipynb  # Data, conversion, analysis, and plots
```

## Results / Metrics

The executed notebook contains six observations per frequency. From 0 to 3 meters, the recorded levels decrease by:

- 500 Hz: 21.5 dB
- 1 kHz: 32.8 dB
- 2 kHz: 35.2 dB
- 4 kHz: 34.5 dB
- 8 kHz: 68.6 dB

These are descriptive changes in the embedded measurements, not model-performance metrics. The notebook also records the converted endpoint pressures; for example, the 500 Hz series changes from approximately 0.005445 Pa to 0.000458 Pa.

## Limitations / Status

- The dataset contains only six distances and does not include repeated trials, uncertainty estimates, or measurement metadata.
- Measurements at 0 meters and uncontrolled acoustic effects make a strict inverse-square-law validation inappropriate.
- The notebook discusses inverse-square behavior qualitatively but does not fit a theoretical attenuation curve or calculate goodness of fit.
- Data provenance and collection conditions are not documented in the repository.
- This is a completed exploratory coursework notebook, not a reusable package or automated experiment pipeline.

## Resume Evaluation Notes

- **Strongest technical aspects:** Correct dB SPL-to-pressure conversion and clear multi-frequency visualization.
- **Most impressive implementation:** A single workflow presents the same acoustic measurements in logarithmic dB and linear pressure units.
- **Weak/basic parts:** Small hard-coded dataset, no statistical testing, no reusable modules, and no automated tests.
- **Verified metrics:** Six observations across five frequencies; recorded 0-to-3-meter decreases range from 21.5 dB to 68.6 dB.
- **What is unique:** Frequency-specific attenuation is compared in both dB SPL and pascals.
- **What may be worth highlighting on a resume:** Python-based acoustic data analysis and visualization, if relevant to a signal-processing or audio role.
- **Claims that cannot be verified from the repo:** Measurement procedure, equipment accuracy, experimental repeatability, and general compliance with the inverse-square law.
