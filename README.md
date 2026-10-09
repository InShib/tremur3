# TremorSense One-Click

A compact, Speedtest-inspired tremor signal demo with a single **GO / START TEST** button and five feature tiles. The Flask backend loads the included TIM-Tremor X/Y `.npy` arrays, extracts signal features, trains a Random Forest on first launch, and predicts a dataset label for each sample window.

## Windows setup
Open this folder in VS Code and run each command in the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001**. First run trains the model and saves it under `model/`. If you already have a virtual environment from the earlier TremorSense folder, you may use that environment after installing `requirements.txt`.

## Demo behavior
Click **GO**. The backend selects a real 128×3 window from the included dataset, computes signal features, runs the model, and returns a predicted label plus the window for plotting. This is dataset playback, not live sensor input.

## Hardware API contract
`POST http://127.0.0.1:5001/api/test`
JSON body for real hardware:
```json
{"window":[[0.01,0.02,0.03], "... exactly 128 rows of numeric X,Y,Z ..."]}
```
Expected input shape: 128 samples × 3 axes. The hardware adapter may need to convert units, axis order, and sample rate to match dataset preprocessing. The dataset was processed at 50 Hz; one 128-sample window represents 2.56 seconds.

## Five displayed features
1. Dominant frequency (Hz)
2. RMS variation of the combined magnitude signal
3. Magnitude spread (standard deviation)
4. Axis variation (standard deviations for X, Y, Z)
5. Frequency-band energy is included in the model feature extraction (not presented as a single numeric tile)

## Important
This is a research/hackathon prototype, not a medical device. Labels are dataset classes, not a Parkinson's diagnosis. The reported single segment-held-out split is exploratory and is not clinical validation.
