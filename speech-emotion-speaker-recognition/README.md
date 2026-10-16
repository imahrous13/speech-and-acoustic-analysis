# Speech Emotion and Speaker Recognition

## Overview

This project classifies emotion and speaker identity from recorded speech. It includes a RAVDESS feature-extraction and model-comparison notebook, serialized inference artifacts, a reusable prediction module, a speaker-model training script, and a Streamlit interface for uploaded audio.

The emotion pipeline compares five classical machine-learning classifiers across eight RAVDESS emotion labels. The speaker pipeline trains a Random Forest to predict the actor directory associated with each recording.

## Features

- Accepts WAV, MP3, and FLAC uploads through a Streamlit interface.
- Predicts one of eight emotions and displays per-class probabilities.
- Provides an alternate speaker-identification mode when speaker artifacts are available.
- Extracts 191 aggregate acoustic features from each audio file.
- Caches extracted emotion features in CSV format to avoid repeating audio processing.
- Compares SVM, Random Forest, XGBoost, Gradient Boosting, and Logistic Regression models.
- Saves models, scalers, label encoders, and ordered feature names for inference.

## Tech Stack

- **Language:** Python
- **Audio processing:** librosa, SoundFile
- **Machine learning:** scikit-learn, XGBoost
- **Data processing:** NumPy, pandas, PyArrow
- **Visualization:** Matplotlib, Seaborn
- **Application:** Streamlit
- **Persistence:** CSV and Python pickle artifacts
- **Dataset included in the repository:** RAVDESS speech audio (stored twice under two directory layouts; see limitations)

## Architecture / How It Works

The emotion-training notebook recursively reads RAVDESS audio and derives labels from the third field of each RAVDESS filename. Each recording is resampled to 22,050 Hz and represented by 191 time-averaged features:

- 40 MFCC values
- 12 chroma values
- 128 mel-spectrogram bands
- spectral centroid and bandwidth
- 7 spectral-contrast values
- zero-crossing rate and RMS energy

The notebook encodes labels, creates stratified train/test and train/validation splits, standardizes the feature matrix, evaluates five classifiers by weighted validation F1, and serializes the selected model and preprocessing objects.

For inference, `src/predict.py` repeats the same feature extraction, restores the saved feature order, scales the vector, and returns the decoded class plus probabilities. `app/demo.py` stores an uploaded file temporarily, calls either the emotion or speaker predictor, and renders the result and probability distribution.

Speaker training is a separate script. It derives a speaker label from each `Actor_*` directory, uses the shared acoustic features, trains a 200-tree Random Forest on a stratified 80/20 split, and writes four speaker-specific artifacts.

## Technical Highlights

- Maintains feature parity between notebook training and application inference through a shared ordered feature-name artifact.
- Implements an end-to-end audio ML workflow: dataset traversal, feature extraction, preprocessing, model selection, evaluation, serialization, and interactive inference.
- Benchmarks several model families under the same preprocessing and validation split before selecting the best weighted-F1 result.
- Supports two related classification tasks through one prediction module and Streamlit UI.
- Reuses a cached 2,880-row feature table, reducing the need to reprocess all audio on every notebook run.

## Setup

Use a Python environment capable of installing the versions declared in `requirements.txt`.

```bash
cd speech-emotion-speaker-recognition
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app/demo.py
```

The repository already contains the saved emotion and speaker artifacts expected under `models/`, so retraining is not required to launch inference. Pickle files should only be loaded from a trusted source.

To rerun the emotion experiment:

```bash
python -m pip install jupyter
jupyter notebook notebooks/complete_pipeline.ipynb
```

Start Jupyter from `notebooks/` or ensure the notebook's `../data` and `../models` paths resolve to the project directories.

To retrain the speaker model from the included RAVDESS files:

```bash
python scripts/train_speaker.py
```

## Usage

In the Streamlit app, select **Emotion** or **Speaker ID**, upload an audio file, and run prediction. The application shows the predicted label, a probability chart, detailed probabilities, and an audio player.

The reusable Python entry points are `predict_emotion(audio_path)` and `predict_speaker(audio_path)` in `src/predict.py`.

## Project Structure

```text
speech-emotion-speaker-recognition/
├── app/demo.py                         # Streamlit upload and prediction UI
├── data/features.csv                   # Cached emotion features and labels
├── data/ravdess/                       # Included RAVDESS WAV recordings
├── models/                             # Serialized models and preprocessing artifacts
├── notebooks/complete_pipeline.ipynb  # Emotion feature, training, and evaluation workflow
├── scripts/train_speaker.py            # Speaker Random Forest training pipeline
├── scripts/smoke_speaker_test.py       # Runs speaker inference on an included sample
├── src/predict.py                      # Shared extraction and inference functions
└── requirements.txt                    # Runtime and notebook dependencies
```

## Results / Metrics

The executed emotion notebook records results for 2,880 feature rows, 191 features, and eight classes. Its stratified test split contains 576 rows. Repository inspection shows that the audio tree contains 1,440 unique filenames stored twice under two directory layouts; each same-name pair has identical SHA-256 content. The cached feature table contains both copies, so the following scores should be treated as notebook-recorded results from a duplicate-contaminated split, not as reliable generalization estimates.

- Selected model: XGBoost
- Validation weighted F1: 0.8615
- Test accuracy: 0.8264
- Test weighted precision: 0.8262
- Test weighted recall: 0.8264
- Test weighted F1: 0.8250

The same saved notebook records validation weighted-F1 scores of 0.5934 for SVM, 0.8371 for Random Forest, 0.8436 for Gradient Boosting, and 0.6369 for Logistic Regression. These values are notebook outputs, not results independently reproduced during README creation.

The repository contains trained speaker artifacts, but it does not retain the speaker model's evaluation output; no speaker-identification accuracy is claimed here.

## Limitations / Status

- The 2,880-row feature table is built from two identical copies of each of 1,440 recordings. A random row-level split can place the same recording in training, validation, and test sets, creating direct data leakage and likely inflating the recorded metrics.
- Emotion evaluation is also not speaker-independent. Recordings from the same actors may occur across splits, so the metrics do not establish generalization to unseen speakers.
- Model selection uses a held-out validation subset, but the repository has no automated test suite or reproducible environment lock file.
- The notebook has two visible code defects in later demonstration/exploration cells: `preditct_emotion` is misspelled, and `mf---cc_1` is not a valid feature name. Stored outputs appear to come from an earlier state, so a clean run may fail at the prediction example.
- The plotting cells call `show()` before `savefig()`, and the referenced `results/` images are not present in the repository.
- `scripts/smoke_speaker_test.py` is a manual inference check rather than an assertion-based automated test.
- Pickle artifacts are Python/library-version sensitive and unsafe to load from untrusted sources.
- The app accepts several file extensions but has no explicit upload-size or duration limit.

## Resume Evaluation Notes

- **Strongest technical aspects:** Complete handcrafted audio-feature ML pipeline, multi-model comparison, serialized preprocessing, and a working inference UI for two classification tasks.
- **Most impressive implementation:** Consistent 191-feature extraction and artifact-backed inference from uploaded audio through Streamlit.
- **Weak/basic parts:** Duplicate-contaminated evaluation data, no automated tests, no speaker-independent evaluation, notebook defects, duplicated feature-extraction code, and no locked dependency versions.
- **Verified metrics:** The notebook records 82.64% test accuracy and 0.8250 weighted test F1 on 576 rows, but the split contains exact duplicate recordings and should not be presented as a valid generalization benchmark; validation scores for five candidate models are also retained.
- **What is unique:** One application combines RAVDESS emotion classification with actor/speaker identification using a shared acoustic representation.
- **What may be worth highlighting on a resume:** Audio feature engineering with librosa, comparative ML experimentation, model serialization, and deployment of inference through Streamlit—qualified as coursework and with the evaluation protocol stated.
- **Claims that cannot be verified from the repo:** Real-world performance, unseen-speaker generalization, production scale, user adoption, latency, and speaker-model accuracy.
