"""Train a speaker identification model from RAVDESS audio files.

This script:
- Walks `data/ravdess/` recursively for .wav files
- Uses `src.predict.extract_all_features` to extract consistent features
- Trains a RandomForest classifier to predict speaker (Actor_xx)
- Saves artifacts into `models/`:
    - `speaker_model.pkl`
    - `speaker_label_encoder.pkl`
    - `speaker_scaler.pkl`
    - `speaker_feature_names.pkl`

Run from project root: python scripts/train_speaker.py
"""
import pickle
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from predict import extract_all_features


def gather_files(data_root: Path):
    files = list(data_root.rglob('*.wav'))
    return files


def speaker_from_path(p: Path):
    # we expect path like .../Actor_01/xxx.wav
    if p.parent.name.startswith('Actor'):
        return p.parent.name
    # fallback: try grandparent
    if p.parent.parent and p.parent.parent.name.startswith('Actor'):
        return p.parent.parent.name
    # otherwise, use filename prefix
    return p.stem.split('-')[-1]


def main():
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data' / 'ravdess'
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)

    print('Scanning for audio files in', data_root)
    files = gather_files(data_root)
    print('Found', len(files), 'wav files')

    records = []
    labels = []
    skipped = 0
    for i, f in enumerate(files, 1):
        try:
            feats = extract_all_features(str(f))
            if feats is None:
                skipped += 1
                continue
            records.append(feats)
            labels.append(speaker_from_path(f))
        except Exception as e:
            skipped += 1
            print(f"Skipping {f} due to error: {e}")

    print(f'Extracted features for {len(records)} files, skipped {skipped}')
    if len(records) == 0:
        raise SystemExit('No features extracted — aborting')

    df = pd.DataFrame(records)
    # fill NaNs
    df = df.fillna(0.0)

    # keep feature names
    feature_names = list(df.columns)
    X = df.values
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print('Training RandomForest classifier...')
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f'Validation accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    with open(models_dir / 'speaker_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(models_dir / 'speaker_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open(models_dir / 'speaker_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(models_dir / 'speaker_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print('Saved speaker model artifacts to', models_dir)


if __name__ == '__main__':
    main()
