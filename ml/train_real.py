"""Train and export an honest ULB historical benchmark and held-out replay archive.

Source: OpenML dataset 1597 (ULB/Worldline anonymized 2013 card transactions).
No customer, device, country, or merchant fields are available in this dataset.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_recall_fscore_support

SOURCE_URL = 'https://www.openml.org/data/v1/download/1673544/creditcard.arff'
COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
FEATURES = COLUMNS[:-1]


def build(input_path: Path, output_dir: Path):
    df = pd.read_csv(input_path, skiprows=33, header=None, names=COLUMNS, low_memory=False)
    if list(df.columns) != COLUMNS or len(df) < 1000:
        raise ValueError('Unexpected ULB archive layout or row count')
    df['Class'] = df['Class'].astype(str).str.strip().str.strip("'").astype(int)
    if not bool(df['Class'].isin([0, 1]).all()):
        raise ValueError('Fraud label must be binary')
    for name in FEATURES:
        df[name] = pd.to_numeric(df[name], errors='raise')
    if df[FEATURES].isna().any().any():
        raise ValueError('Archive contains missing model features')
    if not df['Time'].is_monotonic_increasing:
        raise ValueError('Archive must be ordered by original time offset')
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df['Class'].to_numpy(dtype=np.int8)
    n = len(df)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    if min(y_train.sum(), y_val.sum(), y_test.sum()) == 0:
        raise ValueError('A time split contains no fraud cases')
    weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                              scale_pos_weight=weight, n_jobs=4, random_state=42,
                              eval_metric='aucpr')
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    test_scores = model.predict_proba(X_test)[:, 1]
    pred = (test_scores >= threshold).astype(int)
    p, r, test_f1, _ = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
    metrics = {'test_rows': int(len(y_test)), 'test_fraud_rows': int(y_test.sum()),
               'average_precision': float(average_precision_score(y_test, test_scores)),
               'roc_auc': float(roc_auc_score(y_test, test_scores)),
               'precision': float(p), 'recall': float(r), 'f1': float(test_f1),
               'threshold_selected_on_validation': threshold}
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / 'ulb_xgboost.json')
    metadata = {'source_name': 'ULB/Worldline anonymized credit card fraud archive',
                'source_url': SOURCE_URL, 'period': 'September 2013, two days',
                'mode': 'historical_holdout_replay', 'features': FEATURES,
                'train_rows': int(train_end), 'validation_rows': int(val_end - train_end),
                'metrics': metrics}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    database = sqlite3.connect(output_dir / 'replay.sqlite3')
    database.execute('DROP TABLE IF EXISTS replay_rows')
    database.execute('CREATE TABLE replay_rows (row_id INTEGER PRIMARY KEY, time_offset_seconds REAL NOT NULL, amount REAL NOT NULL, actual_fraud INTEGER NOT NULL, fraud_score REAL NOT NULL)')
    database.executemany('INSERT INTO replay_rows VALUES (?, ?, ?, ?, ?)',
        ((int(val_end + i), float(df.iloc[val_end + i]['Time']), float(df.iloc[val_end + i]['Amount']),
          int(y_test[i]), float(test_scores[i])) for i in range(len(y_test))))
    database.commit()
    database.close()
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/creditcard.arff'))
    parser.add_argument('--output', type=Path, default=Path('models/ulb_real'))
    args = parser.parse_args()
    build(args.input, args.output)
