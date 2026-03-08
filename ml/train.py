"""
XGBoost Fraud Detection Model Training Pipeline
================================================
Trains, optimizes, and exports XGBoost model for real-time fraud scoring.
Includes hyperparameter tuning, calibration, and model evaluation.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from features import FeatureExtractor, extract_features_from_dataframe


class FraudDetectionModel:
    """
    XGBoost-based fraud detection model with calibration.
    Optimized for low-latency inference and high precision.
    """
    
    def __init__(self, model_dir: str = 'models'):
        """Initialize model with default configuration."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names: list = []
        self.metadata: Dict = {}
        
        # Default XGBoost parameters (optimized for fraud detection)
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1,  # Will be computed based on class imbalance
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Fast histogram-based algorithm
            'random_state': 42,
            'n_jobs': -1,
        }
    
    def prepare_features(
        self, 
        df: pd.DataFrame,
        user_profiles_df: Optional[pd.DataFrame] = None,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """Prepare features from transaction DataFrame."""
        
        # Extract features
        X, feature_names = extract_features_from_dataframe(df, user_profiles_df)
        self.feature_names = feature_names
        
        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict] = None,
        calibrate: bool = True,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the fraud detection model.
        
        Args:
            X: Feature matrix
            y: Target labels (0=legitimate, 1=fraud)
            params: XGBoost parameters (uses defaults if None)
            calibrate: Whether to calibrate probabilities
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        print("=" * 60)
        print("FRAUD DETECTION MODEL TRAINING")
        print("=" * 60)
        
        # Compute class weight
        fraud_ratio = y.sum() / len(y)
        scale_pos_weight = (1 - fraud_ratio) / fraud_ratio
        print(f"\nClass distribution:")
        print(f"  - Legitimate: {(1-fraud_ratio)*100:.2f}%")
        print(f"  - Fraud: {fraud_ratio*100:.2f}%")
        print(f"  - Scale pos weight: {scale_pos_weight:.2f}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        print(f"\nDataset sizes:")
        print(f"  - Training: {len(X_train):,}")
        print(f"  - Validation: {len(X_val):,}")
        
        # Prepare parameters
        model_params = self.default_params.copy()
        model_params['scale_pos_weight'] = scale_pos_weight
        if params:
            model_params.update(params)
        
        # Train XGBoost model
        print(f"\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(**model_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Calibrate probabilities (optional but recommended)
        if calibrate:
            print("\nCalibrating probabilities...")
            self.calibrator = CalibratedClassifierCV(
                self.model, method='isotonic', cv='prefit'
            )
            self.calibrator.fit(X_val, y_val)
        
        # Evaluate model
        metrics = self._evaluate(X_val, y_val)
        
        # Store metadata
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_features': X.shape[1],
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'fraud_ratio': float(fraud_ratio),
            'scale_pos_weight': float(scale_pos_weight),
            'params': model_params,
            'metrics': metrics,
            'feature_names': self.feature_names,
        }
        
        return metrics
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y, y_pred_proba)
        print(f"\nROC-AUC: {roc_auc:.4f}")
        
        # Precision-Recall AUC (better for imbalanced datasets)
        pr_auc = average_precision_score(y, y_pred_proba)
        print(f"PR-AUC: {pr_auc:.4f}")
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"  - Precision at threshold: {precision[optimal_idx]:.4f}")
        print(f"  - Recall at threshold: {recall[optimal_idx]:.4f}")
        print(f"  - F1 at threshold: {f1_scores[optimal_idx]:.4f}")
        
        # Classification report at optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        print(f"\nClassification Report (threshold={optimal_threshold:.3f}):")
        print(classification_report(y, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Business metrics
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn)
        false_negative_rate = fn / (fn + tp)
        
        print(f"\nBusiness Metrics:")
        print(f"  - False Positive Rate: {false_positive_rate*100:.2f}%")
        print(f"  - False Negative Rate: {false_negative_rate*100:.2f}%")
        print(f"  - Fraud Catch Rate: {tp/(tp+fn)*100:.2f}%")
        
        # Feature importance
        print("\nTop 15 Feature Importances:")
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        metrics = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'optimal_threshold': float(optimal_threshold),
            'precision_at_threshold': float(precision[optimal_idx]),
            'recall_at_threshold': float(recall[optimal_idx]),
            'f1_at_threshold': float(f1_scores[optimal_idx]),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'confusion_matrix': cm.tolist(),
        }
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get fraud probability predictions."""
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, name: str = 'fraud_model') -> str:
        """Save model and all components."""
        save_path = self.model_dir / name
        save_path.mkdir(exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(save_path / 'xgboost_model.json')
        
        # Save calibrator
        if self.calibrator:
            joblib.dump(self.calibrator, save_path / 'calibrator.joblib')
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, save_path / 'scaler.joblib')
        
        # Save metadata
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\nModel saved to: {save_path}")
        return str(save_path)
    
    def load(self, path: str) -> 'FraudDetectionModel':
        """Load model from disk."""
        load_path = Path(path)
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(load_path / 'xgboost_model.json')
        
        # Load calibrator
        calibrator_path = load_path / 'calibrator.joblib'
        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
        
        # Load scaler
        scaler_path = load_path / 'scaler.joblib'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        with open(load_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata.get('feature_names', [])
        
        print(f"Model loaded from: {load_path}")
        return self
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict:
        """Perform cross-validation to estimate model performance."""
        print(f"\nPerforming {n_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        roc_aucs = []
        pr_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**self.default_params)
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            pr_auc = average_precision_score(y_val, y_pred_proba)
            
            roc_aucs.append(roc_auc)
            pr_aucs.append(pr_auc)
            
            print(f"  Fold {fold+1}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        results = {
            'roc_auc_mean': float(np.mean(roc_aucs)),
            'roc_auc_std': float(np.std(roc_aucs)),
            'pr_auc_mean': float(np.mean(pr_aucs)),
            'pr_auc_std': float(np.std(pr_aucs)),
        }
        
        print(f"\nCross-validation results:")
        print(f"  ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  PR-AUC: {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}")
        
        return results


def train_from_dataset(
    train_path: str = 'data/transactions_train.parquet',
    test_path: str = 'data/transactions_test.parquet',
    user_profiles_path: str = 'data/user_profiles.parquet',
    output_dir: str = 'models'
) -> str:
    """
    Complete training pipeline from saved dataset.
    
    Returns:
        Path to saved model
    """
    print("Loading datasets...")
    
    train_path = Path(train_path)
    test_path = Path(test_path)
    user_profiles_path = Path(user_profiles_path)
    
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    
    user_profiles_df = None
    if user_profiles_path.exists():
        user_profiles_df = pd.read_parquet(user_profiles_path)
    
    print(f"Training set: {len(df_train):,} transactions")
    print(f"Test set: {len(df_test):,} transactions")
    
    # Initialize model
    model = FraudDetectionModel(model_dir=output_dir)
    
    # Prepare features
    print("\nPreparing training features...")
    X_train = model.prepare_features(df_train, user_profiles_df, fit_scaler=True)
    y_train = df_train['is_fraud'].values
    
    print("\nPreparing test features...")
    X_test = model.prepare_features(df_test, user_profiles_df, fit_scaler=False)
    y_test = df_test['is_fraud'].values
    
    # Cross-validation
    model.cross_validate(X_train, y_train)
    
    # Train final model
    model.train(X_train, y_train, calibrate=True)
    
    # Evaluate on held-out test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    model._evaluate(X_test, y_test)
    
    # Save model
    model_path = model.save('fraud_model_v1')
    
    return model_path


def create_sample_model():
    """Create a sample model with synthetic data for quick testing."""
    print("Creating sample model with synthetic data...")
    
    # Generate small synthetic dataset
    np.random.seed(42)
    n_samples = 10000
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    
    # Create fraud signal - adjusted to get ~3% fraud rate
    fraud_score = (
        X[:, 0] * 0.8 +  # amount
        X[:, 5] * 0.5 +  # velocity
        X[:, 10] * 0.4 + # device risk
        np.random.randn(n_samples) * 0.3
    )
    
    # Use a threshold that gives ~2-3% fraud rate
    threshold = np.percentile(fraud_score, 97)  # Top 3% are fraud
    y = (fraud_score > threshold).astype(int)
    
    print(f"Synthetic dataset: {n_samples} samples, {y.sum()} fraud ({y.mean()*100:.1f}%)")
    
    # Create feature names
    feature_names = [
        'amount', 'amount_log', 'hour_of_day', 'day_of_week', 
        'is_weekend', 'is_night', 'is_high_risk_hour',
        'merchant_risk_score', 'is_high_risk_merchant', 
        'is_cash_equivalent', 'is_online',
        'txn_count_1h', 'txn_count_6h', 'txn_count_24h',
        'amount_sum_1h', 'amount_sum_24h',
        'unique_merchants_24h', 'unique_countries_24h',
        'time_since_last_txn_hours', 'velocity_score',
        'amount_zscore', 'amount_vs_avg_ratio',
        'is_above_95th_percentile', 'merchant_diversity_score',
        'spending_velocity_vs_normal',
        'is_foreign', 'is_high_risk_country', 'country_risk_score',
        'distance_from_home', 'is_impossible_travel'
    ]
    
    # Train model
    model = FraudDetectionModel(model_dir='models')
    model.feature_names = feature_names
    model.scaler = StandardScaler()
    X_scaled = model.scaler.fit_transform(X)
    
    model.train(X_scaled, y, calibrate=True)
    
    # Save model
    model_path = model.save('fraud_model_v1')
    
    return model_path


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--sample':
        # Create sample model for testing
        create_sample_model()
    else:
        # Full training pipeline
        try:
            train_from_dataset()
        except FileNotFoundError:
            print("Dataset not found. Generating synthetic dataset first...")
            print("Run: python generate_dataset.py")
            print("\nOr create sample model with: python train.py --sample")
            create_sample_model()
