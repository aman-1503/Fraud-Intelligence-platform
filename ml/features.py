"""
Feature Engineering Module
==========================
Real-time feature extraction pipeline for fraud detection.
Computes velocity, behavioral, geographic, and device features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict
import json


@dataclass
class UserProfile:
    """Cached user profile for fast feature computation."""
    user_id: str
    home_country: str
    account_age_days: int
    avg_txn_amount: float
    std_txn_amount: float
    avg_monthly_txns: int
    primary_device_id: str
    recent_transactions: List[Dict] = field(default_factory=list)
    
    def add_transaction(self, txn: Dict):
        """Add transaction to history, keeping last 100."""
        self.recent_transactions.append(txn)
        if len(self.recent_transactions) > 100:
            self.recent_transactions = self.recent_transactions[-100:]


class FeatureExtractor:
    """
    Real-time feature extraction for fraud scoring.
    Maintains user state for velocity calculations.
    """
    
    # Feature groups
    TRANSACTION_FEATURES = [
        'amount', 'amount_log', 'hour_of_day', 'day_of_week', 
        'is_weekend', 'is_night', 'is_high_risk_hour'
    ]
    
    MERCHANT_FEATURES = [
        'merchant_risk_score', 'is_high_risk_merchant', 
        'is_cash_equivalent', 'is_online'
    ]
    
    VELOCITY_FEATURES = [
        'txn_count_1h', 'txn_count_6h', 'txn_count_24h',
        'amount_sum_1h', 'amount_sum_24h',
        'unique_merchants_24h', 'unique_countries_24h',
        'time_since_last_txn_hours', 'velocity_score'
    ]
    
    BEHAVIORAL_FEATURES = [
        'amount_zscore', 'amount_vs_avg_ratio',
        'is_above_95th_percentile', 'merchant_diversity_score',
        'spending_velocity_vs_normal'
    ]
    
    GEOGRAPHIC_FEATURES = [
        'is_foreign', 'is_high_risk_country', 'country_risk_score',
        'distance_from_home', 'is_impossible_travel'
    ]
    
    DEVICE_FEATURES = [
        'is_new_device', 'device_age_days', 'device_risk_score',
        'devices_used_24h'
    ]
    
    # Risk mappings
    MERCHANT_RISK = {
        'grocery': 0.1, 'gas_station': 0.15, 'restaurant': 0.12,
        'online_retail': 0.3, 'electronics': 0.4, 'jewelry': 0.5,
        'travel': 0.25, 'entertainment': 0.15, 'atm_withdrawal': 0.2,
        'money_transfer': 0.6, 'crypto_exchange': 0.8, 'gambling': 0.55
    }
    
    HIGH_RISK_MERCHANTS = {'money_transfer', 'crypto_exchange', 'gambling', 'jewelry'}
    CASH_EQUIVALENT = {'atm_withdrawal', 'money_transfer', 'crypto_exchange'}
    
    COUNTRY_RISK = {
        'US': 0.1, 'UK': 0.12, 'CA': 0.11, 'DE': 0.12, 'FR': 0.13,
        'BR': 0.35, 'NG': 0.6, 'RU': 0.5, 'CN': 0.3, 'IN': 0.25
    }
    
    HIGH_RISK_COUNTRIES = {'NG', 'RU', 'BR'}
    
    def __init__(self):
        """Initialize feature extractor with empty user cache."""
        self.user_profiles: Dict[str, UserProfile] = {}
        self.feature_names = (
            self.TRANSACTION_FEATURES + 
            self.MERCHANT_FEATURES + 
            self.VELOCITY_FEATURES + 
            self.BEHAVIORAL_FEATURES + 
            self.GEOGRAPHIC_FEATURES + 
            self.DEVICE_FEATURES
        )
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        return self.feature_names.copy()
    
    def load_user_profile(self, user_id: str, profile_data: Dict) -> UserProfile:
        """Load or update user profile from database."""
        profile = UserProfile(
            user_id=user_id,
            home_country=profile_data.get('home_country', 'US'),
            account_age_days=profile_data.get('account_age_days', 0),
            avg_txn_amount=profile_data.get('avg_txn_amount', 100.0),
            std_txn_amount=profile_data.get('std_txn_amount', 50.0),
            avg_monthly_txns=profile_data.get('avg_monthly_txns', 20),
            primary_device_id=profile_data.get('primary_device_id', ''),
            recent_transactions=profile_data.get('recent_transactions', [])
        )
        self.user_profiles[user_id] = profile
        return profile
    
    def extract_features(
        self, 
        transaction: Dict,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, float]:
        """
        Extract all features for a single transaction.
        
        Args:
            transaction: Raw transaction data
            user_profile: Optional pre-loaded user profile
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Get user profile
        user_id = transaction.get('user_id', '')
        if user_profile is None:
            user_profile = self.user_profiles.get(user_id)
        
        # Transaction features
        amount = float(transaction.get('amount', 0))
        features['amount'] = amount
        features['amount_log'] = np.log1p(amount)
        
        hour = int(transaction.get('hour_of_day', 12))
        features['hour_of_day'] = hour
        features['day_of_week'] = int(transaction.get('day_of_week', 0))
        features['is_weekend'] = float(transaction.get('is_weekend', False))
        features['is_night'] = float(hour >= 23 or hour <= 5)
        features['is_high_risk_hour'] = float(hour >= 0 and hour <= 6)
        
        # Merchant features
        merchant_cat = transaction.get('merchant_category', 'other')
        features['merchant_risk_score'] = self.MERCHANT_RISK.get(merchant_cat, 0.3)
        features['is_high_risk_merchant'] = float(merchant_cat in self.HIGH_RISK_MERCHANTS)
        features['is_cash_equivalent'] = float(merchant_cat in self.CASH_EQUIVALENT)
        features['is_online'] = float(merchant_cat in {'online_retail', 'crypto_exchange'})
        
        # Geographic features
        country = transaction.get('country', 'US')
        home_country = user_profile.home_country if user_profile else 'US'
        features['is_foreign'] = float(transaction.get('is_foreign', country != home_country))
        features['is_high_risk_country'] = float(country in self.HIGH_RISK_COUNTRIES)
        features['country_risk_score'] = self.COUNTRY_RISK.get(country, 0.3)
        features['distance_from_home'] = self._estimate_distance(home_country, country)
        features['is_impossible_travel'] = 0.0  # Computed with velocity
        
        # Device features
        features['is_new_device'] = float(transaction.get('is_new_device', False))
        features['device_age_days'] = float(transaction.get('device_age_days', 30))
        features['device_risk_score'] = self._compute_device_risk(
            transaction.get('is_new_device', False),
            transaction.get('device_age_days', 30)
        )
        features['devices_used_24h'] = 1.0  # Default, computed with history
        
        # Velocity features (require user history)
        velocity = self._compute_velocity_features(transaction, user_profile)
        features.update(velocity)
        
        # Behavioral features (require user profile)
        behavioral = self._compute_behavioral_features(amount, user_profile)
        features.update(behavioral)
        
        # Check for impossible travel
        if user_profile and user_profile.recent_transactions:
            features['is_impossible_travel'] = self._check_impossible_travel(
                transaction, user_profile
            )
        
        return features
    
    def _compute_velocity_features(
        self, 
        transaction: Dict,
        user_profile: Optional[UserProfile]
    ) -> Dict[str, float]:
        """Compute velocity-based features from transaction history."""
        features = {
            'txn_count_1h': 0.0,
            'txn_count_6h': 0.0,
            'txn_count_24h': 0.0,
            'amount_sum_1h': 0.0,
            'amount_sum_24h': 0.0,
            'unique_merchants_24h': 0.0,
            'unique_countries_24h': 0.0,
            'time_since_last_txn_hours': 24.0,  # Default to 24h if no history
            'velocity_score': 0.0
        }
        
        if not user_profile or not user_profile.recent_transactions:
            return features
        
        current_time = transaction.get('timestamp')
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)
        elif not isinstance(current_time, datetime):
            current_time = datetime.now()
        
        merchants_24h = set()
        countries_24h = set()
        last_txn_time = None
        
        for txn in user_profile.recent_transactions:
            txn_time = txn.get('timestamp')
            if isinstance(txn_time, str):
                txn_time = datetime.fromisoformat(txn_time)
            elif not isinstance(txn_time, datetime):
                continue
            
            hours_ago = (current_time - txn_time).total_seconds() / 3600
            
            if hours_ago <= 1:
                features['txn_count_1h'] += 1
                features['amount_sum_1h'] += txn.get('amount', 0)
            
            if hours_ago <= 6:
                features['txn_count_6h'] += 1
            
            if hours_ago <= 24:
                features['txn_count_24h'] += 1
                features['amount_sum_24h'] += txn.get('amount', 0)
                merchants_24h.add(txn.get('merchant_category', ''))
                countries_24h.add(txn.get('country', ''))
            
            if last_txn_time is None or txn_time > last_txn_time:
                last_txn_time = txn_time
        
        features['unique_merchants_24h'] = float(len(merchants_24h))
        features['unique_countries_24h'] = float(len(countries_24h))
        
        if last_txn_time:
            features['time_since_last_txn_hours'] = (
                current_time - last_txn_time
            ).total_seconds() / 3600
        
        # Compute velocity score (normalized)
        velocity_score = (
            features['txn_count_1h'] * 3 +
            features['txn_count_6h'] * 0.5 +
            features['txn_count_24h'] * 0.1 +
            min(1, features['amount_sum_1h'] / 1000) * 2
        )
        features['velocity_score'] = min(10, velocity_score)
        
        return features
    
    def _compute_behavioral_features(
        self, 
        amount: float,
        user_profile: Optional[UserProfile]
    ) -> Dict[str, float]:
        """Compute behavioral deviation features."""
        features = {
            'amount_zscore': 0.0,
            'amount_vs_avg_ratio': 1.0,
            'is_above_95th_percentile': 0.0,
            'merchant_diversity_score': 0.5,
            'spending_velocity_vs_normal': 1.0
        }
        
        if not user_profile:
            return features
        
        avg_amount = user_profile.avg_txn_amount
        std_amount = max(user_profile.std_txn_amount, 1.0)  # Avoid div by zero
        
        # Z-score of transaction amount
        features['amount_zscore'] = (amount - avg_amount) / std_amount
        
        # Ratio to average
        features['amount_vs_avg_ratio'] = amount / max(avg_amount, 1.0)
        
        # Above 95th percentile (assume ~2 std devs)
        threshold_95 = avg_amount + 2 * std_amount
        features['is_above_95th_percentile'] = float(amount > threshold_95)
        
        # Merchant diversity from history
        if user_profile.recent_transactions:
            merchants = set(t.get('merchant_category', '') 
                          for t in user_profile.recent_transactions)
            features['merchant_diversity_score'] = min(1.0, len(merchants) / 10)
            
            # Spending velocity vs normal
            recent_daily = sum(t.get('amount', 0) 
                             for t in user_profile.recent_transactions[-10:])
            expected_daily = avg_amount * user_profile.avg_monthly_txns / 30
            features['spending_velocity_vs_normal'] = (
                recent_daily / max(expected_daily, 1.0)
            )
        
        return features
    
    def _compute_device_risk(self, is_new: bool, age_days: int) -> float:
        """Compute device risk score."""
        if is_new:
            return 0.8
        if age_days < 7:
            return 0.5
        if age_days < 30:
            return 0.3
        return 0.1
    
    def _estimate_distance(self, home: str, current: str) -> float:
        """Estimate distance between countries (simplified)."""
        if home == current:
            return 0.0
        
        # Simplified distance categories
        same_region = {
            ('US', 'CA'), ('CA', 'US'),
            ('UK', 'DE'), ('DE', 'UK'), ('UK', 'FR'), ('FR', 'UK'),
            ('DE', 'FR'), ('FR', 'DE'),
        }
        
        if (home, current) in same_region:
            return 1000.0
        
        return 5000.0  # Default cross-region distance
    
    def _check_impossible_travel(
        self, 
        transaction: Dict,
        user_profile: UserProfile
    ) -> float:
        """Check if travel between locations is physically impossible."""
        if not user_profile.recent_transactions:
            return 0.0
        
        current_country = transaction.get('country', 'US')
        current_time = transaction.get('timestamp')
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)
        
        # Check last transaction
        last_txn = user_profile.recent_transactions[-1]
        last_country = last_txn.get('country', 'US')
        last_time = last_txn.get('timestamp')
        if isinstance(last_time, str):
            last_time = datetime.fromisoformat(last_time)
        
        if current_country == last_country:
            return 0.0
        
        hours_diff = (current_time - last_time).total_seconds() / 3600
        distance = self._estimate_distance(last_country, current_country)
        
        # Assume minimum 500 km/h travel speed (fast plane)
        min_hours_needed = distance / 500
        
        if hours_diff < min_hours_needed * 0.5:  # Half the minimum time
            return 1.0
        
        return 0.0
    
    def extract_batch(
        self, 
        transactions: List[Dict],
        user_profiles: Optional[Dict[str, UserProfile]] = None
    ) -> np.ndarray:
        """Extract features for a batch of transactions."""
        features_list = []
        
        for txn in transactions:
            user_id = txn.get('user_id', '')
            profile = None
            if user_profiles:
                profile = user_profiles.get(user_id)
            elif user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
            
            features = self.extract_features(txn, profile)
            features_list.append([features[name] for name in self.feature_names])
        
        return np.array(features_list)


def extract_features_from_dataframe(
    df: pd.DataFrame,
    user_profiles_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from a DataFrame of transactions.
    Used for training the model.
    """
    extractor = FeatureExtractor()
    
    # Load user profiles if provided
    if user_profiles_df is not None:
        for _, row in user_profiles_df.iterrows():
            extractor.load_user_profile(row['user_id'], row.to_dict())
    
    # Convert DataFrame to list of dicts
    transactions = df.to_dict('records')
    
    # Extract features
    features = extractor.extract_batch(transactions)
    
    return features, extractor.get_feature_names()


if __name__ == '__main__':
    # Test feature extraction
    extractor = FeatureExtractor()
    
    # Sample transaction
    sample_txn = {
        'transaction_id': 'TXN001',
        'user_id': 'U000001',
        'timestamp': datetime.now(),
        'amount': 150.00,
        'currency': 'USD',
        'merchant_category': 'electronics',
        'country': 'US',
        'is_foreign': False,
        'device_id': 'device123',
        'device_age_days': 45,
        'is_new_device': False,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_weekend': False,
    }
    
    # Load sample user profile
    extractor.load_user_profile('U000001', {
        'home_country': 'US',
        'account_age_days': 365,
        'avg_txn_amount': 100.0,
        'std_txn_amount': 50.0,
        'avg_monthly_txns': 15,
        'primary_device_id': 'device123',
    })
    
    # Extract features
    features = extractor.extract_features(sample_txn)
    
    print("Extracted Features:")
    print("-" * 50)
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\nTotal features: {len(features)}")
