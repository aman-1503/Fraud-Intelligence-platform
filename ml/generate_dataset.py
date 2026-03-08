"""
Synthetic Fraud Dataset Generator
=================================
Generates realistic transaction data with fraud patterns based on real-world fraud typologies.
Creates 500K transactions with ~2-3% fraud rate (realistic class imbalance).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import hashlib
import json
from pathlib import Path


class FraudDatasetGenerator:
    """Generates synthetic fraud transaction dataset with realistic patterns."""
    
    # Merchant category codes and their fraud risk levels
    MERCHANT_CATEGORIES = {
        'grocery': {'code': 5411, 'risk': 0.01, 'avg_amount': 85},
        'gas_station': {'code': 5541, 'risk': 0.02, 'avg_amount': 45},
        'restaurant': {'code': 5812, 'risk': 0.015, 'avg_amount': 35},
        'online_retail': {'code': 5999, 'risk': 0.04, 'avg_amount': 120},
        'electronics': {'code': 5732, 'risk': 0.05, 'avg_amount': 350},
        'jewelry': {'code': 5944, 'risk': 0.06, 'avg_amount': 500},
        'travel': {'code': 4722, 'risk': 0.03, 'avg_amount': 800},
        'entertainment': {'code': 7832, 'risk': 0.02, 'avg_amount': 50},
        'atm_withdrawal': {'code': 6011, 'risk': 0.025, 'avg_amount': 200},
        'money_transfer': {'code': 4829, 'risk': 0.08, 'avg_amount': 500},
        'crypto_exchange': {'code': 6051, 'risk': 0.10, 'avg_amount': 1000},
        'gambling': {'code': 7995, 'risk': 0.07, 'avg_amount': 150},
    }
    
    # Country risk profiles
    COUNTRIES = {
        'US': {'risk': 0.01, 'weight': 0.60},
        'UK': {'risk': 0.015, 'weight': 0.10},
        'CA': {'risk': 0.012, 'weight': 0.08},
        'DE': {'risk': 0.013, 'weight': 0.05},
        'FR': {'risk': 0.014, 'weight': 0.04},
        'BR': {'risk': 0.03, 'weight': 0.03},
        'NG': {'risk': 0.06, 'weight': 0.02},
        'RU': {'risk': 0.05, 'weight': 0.02},
        'CN': {'risk': 0.025, 'weight': 0.03},
        'IN': {'risk': 0.02, 'weight': 0.03},
    }
    
    def __init__(self, n_users: int = 50000, seed: int = 42):
        """Initialize generator with user count and random seed."""
        np.random.seed(seed)
        self.n_users = n_users
        self.users = self._generate_user_profiles()
        
    def _generate_user_profiles(self) -> pd.DataFrame:
        """Generate realistic user profiles with behavioral baselines."""
        users = pd.DataFrame({
            'user_id': [f'U{i:06d}' for i in range(self.n_users)],
            'home_country': np.random.choice(
                list(self.COUNTRIES.keys()),
                size=self.n_users,
                p=[c['weight'] for c in self.COUNTRIES.values()]
            ),
            'account_age_days': np.random.exponential(365, self.n_users).astype(int) + 30,
            'avg_monthly_txns': np.random.lognormal(2.5, 0.8, self.n_users).astype(int) + 5,
            'avg_txn_amount': np.random.lognormal(3.5, 1.0, self.n_users),
            'risk_score_base': np.random.beta(2, 10, self.n_users),  # Most users low risk
        })
        
        # Generate device fingerprints
        users['primary_device_id'] = [
            hashlib.md5(f'device_{i}'.encode()).hexdigest()[:16] 
            for i in range(self.n_users)
        ]
        
        return users
    
    def _generate_fraud_patterns(self, n_fraud: int) -> List[Dict]:
        """Generate various fraud attack patterns."""
        patterns = []
        
        # Pattern 1: Card testing (small amounts, rapid succession)
        n_card_test = int(n_fraud * 0.25)
        for _ in range(n_card_test):
            patterns.append({
                'type': 'card_testing',
                'amount_range': (0.50, 5.00),
                'velocity': 'high',
                'merchant_pref': ['online_retail', 'gas_station'],
                'time_pref': 'night',
                'new_device': True,
            })
        
        # Pattern 2: Account takeover (unusual location, high amounts)
        n_ato = int(n_fraud * 0.30)
        for _ in range(n_ato):
            patterns.append({
                'type': 'account_takeover',
                'amount_range': (500, 5000),
                'velocity': 'medium',
                'merchant_pref': ['electronics', 'jewelry', 'money_transfer'],
                'time_pref': 'any',
                'new_device': True,
                'foreign': True,
            })
        
        # Pattern 3: Bust-out fraud (gradual escalation)
        n_bustout = int(n_fraud * 0.20)
        for _ in range(n_bustout):
            patterns.append({
                'type': 'bust_out',
                'amount_range': (200, 2000),
                'velocity': 'escalating',
                'merchant_pref': ['crypto_exchange', 'money_transfer', 'gambling'],
                'time_pref': 'any',
                'new_device': False,
            })
        
        # Pattern 4: Synthetic identity (mixed behavior)
        n_synthetic = int(n_fraud * 0.15)
        for _ in range(n_synthetic):
            patterns.append({
                'type': 'synthetic_identity',
                'amount_range': (100, 1000),
                'velocity': 'low',
                'merchant_pref': list(self.MERCHANT_CATEGORIES.keys()),
                'time_pref': 'business',
                'new_device': False,
            })
        
        # Pattern 5: Friendly fraud / chargeback abuse
        n_friendly = n_fraud - n_card_test - n_ato - n_bustout - n_synthetic
        for _ in range(n_friendly):
            patterns.append({
                'type': 'friendly_fraud',
                'amount_range': (50, 500),
                'velocity': 'low',
                'merchant_pref': ['online_retail', 'entertainment'],
                'time_pref': 'any',
                'new_device': False,
            })
        
        return patterns
    
    def _generate_transaction(
        self, 
        user: pd.Series, 
        timestamp: datetime,
        is_fraud: bool = False,
        fraud_pattern: Dict = None
    ) -> Dict:
        """Generate a single transaction record."""
        
        if is_fraud and fraud_pattern:
            # Fraud transaction
            merchant_cat = np.random.choice(fraud_pattern['merchant_pref'])
            amount = np.random.uniform(*fraud_pattern['amount_range'])
            
            # Fraudsters often use new devices
            if fraud_pattern.get('new_device', False):
                device_id = hashlib.md5(
                    f'fraud_device_{np.random.randint(1e9)}'.encode()
                ).hexdigest()[:16]
                device_age_days = np.random.randint(0, 3)
            else:
                device_id = user['primary_device_id']
                device_age_days = np.random.randint(30, 365)
            
            # Foreign transaction for certain fraud types
            if fraud_pattern.get('foreign', False):
                high_risk_countries = ['NG', 'RU', 'BR']
                country = np.random.choice(high_risk_countries)
            else:
                country = user['home_country'] if np.random.random() > 0.3 else np.random.choice(list(self.COUNTRIES.keys()))
            
            # Adjust timestamp for fraud patterns
            if fraud_pattern['time_pref'] == 'night':
                timestamp = timestamp.replace(hour=np.random.randint(0, 6))
            elif fraud_pattern['time_pref'] == 'business':
                timestamp = timestamp.replace(hour=np.random.randint(9, 17))
                
        else:
            # Legitimate transaction
            merchant_weights = [1/cat['risk'] for cat in self.MERCHANT_CATEGORIES.values()]
            merchant_weights = np.array(merchant_weights) / sum(merchant_weights)
            merchant_cat = np.random.choice(
                list(self.MERCHANT_CATEGORIES.keys()),
                p=merchant_weights
            )
            
            base_amount = self.MERCHANT_CATEGORIES[merchant_cat]['avg_amount']
            amount = max(0.01, np.random.lognormal(
                np.log(base_amount), 
                0.5
            ) * (user['avg_txn_amount'] / 100))
            
            device_id = user['primary_device_id']
            device_age_days = np.random.randint(30, min(365, user['account_age_days']))
            
            # Mostly home country, occasionally travel
            country = user['home_country'] if np.random.random() > 0.05 else np.random.choice(list(self.COUNTRIES.keys()))
        
        merchant_info = self.MERCHANT_CATEGORIES[merchant_cat]
        
        return {
            'transaction_id': hashlib.md5(
                f'{user["user_id"]}_{timestamp.isoformat()}_{np.random.randint(1e9)}'.encode()
            ).hexdigest()[:24],
            'user_id': user['user_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': 'USD',
            'merchant_category': merchant_cat,
            'merchant_category_code': merchant_info['code'],
            'country': country,
            'is_foreign': country != user['home_country'],
            'device_id': device_id,
            'device_age_days': device_age_days,
            'is_new_device': device_id != user['primary_device_id'],
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'user_account_age_days': user['account_age_days'],
            'is_fraud': int(is_fraud),
            'fraud_type': fraud_pattern['type'] if is_fraud and fraud_pattern else None,
        }
    
    def generate_dataset(
        self, 
        n_transactions: int = 500000,
        fraud_rate: float = 0.025,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Generate complete transaction dataset."""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
        
        n_fraud = int(n_transactions * fraud_rate)
        n_legitimate = n_transactions - n_fraud
        
        print(f"Generating {n_transactions:,} transactions...")
        print(f"  - Legitimate: {n_legitimate:,} ({100*(1-fraud_rate):.1f}%)")
        print(f"  - Fraudulent: {n_fraud:,} ({100*fraud_rate:.1f}%)")
        
        transactions = []
        
        # Generate legitimate transactions
        print("Generating legitimate transactions...")
        for i in range(n_legitimate):
            if i % 100000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_legitimate:,}")
            
            user = self.users.iloc[np.random.randint(self.n_users)]
            timestamp = start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            transactions.append(self._generate_transaction(user, timestamp, is_fraud=False))
        
        # Generate fraud transactions
        print("Generating fraud transactions...")
        fraud_patterns = self._generate_fraud_patterns(n_fraud)
        
        for i, pattern in enumerate(fraud_patterns):
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_fraud:,}")
            
            user = self.users.iloc[np.random.randint(self.n_users)]
            timestamp = start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            transactions.append(self._generate_transaction(
                user, timestamp, is_fraud=True, fraud_pattern=pattern
            ))
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Sort by timestamp for realistic time-series
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nDataset generated successfully!")
        print(f"Shape: {df.shape}")
        print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        
        return df
    
    def add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based features computed from transaction history."""
        print("Computing velocity features...")
        
        df = df.sort_values(['user_id', 'timestamp']).copy()
        
        # Initialize velocity columns
        df['txn_count_1h'] = 0
        df['txn_count_24h'] = 0
        df['txn_amount_1h'] = 0.0
        df['txn_amount_24h'] = 0.0
        df['unique_merchants_24h'] = 0
        df['unique_countries_24h'] = 0
        df['time_since_last_txn'] = 0.0
        
        # Group by user and compute rolling features
        for user_id, group in df.groupby('user_id'):
            indices = group.index.tolist()
            timestamps = group['timestamp'].values
            amounts = group['amount'].values
            merchants = group['merchant_category'].values
            countries = group['country'].values
            
            for i, idx in enumerate(indices):
                current_time = timestamps[i]
                
                # Look back at previous transactions
                time_1h_ago = current_time - np.timedelta64(1, 'h')
                time_24h_ago = current_time - np.timedelta64(24, 'h')
                
                mask_1h = (timestamps[:i] > time_1h_ago) & (timestamps[:i] <= current_time)
                mask_24h = (timestamps[:i] > time_24h_ago) & (timestamps[:i] <= current_time)
                
                df.loc[idx, 'txn_count_1h'] = mask_1h.sum()
                df.loc[idx, 'txn_count_24h'] = mask_24h.sum()
                df.loc[idx, 'txn_amount_1h'] = amounts[:i][mask_1h].sum()
                df.loc[idx, 'txn_amount_24h'] = amounts[:i][mask_24h].sum()
                df.loc[idx, 'unique_merchants_24h'] = len(set(merchants[:i][mask_24h]))
                df.loc[idx, 'unique_countries_24h'] = len(set(countries[:i][mask_24h]))
                
                if i > 0:
                    time_diff = (current_time - timestamps[i-1]) / np.timedelta64(1, 'h')
                    df.loc[idx, 'time_since_last_txn'] = float(time_diff)
        
        print("Velocity features computed!")
        return df


def main():
    """Generate and save the fraud dataset."""
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    generator = FraudDatasetGenerator(n_users=50000, seed=42)
    df = generator.generate_dataset(
        n_transactions=500000,
        fraud_rate=0.025,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31)
    )
    
    # Add velocity features (this takes a while for large datasets)
    # For faster generation, we'll compute these during training
    # df = generator.add_velocity_features(df)
    
    # Save dataset
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    df_train.to_parquet(output_dir / 'transactions_train.parquet', index=False)
    df_test.to_parquet(output_dir / 'transactions_test.parquet', index=False)
    
    # Also save as CSV for inspection
    df_train.head(10000).to_csv(output_dir / 'transactions_sample.csv', index=False)
    
    # Save user profiles
    generator.users.to_parquet(output_dir / 'user_profiles.parquet', index=False)
    
    print(f"\nDatasets saved to {output_dir}")
    print(f"  - Training set: {len(df_train):,} transactions")
    print(f"  - Test set: {len(df_test):,} transactions")
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Fraud rate (train): {df_train['is_fraud'].mean()*100:.2f}%")
    print(f"Fraud rate (test): {df_test['is_fraud'].mean()*100:.2f}%")
    print(f"\nFraud types distribution:")
    print(df[df['is_fraud']==1]['fraud_type'].value_counts())
    print(f"\nMerchant category distribution:")
    print(df['merchant_category'].value_counts())


if __name__ == '__main__':
    main()
