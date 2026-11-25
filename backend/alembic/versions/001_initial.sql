-- Canonical schema. Alembic applies this file. Compose can mount it too.
-- Profile columns are nullable so a first-seen account is not pretend-US.

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    home_country VARCHAR(3),
    account_age_days INTEGER NOT NULL DEFAULT 0,
    avg_txn_amount NUMERIC(12, 2) NOT NULL DEFAULT 0,
    std_txn_amount NUMERIC(12, 2) NOT NULL DEFAULT 0,
    avg_monthly_txns INTEGER NOT NULL DEFAULT 0,
    primary_device_id VARCHAR(64),
    merchant_diversity NUMERIC(5, 4) DEFAULT 0.5,
    risk_score NUMERIC(5, 4) DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    amount NUMERIC(12, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    merchant_category VARCHAR(50) NOT NULL,
    country VARCHAR(3) NOT NULL,
    device_id VARCHAR(64),
    timestamp TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transactions_user_timestamp
    ON transactions(user_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS fraud_scores (
    transaction_id VARCHAR(64) PRIMARY KEY REFERENCES transactions(transaction_id),
    request_hash CHAR(64) NOT NULL,
    fraud_score DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fraud_scores_scored_at ON fraud_scores(scored_at DESC);

CREATE TABLE IF NOT EXISTS investigation_cases (
    case_id TEXT PRIMARY KEY,
    transaction_id VARCHAR(64) UNIQUE NOT NULL REFERENCES transactions(transaction_id),
    status TEXT NOT NULL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON investigation_cases(status);

INSERT INTO users (
    user_id, home_country, account_age_days, avg_txn_amount, std_txn_amount,
    avg_monthly_txns, primary_device_id, merchant_diversity, risk_score
) VALUES
    ('U000000', 'US', 30, 40.00, 15.00, 8, 'device_000', 0.3, 0.20),
    ('U000001', 'US', 365, 150.00, 75.00, 25, 'device_001', 0.6, 0.15),
    ('U000002', 'UK', 180, 200.00, 100.00, 15, 'device_002', 0.4, 0.25),
    ('U000003', 'US', 730, 500.00, 200.00, 40, 'device_003', 0.8, 0.10),
    ('U000004', 'CA', 90, 80.00, 40.00, 10, 'device_004', 0.3, 0.30),
    ('U000005', 'DE', 450, 300.00, 150.00, 20, 'device_005', 0.5, 0.20),
    ('U000006', 'US', 60, 55.00, 20.00, 12, 'device_006', 0.4, 0.18),
    ('U000007', 'FR', 220, 120.00, 40.00, 18, 'device_007', 0.5, 0.16),
    ('U000008', 'US', 15, 25.00, 10.00, 6, 'device_008', 0.2, 0.28),
    ('U000009', 'UK', 800, 250.00, 90.00, 22, 'device_009', 0.7, 0.12)
ON CONFLICT (user_id) DO NOTHING;
