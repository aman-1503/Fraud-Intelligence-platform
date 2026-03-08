-- Fraud Intelligence Platform - Database Initialization
-- PostgreSQL Schema

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ==========================================
-- Users Table
-- ==========================================
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    home_country VARCHAR(3) NOT NULL DEFAULT 'US',
    account_age_days INTEGER NOT NULL DEFAULT 0,
    avg_txn_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    std_txn_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    avg_monthly_txns INTEGER NOT NULL DEFAULT 0,
    primary_device_id VARCHAR(64),
    merchant_diversity DECIMAL(5, 4) DEFAULT 0.5,
    risk_score DECIMAL(5, 4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_risk_score ON users(risk_score DESC);

-- ==========================================
-- Transactions Table
-- ==========================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    merchant_category VARCHAR(50) NOT NULL,
    merchant_category_code INTEGER,
    country VARCHAR(3) NOT NULL,
    is_foreign BOOLEAN DEFAULT FALSE,
    device_id VARCHAR(64),
    device_age_days INTEGER,
    is_new_device BOOLEAN DEFAULT FALSE,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_user_timestamp ON transactions(user_id, timestamp DESC);

-- ==========================================
-- Fraud Scores Table
-- ==========================================
CREATE TABLE IF NOT EXISTS fraud_scores (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL,
    fraud_score DECIMAL(5, 4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    risk_factors JSONB DEFAULT '[]',
    model_version VARCHAR(20) NOT NULL,
    latency_ms DECIMAL(10, 2) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fraud_scores_transaction_id ON fraud_scores(transaction_id);
CREATE INDEX idx_fraud_scores_scored_at ON fraud_scores(scored_at DESC);
CREATE INDEX idx_fraud_scores_risk_level ON fraud_scores(risk_level);
CREATE INDEX idx_fraud_scores_decision ON fraud_scores(decision);

-- ==========================================
-- User Transaction History (for velocity)
-- ==========================================
CREATE TABLE IF NOT EXISTS user_transaction_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    merchant_category VARCHAR(50) NOT NULL,
    country VARCHAR(3) NOT NULL,
    device_id VARCHAR(64),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_user_history_user_timestamp ON user_transaction_history(user_id, timestamp DESC);

-- ==========================================
-- Analytics Aggregates (Hourly)
-- ==========================================
CREATE TABLE IF NOT EXISTS fraud_analytics_hourly (
    hour_bucket TIMESTAMP WITH TIME ZONE PRIMARY KEY,
    total_transactions INTEGER DEFAULT 0,
    fraud_count INTEGER DEFAULT 0,
    declined_count INTEGER DEFAULT 0,
    review_count INTEGER DEFAULT 0,
    total_amount DECIMAL(15, 2) DEFAULT 0,
    fraud_amount DECIMAL(15, 2) DEFAULT 0,
    avg_score DECIMAL(5, 4) DEFAULT 0,
    avg_latency_ms DECIMAL(10, 2) DEFAULT 0,
    p50_latency_ms DECIMAL(10, 2) DEFAULT 0,
    p95_latency_ms DECIMAL(10, 2) DEFAULT 0,
    p99_latency_ms DECIMAL(10, 2) DEFAULT 0
);

-- ==========================================
-- Model Metadata Table
-- ==========================================
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    metrics JSONB NOT NULL,
    feature_names JSONB NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_model_active ON model_metadata(is_active) WHERE is_active = TRUE;

-- ==========================================
-- Sample Data for Testing
-- ==========================================
INSERT INTO users (user_id, home_country, account_age_days, avg_txn_amount, std_txn_amount, avg_monthly_txns, primary_device_id, merchant_diversity, risk_score)
VALUES 
    ('U000001', 'US', 365, 150.00, 75.00, 25, 'device_001', 0.6, 0.15),
    ('U000002', 'UK', 180, 200.00, 100.00, 15, 'device_002', 0.4, 0.25),
    ('U000003', 'US', 730, 500.00, 200.00, 40, 'device_003', 0.8, 0.10),
    ('U000004', 'CA', 90, 80.00, 40.00, 10, 'device_004', 0.3, 0.30),
    ('U000005', 'DE', 450, 300.00, 150.00, 20, 'device_005', 0.5, 0.20)
ON CONFLICT (user_id) DO NOTHING;

-- ==========================================
-- Functions and Triggers
-- ==========================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to compute velocity features
CREATE OR REPLACE FUNCTION get_user_velocity(
    p_user_id VARCHAR(50),
    p_window_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    txn_count BIGINT,
    total_amount DECIMAL(15, 2),
    unique_merchants BIGINT,
    unique_countries BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT,
        COALESCE(SUM(amount), 0)::DECIMAL(15, 2),
        COUNT(DISTINCT merchant_category)::BIGINT,
        COUNT(DISTINCT country)::BIGINT
    FROM user_transaction_history
    WHERE user_id = p_user_id
      AND timestamp > NOW() - (p_window_hours || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- Views
-- ==========================================

-- Recent high-risk transactions view
CREATE OR REPLACE VIEW v_high_risk_transactions AS
SELECT 
    t.transaction_id,
    t.user_id,
    t.amount,
    t.merchant_category,
    t.country,
    fs.fraud_score,
    fs.risk_level,
    fs.decision,
    fs.risk_factors,
    fs.scored_at
FROM transactions t
JOIN fraud_scores fs ON t.transaction_id = fs.transaction_id
WHERE fs.fraud_score > 0.6
ORDER BY fs.scored_at DESC
LIMIT 1000;

-- Daily fraud summary view
CREATE OR REPLACE VIEW v_daily_fraud_summary AS
SELECT 
    DATE_TRUNC('day', scored_at) AS day,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN fraud_score > 0.5 THEN 1 ELSE 0 END) AS fraud_count,
    AVG(fraud_score) AS avg_fraud_score,
    AVG(latency_ms) AS avg_latency_ms,
    SUM(CASE WHEN decision = 'DECLINE' THEN 1 ELSE 0 END) AS declined_count,
    SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count
FROM fraud_scores
GROUP BY DATE_TRUNC('day', scored_at)
ORDER BY day DESC;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
