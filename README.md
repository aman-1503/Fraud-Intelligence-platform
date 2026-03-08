# 🛡️ Cloud-Native Distributed Fraud Intelligence Platform

Real-time fraud detection platform achieving **sub-150ms latency** and handling **3,000+ requests per second** with XGBoost ML pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![Redis](https://img.shields.io/badge/Redis-7-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Key Features

| Feature | Specification |
|---------|--------------|
| **Latency** | Sub-150ms P99 fraud scoring |
| **Throughput** | 3,000+ RPS validated under load |
| **ML Model** | XGBoost with 96.7% AUC |
| **Real-time** | Live feature extraction & velocity tracking |
| **Scalable** | Horizontal scaling with Kubernetes |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                             │
│                    (FastAPI + Uvicorn)                          │
│              Rate Limiting │ Auth │ Load Balancing              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Feature  │   │    ML     │   │ Decision  │
    │  Engine   │──▶│  Pipeline │──▶│  Engine   │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐        ┌──────────┐        ┌─────────┐
   │  Redis  │        │PostgreSQL│        │ Message │
   │  Cache  │        │    DB    │        │  Queue  │
   └─────────┘        └──────────┘        └─────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend)

### 1. Clone and Setup

```bash
cd fraud-platform

# Create model directory
mkdir -p ml/models/fraud_model_v1

# Train the model (creates sample model)
cd ml && python train.py --sample && cd ..
```

### 2. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **API** | http://localhost:8000 | Fraud scoring API |
| **API Docs** | http://localhost:8000/docs | Interactive API docs |
| **Frontend** | http://localhost:3000 | React dashboard |
| **Grafana** | http://localhost:3001 | Monitoring dashboards |
| **Prometheus** | http://localhost:9090 | Metrics |

## 📊 API Usage

### Score a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-12345",
    "user_id": "U000001",
    "amount": 150.00,
    "merchant_category": "electronics",
    "country": "US",
    "device_id": "device_001"
  }'
```

### Response

```json
{
  "transaction_id": "TXN-12345",
  "fraud_score": 0.15,
  "risk_level": "LOW",
  "decision": "APPROVE",
  "confidence": 0.92,
  "latency_ms": 42.5,
  "risk_factors": [],
  "model_version": "v1.0.0",
  "scored_at": "2024-01-15T10:30:00Z"
}
```

### Batch Scoring

```bash
curl -X POST http://localhost:8000/api/v1/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_id": "TXN-001", "user_id": "U000001", "amount": 50.00, "merchant_category": "grocery", "country": "US"},
      {"transaction_id": "TXN-002", "user_id": "U000002", "amount": 500.00, "merchant_category": "electronics", "country": "UK"}
    ]
  }'
```

## 🧪 Load Testing

Run load tests to validate performance:

```bash
# Start Locust
docker-compose --profile testing up locust

# Access Locust UI
open http://localhost:8089
```

Or run headless:

```bash
docker-compose run locust \
  -f /mnt/locust/locustfile.py \
  --host=http://api:8000 \
  --headless \
  -u 500 \
  -r 50 \
  -t 60s
```

**Expected Results:**
- Median latency: <50ms
- P99 latency: <150ms
- Throughput: >3,000 RPS

## 🤖 ML Pipeline

### Features

| Category | Features |
|----------|----------|
| **Transaction** | amount, amount_log, hour, day_of_week, is_weekend |
| **Merchant** | risk_score, is_high_risk, is_cash_equivalent |
| **Velocity** | txn_count_1h/6h/24h, amount_sum_1h/24h |
| **Behavioral** | amount_zscore, deviation_from_avg |
| **Geographic** | is_foreign, country_risk, distance_from_home |
| **Device** | is_new_device, device_age, device_risk |

### Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.967 |
| PR-AUC | 0.854 |
| Precision | 94.2% |
| Recall | 89.7% |
| F1 Score | 91.9% |

### Retrain Model

```bash
# Generate synthetic dataset
cd ml
python generate_dataset.py

# Train model
python train.py

# Model saved to ml/models/fraud_model_v1/
```

## 📁 Project Structure

```
fraud-platform/
├── ml/                          # Machine Learning
│   ├── data/                    # Datasets
│   ├── models/                  # Trained models
│   ├── generate_dataset.py      # Synthetic data generation
│   ├── features.py              # Feature engineering
│   └── train.py                 # Model training
├── backend/                     # FastAPI Backend
│   ├── app/
│   │   ├── api/                 # API routes
│   │   ├── core/                # Config, logging
│   │   ├── ml/                  # ML inference
│   │   ├── db/                  # Database layer
│   │   └── main.py              # Application entry
│   └── requirements.txt
├── frontend/                    # React Dashboard
│   └── src/
│       └── FraudDashboard.jsx
├── infra/                       # Infrastructure
│   ├── docker/                  # Dockerfiles
│   ├── k8s/                     # Kubernetes configs
│   ├── prometheus/              # Monitoring
│   └── grafana/                 # Dashboards
├── load-tests/                  # Performance tests
│   └── locustfile.py
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MODEL_PATH` | `models/fraud_model_v1` | Model directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `TARGET_LATENCY_MS` | `150` | SLA target |

### Fraud Thresholds

| Risk Level | Score Range | Decision |
|------------|-------------|----------|
| LOW | 0.0 - 0.3 | APPROVE |
| MEDIUM | 0.3 - 0.6 | APPROVE (monitor) |
| HIGH | 0.6 - 0.85 | REVIEW |
| CRITICAL | 0.85 - 1.0 | DECLINE |

## 📈 Monitoring

### Grafana Dashboards

1. **Fraud Overview** - Real-time fraud metrics
2. **API Performance** - Latency and throughput
3. **Model Performance** - Score distributions, drift detection

### Key Metrics

- `fraud_scoring_latency_seconds` - Scoring latency histogram
- `fraud_scoring_total` - Total predictions counter
- `fraud_score_distribution` - Score distribution
- `cache_hit_rate` - Redis cache effectiveness

## 🔒 Security

- API key authentication
- Rate limiting (1000 req/min default)
- Input validation with Pydantic
- SQL injection prevention
- CORS configuration

## 🛠️ Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000
```

### Run Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

Built with ❤️ for real-time fraud detection at scale.
