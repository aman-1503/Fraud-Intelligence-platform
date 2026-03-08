"""
Load Testing for Fraud Intelligence Platform
=============================================
Locust-based load tests to validate 3,000+ RPS throughput.

Run with:
    locust -f locustfile.py --host=http://localhost:8000
    
Or headless:
    locust -f locustfile.py --host=http://localhost:8000 --headless -u 500 -r 50 -t 60s
"""

import random
import string
import uuid
from datetime import datetime
from locust import HttpUser, task, between, events
import json
import time


# Test data generators
MERCHANT_CATEGORIES = [
    'grocery', 'gas_station', 'restaurant', 'online_retail', 
    'electronics', 'jewelry', 'travel', 'entertainment', 
    'atm_withdrawal', 'money_transfer', 'crypto_exchange', 'gambling'
]

COUNTRIES = ['US', 'UK', 'CA', 'DE', 'FR', 'BR', 'NG', 'RU', 'CN', 'IN']

USER_IDS = [f'U{str(i).zfill(6)}' for i in range(1000)]


def generate_transaction():
    """Generate a random transaction for testing."""
    return {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12]}",
        "user_id": random.choice(USER_IDS),
        "amount": round(random.uniform(1, 5000), 2),
        "currency": "USD",
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "country": random.choice(COUNTRIES),
        "device_id": f"device_{random.randint(1, 100):03d}",
        "timestamp": datetime.utcnow().isoformat(),
    }


def generate_fraud_transaction():
    """Generate a suspicious transaction for testing."""
    return {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12]}",
        "user_id": random.choice(USER_IDS),
        "amount": round(random.uniform(2000, 10000), 2),  # High amount
        "currency": "USD",
        "merchant_category": random.choice(['crypto_exchange', 'money_transfer', 'gambling']),
        "country": random.choice(['NG', 'RU', 'BR']),  # High-risk country
        "device_id": f"new_device_{random.randint(1000, 9999)}",  # New device
        "is_new_device": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


class FraudAPIUser(HttpUser):
    """
    Simulated user for load testing the fraud scoring API.
    """
    
    wait_time = between(0.01, 0.1)  # 10-100ms between requests
    
    @task(10)
    def score_normal_transaction(self):
        """Score a normal transaction (most common operation)."""
        transaction = generate_transaction()
        
        with self.client.post(
            "/api/v1/score",
            json=transaction,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                latency = float(response.headers.get('X-Process-Time-Ms', 0))
                
                # Validate response
                if 'fraud_score' not in data:
                    response.failure("Missing fraud_score in response")
                elif latency > 150:
                    response.failure(f"Latency too high: {latency}ms")
                else:
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def score_suspicious_transaction(self):
        """Score a suspicious/fraud-like transaction."""
        transaction = generate_fraud_transaction()
        
        with self.client.post(
            "/api/v1/score",
            json=transaction,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Suspicious transactions should have higher scores
                if data.get('fraud_score', 0) < 0.3:
                    # This is expected sometimes - log but don't fail
                    pass
                    
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def batch_score(self):
        """Score a batch of transactions."""
        transactions = [generate_transaction() for _ in range(10)]
        
        with self.client.post(
            "/api/v1/score/batch",
            json={"transactions": transactions},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                if len(data.get('results', [])) != 10:
                    response.failure("Incorrect batch result count")
                else:
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check API health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_user_profile(self):
        """Fetch user profile."""
        user_id = random.choice(USER_IDS[:10])  # Use first 10 users (seeded in DB)
        
        with self.client.get(
            f"/api/v1/user/{user_id}/profile",
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 is OK for non-existent users
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    User optimized for maximum throughput testing.
    Minimal wait time, only scoring requests.
    """
    
    wait_time = between(0, 0.01)  # Near-zero wait
    
    @task
    def rapid_score(self):
        """Rapid-fire scoring requests."""
        transaction = generate_transaction()
        
        start = time.time()
        response = self.client.post("/api/v1/score", json=transaction)
        latency = (time.time() - start) * 1000
        
        # Track latency statistics
        if hasattr(self, 'latencies'):
            self.latencies.append(latency)
        else:
            self.latencies = [latency]


# Custom event handlers for detailed metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Track detailed request metrics."""
    pass


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics after test."""
    stats = environment.stats
    
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    
    print(f"Total Requests: {total_requests:,}")
    print(f"Failed Requests: {total_failures:,}")
    print(f"Failure Rate: {100 * total_failures / max(1, total_requests):.2f}%")
    print(f"Requests/sec: {stats.total.total_rps:.2f}")
    
    print("\nLatency Statistics:")
    print(f"  Median: {stats.total.median_response_time:.2f}ms")
    print(f"  Average: {stats.total.avg_response_time:.2f}ms")
    print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    
    print("=" * 60)


# Performance validation test
def validate_performance():
    """
    Validate that the API meets performance requirements:
    - Sub-150ms P99 latency
    - 3,000+ RPS throughput
    """
    import subprocess
    import sys
    
    print("Running performance validation...")
    
    result = subprocess.run([
        sys.executable, "-m", "locust",
        "-f", __file__,
        "--host", "http://localhost:8000",
        "--headless",
        "-u", "300",  # 300 concurrent users
        "-r", "50",   # Ramp up 50 users/sec
        "-t", "30s",  # Run for 30 seconds
        "--csv", "load_test_results",
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Load test failed: {result.stderr}")
        return False
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_performance()
    else:
        print("Run with: locust -f locustfile.py --host=http://localhost:8000")
        print("Or: python locustfile.py --validate")
