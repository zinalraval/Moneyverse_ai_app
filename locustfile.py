from locust import HttpUser, task, between
import os
import random
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
TEST_EMAIL = os.getenv("LOCUST_EMAIL", "testuser@example.com")
TEST_PASSWORD = os.getenv("LOCUST_PASSWORD", "testpassword")

class MoneyverseUser(HttpUser):
    wait_time = between(1, 3)
    auth_token = None

    def on_start(self):
        self.login()

    def login(self):
        resp = self.client.post(f"{API_BASE_URL}/auth/login", data={"username": TEST_EMAIL, "password": TEST_PASSWORD})
        if resp.status_code == 200 and "access_token" in resp.json():
            self.auth_token = resp.json()["access_token"]
        else:
            self.auth_token = None

    @task(2)
    def fetch_signals(self):
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        self.client.get(f"{API_BASE_URL}/signals/active", headers=headers)

    @task(2)
    def fetch_market_data(self):
        pair = random.choice([
            "BTC/USD", "ETH/USD", "XAU/USD", "EUR/USD", "USD/JPY"
        ])
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        self.client.get(f"{API_BASE_URL}/market-data/{pair}", headers=headers)

    @task(1)
    def fetch_news(self):
        pair = random.choice([
            "BTC-USD", "ETH-USD", "XAU-USD", "EUR-USD", "USD-JPY"
        ])
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        self.client.get(f"{API_BASE_URL}/news/{pair}", headers=headers)

    @task(1)
    def health_check(self):
        self.client.get(f"{API_BASE_URL}/health") 