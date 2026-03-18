from locust import HttpUser, task, between

class FastApiUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def test_predict(self):
        payload = {
            "sepalLength": 5.1,
            "sepalWidth": 3.5,
            "petalLength": 1.4,
            "petalWidth": 0.2
        }
        self.client.post("/predict", json=payload)