import os
from locust import HttpUser, task, between
from fastapi.testclient import TestClient
import random

# Assuming your FastAPI app is in a file named 'app.py'
from app import app  # Replace 'app' with the actual filename if different

# Sample files for testing (adjust paths as needed)
SAMPLE_IMAGE_PATH = "/Users/samenergy/Documents/Projects/PlantDiseaseDetection/Data/Tests/test/TomatoYellowCurlVirus6.JPG"  # A single image file for /predict
SAMPLE_ZIP_PATH = "/Users/samenergy/Documents/Projects/PlantDiseaseDetection/Archive.zip"   # A ZIP file for /retrain

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 5)  # Simulate users waiting 1-5 seconds between tasks
    host = " https://appdeploy-production.up.railway.app/ "  # The host where your FastAPI app is running

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use FastAPI's TestClient for direct testing (optional, remove if using real HTTP)
        self.client = TestClient(app)

    @task(3)  # Weight: 3x more likely than /retrain
    def test_predict(self):
        """Simulate a user uploading an image to /predict."""
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            print(f"Sample image not found at {SAMPLE_IMAGE_PATH}")
            return

        with open(SAMPLE_IMAGE_PATH, "rb") as image_file:
            files = {"file": ("sample.jpg", image_file, "image/jpeg")}
            # Using Locust's HTTP client
            response = self.client.post("/predict", files=files)
            if response.status_code == 200:
                print(f"Predict response: {response.json()}")
            else:
                print(f"Predict failed with status {response.status_code}: {response.text}")

    @task(1)  # Weight: 1x (less frequent than /predict)
    def test_retrain(self):
        """Simulate a user uploading a ZIP file to /retrain."""
        if not os.path.exists(SAMPLE_ZIP_PATH):
            print(f"Sample ZIP not found at {SAMPLE_ZIP_PATH}")
            return

        with open(SAMPLE_ZIP_PATH, "rb") as zip_file:
            files = {"files": ("dataset.zip", zip_file, "application/zip")}
            # Using Locust's HTTP client with query params
            response = self.client.post(
                "/retrain",
                files=files,
                data={"learning_rate": "0.0001", "epochs": "10"}
            )
            if response.status_code == 200:
                print(f"Retrain response: {response.json()}")
            else:
                print(f"Retrain failed with status {response.status_code}: {response.text}")
