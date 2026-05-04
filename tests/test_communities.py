import requests
import json
import time

API_URL = "http://localhost:8200/v1"
HEADERS = {"X-API-Key": "dev-secret"}

def test_community_recompute():
    print("Testing Community Recompute...")
    # Request recompute with DSCF algorithm
    try:
        response = requests.post(
            f"{API_URL}/communities/recompute",
            headers=HEADERS,
            json={"algorithm": "dscf", "resolution": 1.0, "broadcast": True}
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", json.dumps(response.json(), indent=2)[:500] + "...")
        else:
            print("Response:", response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_community_recompute()
