from gradio_client import Client
import time

def inspect():
    for _ in range(10):
        try:
            client = Client("http://127.0.0.1:7860/")
            print("Connected to UI.")
            client.view_api()
            return
        except Exception as e:
            print(f"Waiting for UI... {e}")
            time.sleep(2)

if __name__ == "__main__":
    inspect()
