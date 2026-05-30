import time
import pytest
from gradio_client import Client, handle_file

UI_URL = "http://127.0.0.1:7860/"

@pytest.fixture(scope="module")
def client():
    # Wait for server to be ready
    for _ in range(15):
        try:
            return Client(UI_URL)
        except Exception:
            time.sleep(2)
    pytest.skip("UI Server not responsive at " + UI_URL)

def test_ui_load_and_reason(client):
    """Test full pipeline from load to reasoning."""
    # api_name="/load_graph"
    res = client.predict(
        None, # file_obj
        "tests/fixtures/toy_graph.csv", # csv_path_text
        "Random (Fast)", # embedding_type
        api_name="/load_graph"
    )
    assert "[OK]" in res[0]
    
    # api_name="/run_reasoning"
    reason_res = client.predict(
        "newton", # query
        20, # beam_width
        3, # max_hop
        10, # top_k
        90, # mem_threshold
        api_name="/run_reasoning"
    )
    assert "newton" in reason_res[0]
    assert len(reason_res[1]) > 0
    assert reason_res[2] is not None

def test_ui_analytics(client):
    """Test structural analytics endpoint."""
    res = client.predict(api_name="/refresh_analytics")
    assert res[0] is not None
    assert "Nodes" in res[1]

def test_ui_weight_profiler(client):
    """Test real-time parameter radar (10 params: alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)."""
    res = client.predict(0.5, 0.5, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0, api_name="/update_param_profile")
    assert res is not None
    assert res["type"] == "plotly"

if __name__ == "__main__":
    # Manual run if needed
    c = Client(UI_URL)
    test_ui_load_and_reason(c)
    test_ui_analytics(c)
    test_ui_weight_profiler(c)
    print("UI Robustness Tests PASSED.")
