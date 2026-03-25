import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print("health ok")


def test_retrieve():
    resp = requests.post(
        f"{BASE_URL}/api/retrieve",
        json={"query": "故意杀人罪如何认定？", "top_k": 3},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) > 0
    print("retrieve ok")


def test_chat():
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={"query": "故意杀人罪如何认定？"},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    print("chat ok")


if __name__ == "__main__":
    test_health()
    test_retrieve()
    test_chat()
    print("all api tests passed")