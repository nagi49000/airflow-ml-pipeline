from pytest import (
    fixture,
    approx
)
import numpy as np
from starlette.testclient import TestClient
from api.src.api.api_app import create_app


@fixture()
def client():
    with TestClient(create_app()) as client:
        yield client


def test_hello(client):
    response = client.get("/hello")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello"}


def test_get_samples(client):
    np.random.seed(0)

    response = client.post("/get-samples", json={})
    assert response.status_code == 200
    j = response.json()
    assert j == {
        "samples": [
            [approx(0.5488135039273248), approx(0.6229726780082409)]
        ]
    }

    response = client.post("/get-samples", json={"n_samples": 3})
    assert response.status_code == 200
    j = response.json()
    assert j == {
        "samples":
        [
            [approx(0.5448831829968969), approx(0.7001745551881785)],
            [approx(0.4236547993389047), approx(0.5186636410914637)],
            [approx(0.6458941130666561), approx(0.6307583922368863)],
        ]
    }


def test_put_line_params(client):
    np.random.seed(0)

    # check that state updates
    put_json = {
        "gradient": -1.0,
        "intercept": 100.0,
        "noise_sigma": 0.0
    }
    response = client.put("/put-line-params", json=put_json)
    assert response.status_code == 200

    # check that the new state has an impact on numbers coming out
    response = client.post("/get-samples", json={})
    assert response.status_code == 200
    j = response.json()
    assert j == {
        "samples": [
            [approx(0.5488135039273248), approx(99.45118649607268)]
        ]
    }
