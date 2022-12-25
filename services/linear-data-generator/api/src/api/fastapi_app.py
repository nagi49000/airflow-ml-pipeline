from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np


class HelloWorldResponse(BaseModel):
    message: str


class GetSamplesRequest(BaseModel):
    n_samples: int = 1


class GetSamplesResponse(BaseModel):
    samples: List[Tuple[float, float]]


class PutLineParamsRequest(BaseModel):
    gradient: float
    intercept: float
    noise_sigma: float


def create_app():
    app = FastAPI(title="Samples on a line generator")
    app.line_samples = {
        "gradient": 1.0,
        "intercept": 0.0,
        "noise_sigma": 0.1
    }

    @app.get("/hello", response_model=HelloWorldResponse)
    async def hello_world():
        return {"message": "Hello"}

    @app.post("/get-samples", response_model=GetSamplesResponse)
    async def get_samples(r: GetSamplesRequest):
        sigma = app.line_samples["noise_sigma"]
        m = app.line_samples["gradient"]
        c = app.line_samples["intercept"]
        # get the requested number of x vals in [0, 1]
        x_vals = np.random.rand(r.n_samples)
        # get the requested y values
        # which are vals on the line plus some noise
        y_vals = np.random.randn(r.n_samples) * sigma + m * x_vals + c
        samples = list(zip(x_vals, y_vals))
        return {"samples": samples}

    @app.put("/put-line-params")
    async def put_line_params(r: PutLineParamsRequest):
        app.line_samples["noise_sigma"] = r.noise_sigma
        app.line_samples["gradient"] = r.gradient
        app.line_samples["intercept"] = r.intercept

    return app
