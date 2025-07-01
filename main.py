import logging
import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from dwave.system import DWaveSampler, EmbeddingComposite
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np

# Load environment variables
load_dotenv()
EXPECTED_API_TOKEN = os.getenv("API_TOKEN")
DWAVE_TOKEN = os.getenv("DWAVE_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}))

# Auth Dependency
api_key_header = APIKeyHeader(name="Authorization")

def verify_token(token: str = Depends(api_key_header)):
    if token != EXPECTED_API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")
    return token

# Models
class QuboRequest(BaseModel):
    qubo: Dict[str, Dict[str, float]]
    region: str
    solver: str

class HamiltonianRequest(BaseModel):
    hamiltonian: List[List[float]]
    region: str
    solver: str

# Sampler caching
sampler_cache = {}

def get_sampler(region: str, solver: str):
    key = (region, solver)
    if key not in sampler_cache:
        sampler_cache[key] = EmbeddingComposite(
            DWaveSampler(region=region, token=DWAVE_TOKEN, solver=solver)
        )
    return sampler_cache[key]

# ENDPOINTS

@app.get("/", dependencies=[Depends(verify_token)])
@limiter.limit("20/minute")
async def read_root(request: Request):
    return {"message": "Welcome to the QUBO Solver API"}

@app.post("/solve_qubo", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def solve_qubo(request: QuboRequest):
    try:
        sampler = EmbeddingComposite(
            DWaveSampler(region=request.region, token=DWAVE_TOKEN, solver=request.solver)
        )
        response = sampler.sample_qubo(request.qubo)
        solutions = response.record
        return {"solutions": solutions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve_hamiltonian", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def solve_hamiltonian(request: HamiltonianRequest):
    try:
        sampler = get_sampler(request.region, request.solver)
        qubo = hamiltonian_to_qubo(request.hamiltonian)
        response = sampler.sample_qubo(qubo, num_reads=5000)

        solutions = [
            {
                "sample": sample.tolist(),
                "energy": float(energy),
                "num_occurrences": int(num_occurrences)
            }
            for sample, energy, num_occurrences, _ in response.record
        ]
        return {"solutions": solutions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def hamiltonian_to_qubo(hamiltonian: List[List[float]]):
    qubo = {(i, i): 0.0 for i in range(len(hamiltonian))}
    for index, value in np.ndenumerate(hamiltonian):
        if value != 0:
            qubo[index] = value
    return qubo

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
