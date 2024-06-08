import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your Next.js frontend during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuboRequest(BaseModel):
    qubo: Dict[str, Dict[str, float]]
    token: str
    region: str
    solver: str

class HamiltonianRequest(BaseModel):
    hamiltonian: List[List[float]]
    token: str
    region: str
    solver: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the QUBO Solver API"}

@app.post("/solve_qubo")
async def solve_qubo(request: QuboRequest):
    try:
        sampler = DWaveSampler(region=request.region, token=request.token, solver=request.solver)
        sampler = EmbeddingComposite(sampler)
        response = sampler.sample_qubo(request.qubo)
        solutions = response.record
        return {"solutions": solutions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve_hamiltonian")
async def solve_hamiltonian(request: HamiltonianRequest):
    try:
        logging.info("Begin solving")
        sampler = DWaveSampler(region=request.region, token=request.token, solver=request.solver)
        sampler = EmbeddingComposite(sampler)
        # Convert Hamiltonian to QUBO or use directly as required
        qubo = hamiltonian_to_qubo(request.hamiltonian)
        response = sampler.sample_qubo(qubo, num_reads=5000)
        solutions = response.record
        solutions = []

        for sample, energy, num_occurrences, cbf in response.record:
            logging.info(f"Sample: {sample}, energy: {energy}, num: {num_occurrences}")
            solutions.append({
                "sample": sample.tolist(),
                "energy": float(energy),
                "num_occurrences": int(num_occurrences)
            })

        return {"solutions": solutions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def hamiltonian_to_qubo(hamiltonian: List[List[float]]):
    qubo = {(i, i): 0.0 for i in range(len(hamiltonian))}
    
    # Necessary to keep the order of the sample columns consistent
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
