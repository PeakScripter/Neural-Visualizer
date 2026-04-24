from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from models import MODEL_CLASSES
from datasets import generate_dataset
from compute import (
    build_network_graph,
    compute_decision_boundary,
    compute_loss_landscape,
    simulate_training,
    get_forward_propagation_steps,
    get_backward_propagation_steps,
)

app = FastAPI(title="Neural Visualizer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state
_network_graph = {"nodes": [], "edges": []}
_last_config = {}


class NetworkConfig(BaseModel):
    model_type: str = "ANN"
    n_layers: int = 3
    neurons: List[int] = [32, 32, 32]
    activations: List[str] = ["ReLU", "ReLU", "ReLU"]
    loss_fn: str = "Binary Cross Entropy"
    reg_type: str = "None"
    reg_rate: float = 0.01


class TrainingConfig(BaseModel):
    problem_type: str = "Classification"
    dataset: str = "Circle"
    noise: float = 10.0
    batch_size: int = 32
    learning_rate: float = 0.01
    reg_type: str = "None"
    reg_rate: float = 0.01
    epochs: int = 5
    loss_fn: str = "Binary Cross Entropy"


class FullConfig(BaseModel):
    network: NetworkConfig
    training: TrainingConfig


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/build-network")
def build_network(config: NetworkConfig):
    global _network_graph, _last_config
    _last_config = config.dict()
    neurons = config.neurons[:config.n_layers]
    activations = config.activations[:config.n_layers]
    _network_graph = build_network_graph(config.n_layers, neurons, activations, config.model_type)
    return _network_graph


@app.get("/api/network-graph")
def get_network_graph():
    return _network_graph


@app.post("/api/forward-propagation")
def get_forward_propagation(config: NetworkConfig):
    global _network_graph
    if not _network_graph["nodes"]:
        neurons = config.neurons[:config.n_layers]
        activations = config.activations[:config.n_layers]
        _network_graph = build_network_graph(config.n_layers, neurons, activations, config.model_type)
    steps = get_forward_propagation_steps(_network_graph["nodes"], _network_graph["edges"])
    return {"steps": steps, "graph": _network_graph}


@app.post("/api/backward-propagation")
def get_backward_propagation(config: NetworkConfig):
    global _network_graph
    if not _network_graph["nodes"]:
        neurons = config.neurons[:config.n_layers]
        activations = config.activations[:config.n_layers]
        _network_graph = build_network_graph(config.n_layers, neurons, activations, config.model_type)
    steps = get_backward_propagation_steps(_network_graph["nodes"], _network_graph["edges"])
    return {"steps": steps, "graph": _network_graph}


@app.post("/api/decision-boundary")
def get_decision_boundary(config: FullConfig):
    model_class = MODEL_CLASSES.get(config.network.model_type, MODEL_CLASSES["ANN"])
    neurons = config.network.neurons[:config.network.n_layers]
    activations = config.network.activations[:config.network.n_layers]
    result = compute_decision_boundary(
        model_class, neurons, activations,
        config.training.dataset, config.training.noise,
        config.training.problem_type
    )
    return result


@app.post("/api/loss-landscape")
def get_loss_landscape(config: NetworkConfig):
    model_class = MODEL_CLASSES.get(config.model_type, MODEL_CLASSES["ANN"])
    neurons = config.neurons[:config.n_layers]
    activations = config.activations[:config.n_layers]
    return compute_loss_landscape(model_class, neurons, activations)


@app.post("/api/simulate-training")
def run_training(config: FullConfig):
    neurons = config.network.neurons[:config.network.n_layers]
    activations = config.network.activations[:config.network.n_layers]
    result = simulate_training(
        neurons, activations,
        config.network.model_type,
        config.training.dataset,
        config.training.noise,
        config.training.learning_rate,
        config.training.batch_size,
        config.training.epochs,
        config.training.loss_fn,
        config.training.reg_type,
        config.training.reg_rate,
        config.training.problem_type,
    )
    return result


@app.post("/api/dataset")
def get_dataset_preview(config: TrainingConfig):
    X, y = generate_dataset(config.dataset, config.noise / 100.0)
    return {"X": X, "y": y}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
