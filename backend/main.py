from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np
from network import NeuralNetwork
import os
import gzip
import struct
from pathlib import Path
import urllib.request

app = FastAPI()

# Enable CORS so React can talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory storage for uploaded dataset ----
uploaded_dataset = {
    "X_train": None,
    "y_train": None,
    "X_test": None,
    "y_test": None,
}

# ---- Data Model ----
class TrainRequest(BaseModel):
    activation: str
    hidden_size: int
    epochs: int
    learning_rate: float
    dataset_type: str = "xor"  # "xor" or "uploaded"

# ---- XOR Dataset ----
def prepare_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)  # TRUE XOR
    return X, y

def _normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max normalize features column-wise to [0, 1]."""

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (X - mins) / denom

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MNIST_FILES = {
    "train_images": (
        "train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    ),
}

def _download_mnist_if_needed():
    for filename, url in MNIST_FILES.values():
        path = DATA_DIR / filename
        if not path.exists():
            urllib.request.urlretrieve(url, path.as_posix())

def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols).astype(np.float32)
        return data / 255.0  # normalize to [0, 1]

def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.int64).reshape(-1, 1)

def prepare_mnist_builtin():
    """
    Download (if needed) and load MNIST into NumPy arrays.

    Returns:
      X_train, y_train, X_test, y_test
      X: float32 in [0,1], shape (N, 784)
      y: int labels 0–9, shape (N, 1)
    """
    _download_mnist_if_needed()

    train_images_file, _ = MNIST_FILES["train_images"]
    train_labels_file, _ = MNIST_FILES["train_labels"]
    test_images_file, _ = MNIST_FILES["test_images"]
    test_labels_file, _ = MNIST_FILES["test_labels"]

    X_train = _read_idx_images(DATA_DIR / train_images_file)
    y_train = _read_idx_labels(DATA_DIR / train_labels_file)
    X_test = _read_idx_images(DATA_DIR / test_images_file)
    y_test = _read_idx_labels(DATA_DIR / test_labels_file)

    return X_train, y_train, X_test, y_test

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), test_ratio: float = Form(0.2)):
    # Basic validation of test_ratio
    if test_ratio <= 0 or test_ratio >= 1:
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read uploaded file as UTF-8 text")

    try:
        data = np.genfromtxt(
            text.splitlines(),
            delimiter=",",
            skip_header=1,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to parse CSV file")

    if data.ndim != 2 or data.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns (features + label)")

    if np.isnan(data).any():
        raise HTTPException(status_code=400, detail="CSV contains missing values (NaN), which are not supported")

    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(float).reshape(-1, 1)

    if X.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Dataset must contain at least 2 rows")

    X = _normalize_features(X)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_ratio)
    if test_size < 1:
        test_size = 1
    if test_size >= n_samples:
        test_size = n_samples - 1

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    uploaded_dataset["X_train"] = X_train
    uploaded_dataset["y_train"] = y_train
    uploaded_dataset["X_test"] = X_test
    uploaded_dataset["y_test"] = y_test

    return {
        "num_samples": int(n_samples),
        "num_features": int(X.shape[1]),
        "test_ratio": float(test_ratio),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

@app.post("/train")
def train_network(req: TrainRequest):
    # Select dataset
    if req.dataset_type == "xor":
        X, y = prepare_xor()
        X_eval, y_eval = X, y
    elif req.dataset_type == "uploaded":
        X_train = uploaded_dataset["X_train"]
        y_train = uploaded_dataset["y_train"]
        X_test = uploaded_dataset["X_test"]
        y_test = uploaded_dataset["y_test"]

        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise HTTPException(status_code=400, detail="No uploaded dataset found. Please upload a CSV first.")

        X, y = X_train, y_train
        X_eval, y_eval = X_test, y_test
    elif req.dataset_type == "mnist":
        X_train, y_train, X_test, y_test = prepare_mnist_builtin()
        X, y = X_train, y_train
        X_eval, y_eval = X_test, y_test
    else:
        raise HTTPException(status_code=400, detail="Invalid dataset_type. Use 'xor', 'uploaded' or 'mnist'.")

    # Determine if this is binary or multi-class
    # Assumes labels are integers 0..K-1 for multi-class (e.g. MNIST 0..9)
    unique_labels = np.unique(y)
    num_classes = int(unique_labels.max()) + 1

    if num_classes <= 2:
        # Binary classification – keep existing behavior
        output_size = 1
        y_train_for_net = y
        y_eval_for_net = y_eval
    else:
        # Multi-class: one-hot encode labels for training
        output_size = num_classes
        y_int = y.astype(int).reshape(-1)
        y_eval_int = y_eval.astype(int).reshape(-1, 1)

        eye = np.eye(num_classes, dtype=float)
        y_train_for_net = eye[y_int]
        y_eval_for_net = y_eval_int  # keep integers for evaluation

    # Adjust training configuration for MNIST to avoid very long runs
    if req.dataset_type == "mnist":
        max_train_samples = 20000
        X = X[:max_train_samples]
        y_train_for_net = y_train_for_net[:max_train_samples]
        epochs = min(req.epochs, 100)
    else:
        epochs = req.epochs

    nn = NeuralNetwork(
        input_size=X.shape[1],
        hidden_size=req.hidden_size,
        output_size=output_size,
        activation=req.activation,
    )

    losses = nn.train(
        X,
        y_train_for_net,
        epochs=epochs,
        lr=req.learning_rate,
        return_losses=True,
    )

    acc, preds = nn.evaluate(X_eval, y_eval_for_net, return_values=True)

    # Lightweight summaries for visualization: show a slice of weights and activations
    # Use the first hidden neuron and at most 10 inputs/samples.
    first_hidden_idx = 0

    # Input-to-hidden weights summary (first 10 features into first hidden neuron)
    weights_summary = None
    if hasattr(nn, "W1"):
        weights_summary = nn.W1[:10, first_hidden_idx].tolist()

    # Hidden activations summary on evaluation data (first 10 samples for first hidden neuron)
    activations_summary = None
    if X_eval is not None:
        X_vis = X_eval[:10]
        _ = nn.forward(nn._preprocess_inputs(X_vis))
        if hasattr(nn, "a1"):
            activations_summary = nn.a1[:10, first_hidden_idx].tolist()

    return {
        "losses": losses,
        "accuracy": acc,
        "predictions": preds.tolist(),  # XOR: (N,1) 0/1; MNIST: (N,1) predicted digit
        "dataset_type": req.dataset_type,
        "weights_summary": weights_summary,
        "activations_summary": activations_summary,
    }