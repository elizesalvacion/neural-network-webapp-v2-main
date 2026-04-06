# Backend – Neural Network API (FastAPI)

This backend exposes a simple neural network training API used by the React frontend.
It is built with **FastAPI**, uses **NumPy** for numerical computation, and is served with
**uvicorn**.

The API supports:

- **Training on a built‑in XOR dataset**
- **Training on a built‑in MNIST dataset** (digits 0–9)
- **Uploading a custom CSV dataset** (features + label) and training on it

---

## Tech Stack

- **Language:** Python 3
- **Framework:** FastAPI
- **Server:** uvicorn
- **Numerical library:** NumPy

Dependencies are listed in [requirements.txt]

---

## Folder Structure

```
backend/
├── main.py           # FastAPI app with /upload-dataset and /train routes
├── network.py        # Minimal NumPy neural net (sigmoid/ReLU, training & evaluation helpers)
├── requirements.txt  # FastAPI, uvicorn, NumPy
└── .venv/            # Local virtual environment (ignored in version control)
```

- `main.py` exposes the HTTP API and handles dataset storage/validation.
- `network.py` contains the `NeuralNetwork` class used during training.
- `requirements.txt` lists the Python dependencies needed to run the service.

---

## How the Backend Works

1. **Dataset preparation**
   - **Built‑in XOR dataset** is defined in [main.py](./main.py).
   - **Built‑in MNIST dataset** is downloaded on first use (using gzip IDX files) and cached under
     `backend/data/`. `prepare_mnist_builtin()` loads:
     - `X_train, y_train` (60k images) and `X_test, y_test` (10k images)
     - Images are flattened to 784‑dim vectors and normalized to `[0, 1]`.
   - For **uploaded CSVs**, `POST /upload-dataset`:
     - Reads the uploaded file as UTF‑8 text.
     - Parses comma‑separated values with `numpy.genfromtxt`.
     - Splits into:
       - `X`: all columns except the last (features)
       - `y`: last column (label), reshaped to a column vector
     - Min–max normalizes features column‑wise.
     - Randomly splits into **train** and **test** sets and stores them in memory in the
       `uploaded_dataset` dictionary.

2. **Training (binary vs multi‑class)**
   - `POST /train` selects the dataset based on `dataset_type`:
     - `"xor"` → built‑in XOR
     - `"uploaded"` → previously uploaded CSV split
     - `"mnist"` → built‑in MNIST
   - Labels are inspected to determine the number of classes:
     - For **binary problems** (≤ 2 unique labels):
       - `output_size = 1`
       - Training uses the labels directly (`y_train_for_net = y`).
       - Loss: **mean squared error (MSE)** with a sigmoid output.
     - For **multi‑class problems** (> 2 unique labels, e.g. MNIST 0–9):
       - `output_size = num_classes`.
       - Training labels are converted to **one‑hot** vectors.
       - Loss: **softmax + cross‑entropy**.

   - To keep MNIST training responsive on typical machines, the backend applies safety limits:
     - Uses only the first **20,000** training samples for MNIST.
     - Caps epochs for MNIST at **100** even if the frontend sends a larger number.

   - A [NeuralNetwork](./network.py) instance is created from `network.py` with:
     - `input_size`: number of feature columns
     - `hidden_size`: from request body
     - `activation`: `"sigmoid"` or `"relu"` from request body
     - `output_size`: 1 (binary) or `num_classes` (multi‑class)

   - The training loop is implemented entirely in **NumPy** using full‑batch gradient descent and
     returns a list of loss values per epoch.

3. **Evaluation and visualization data**
   - After training, `nn.evaluate(...)` computes accuracy on the evaluation split
     (XOR, uploaded test set, or MNIST test set).
   - The backend returns a JSON payload with:
     - `losses`: list of loss values per epoch
     - `accuracy`: scalar accuracy on the evaluation data
     - `predictions`: predicted outputs/classes for the evaluation data
     - `dataset_type`: `"xor"`, `"uploaded"`, or `"mnist"`
     - `weights_summary`: first few input‑to‑hidden weights feeding into hidden neuron #1
     - `activations_summary`: activations of hidden neuron #1 for the first few evaluation samples

4. **Frontend integration**
   - The React frontend talks to `http://localhost:8000`:
     - `POST /upload-dataset` with `FormData` when using a custom CSV
     - `POST /train` with the chosen hyperparameters and dataset type
   - The frontend visualizes:
     - Training loss over epochs (line chart)
     - Final accuracy
     - Sample predictions
     - The weight and activation summaries for one hidden neuron

---

## How to Run the Backend

These commands assume you are on **Windows** and start from the project root directory.

1. **Open a terminal**  
   PowerShell, Command Prompt, or Git Bash.

2. **Go to the backend folder**

   ```bash
   cd backend
   ```

3. **Activate the virtual environment** (if it exists)
   - PowerShell / CMD:

     ```powershell
     .\.venv\Scripts\activate
     ```

   - Git Bash:

     ```bash
     source venv/Scripts/activate
     ```

4. **Install Python dependencies** (first time or after changes)

   ```bash
   python -m pip install -r requirements.txt
   ```

5. **Start the FastAPI server with uvicorn**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   If `uvicorn` is not found, use:

   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. The API will be available at:
   - `http://localhost:8000`
   - Interactive docs at `http://localhost:8000/docs`

---

## Example API Usage

- **Upload CSV dataset**
  - Endpoint: `POST /upload-dataset`
  - Body: `multipart/form-data` with fields:
    - `file`: CSV file
    - `test_ratio` (optional, default `0.2`)

- **Train on XOR (built‑in)**
  - Endpoint: `POST /train`
  - JSON body:

    ```json
    {
      "activation": "sigmoid",
      "hidden_size": 4,
      "epochs": 2000,
      "learning_rate": 0.1,
      "dataset_type": "xor"
    }
    ```

- **Train on MNIST (built‑in)**
  - Endpoint: `POST /train`
  - JSON body:

    ```json
    {
      "activation": "sigmoid",
      "hidden_size": 64,
      "epochs": 500,
      "learning_rate": 0.0005,
      "dataset_type": "mnist"
    }
    ```

    The backend will internally cap MNIST to 20,000 training samples and at most 100 epochs.

- **Train on an uploaded CSV dataset**
  - First call `/upload-dataset`, then:

    ```json
    {
      "activation": "sigmoid",
      "hidden_size": 8,
      "epochs": 1000,
      "learning_rate": 0.01,
      "dataset_type": "uploaded"
    }
    ```

  - The response will always include `losses`, `accuracy`, `predictions`, `dataset_type`, and
    (when available) `weights_summary` and `activations_summary` for visualization.
