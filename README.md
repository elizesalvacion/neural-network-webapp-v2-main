# Neural Network Web Application

A full-stack web application for training and visualizing simple neural networks. Built with **React + Vite** on the frontend and **FastAPI + NumPy** on the backend, this project demonstrates supervised learning with an intuitive user interface for configuring, training, and inspecting neural networks across multiple datasets.

---

## Overview

**Neural Network Trainer** lets you:

- Choose from **built-in datasets** (XOR, MNIST) or **upload your own CSV file**
- Configure **hyperparameters**:
  - Activation function (sigmoid or ReLU)
  - Hidden layer size
  - Training epochs
  - Learning rate
- Train a neural network and visualize:
  - **Loss curve** over epochs
  - **Accuracy** on test data
  - **Predictions** on samples
  - **Weight and activation details** from the hidden layer
- Understand neural networks through interactive experimentation

---

## Project Structure

```
neural-network-webapp-v2/
├── frontend/                  # React + Vite UI
│   ├── src/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── README.md
│
├── backend/                   # FastAPI server
│   ├── main.py               # API endpoints
│   ├── network.py            # Neural network implementation
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── data/                 # MNIST dataset (auto-downloaded)
│   └── README.md
│
└── README.md                 # (This file)
```

---

## Tech Stack

### Frontend

- **Framework:** React 19 with TypeScript support
- **Build tool:** Vite
- **Styling:** Tailwind CSS 4
- **Charts:** Recharts (for loss curves and accuracy visualization)
- **Icons:** Lucide React
- **Linting:** ESLint 9

### Backend

- **Language:** Python 3.14+
- **Framework:** FastAPI
- **Server:** Uvicorn
- **Numerical computation:** NumPy 2
- **File handling:** python-multipart

---

## Getting Started

### Prerequisites

- **Node.js** (16+ recommended) for frontend
- **Python** (3.14+) for backend
- **pip** for Python package management

### Backend Setup

1. **Navigate to the backend folder:**

   ```bash
   cd backend
   ```

2. **(Optional) Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Or using pyproject.toml:

   ```bash
   pip install -e .
   ```

4. **Start the FastAPI server:**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`
   - **API docs:** `http://localhost:8000/docs` (Swagger UI)
   - **Alternative docs:** `http://localhost:8000/redoc` (ReDoc)

### Frontend Setup

1. **Navigate to the frontend folder:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173` (or the URL shown by Vite)

### Running Both Together

Open two terminals:

**Terminal 1 (Backend):**

```bash
cd backend
uvicorn main:app --reload
```

**Terminal 2 (Frontend):**

```bash
cd frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## How It Works

### 1. **Dataset Selection**

The app supports three dataset types:

- **XOR**: A built-in 4-sample dataset for binary classification (True XOR function)
- **MNIST**: Handwritten digit dataset (60k training, 10k test images)
  - Automatically downloaded on first use
  - Cached in `backend/data/` for future runs
- **Custom CSV**: Upload your own dataset with features + labels

### 2. **Data Processing**

- All numerical features are **normalized to [0, 1]** using min-max scaling
- MNIST images are flattened from 28×28 to 784-dimensional vectors and normalized
- Uploaded CSV datasets are split into 80% train / 20% test sets

### 3. **Neural Network Architecture**

A **2-layer fully-connected network** (1 hidden layer):

```
Input → [Hidden Layer] → Output
```

- **Activation functions:** Sigmoid or ReLU in hidden layer
- **Output layer:** Sigmoid (binary classification) or Softmax (multi-class)
- **Loss function:** MSE (binary) or Cross-Entropy (multi-class)
- **Optimization:** Full-batch gradient descent

### 4. **Training**

- Entire network is implemented in **NumPy** (no external ML libraries)
- Weights initialized with Xavier/He scaling for stable training
- **Safety limits:** MNIST training uses max 20k samples and caps epochs at 100 (configurable)
- Returns loss per epoch and final accuracy

### 5. **Visualization & Inspection**

After training, the frontend displays:

- **Loss Curve:** How well the network learned over epochs
- **Accuracy:** Final test set accuracy
- **Predictions:** Sample outputs on test data
- **Network Internals:** Weights into the first hidden neuron and its activations on sample inputs

---

## API Endpoints

### `POST /upload-dataset`

Upload a custom CSV dataset.

**Request body (form-data):**

- `file`: CSV file with features + label (last column)

**Response:**

```json
{
  "filename": "data.csv",
  "rows": 100,
  "features": 4,
  "message": "Dataset uploaded successfully"
}
```

### `POST /train`

Train the neural network.

**Request body (JSON):**

```json
{
  "dataset_type": "xor",
  "activation": "sigmoid",
  "hidden_size": 8,
  "epochs": 100,
  "learning_rate": 0.1
}
```

**Response:**

```json
{
  "losses": [0.25, 0.24, 0.22, ...],
  "accuracy": 0.95,
  "predictions": [[0.1], [0.9], ...],
  "dataset_type": "xor",
  "weights_summary": [[-0.5, 0.3], ...],
  "activations_summary": [[0.2, 0.8], ...]
}
```

---

## Development

### Frontend Commands

```bash
npm run dev       # Start development server (with hot reload)
npm run build     # Build for production
npm run preview   # Preview production build locally
npm run lint      # Run ESLint checks
```

### Backend Commands

```bash
uvicorn main:app --reload              # Development with auto-restart
uvicorn main:app --host 0.0.0.0        # Expose to network (for testing on other machines)
```

---

## Key Features Explained

### Binary vs. Multi-class Classification

The backend automatically detects the problem type:

- **Binary (≤ 2 classes):** Single output neuron with sigmoid activation; uses MSE loss
- **Multi-class (> 2 classes):** `num_classes` output neurons with softmax activation; uses cross-entropy loss

This allows the same network architecture to handle both XOR (binary) and MNIST (10-class) problems seamlessly.

### MNIST Optimization

MNIST training on typical machines can be slow. The backend applies practical limits:

- Uses first **20,000** training samples (instead of all 60k)
- Caps epochs at **100** (even if frontend requests more)

These can be adjusted in `backend/main.py` if you have faster hardware.

### Automatic Data Downloading

The first time you select MNIST, the backend automatically:

1. Downloads the MNIST gzip files from Google's MNIST download server
2. Unpacks and caches them in `backend/data/`
3. Loads train/test splits into NumPy arrays

Subsequent runs use the cached data (no re-downloads).

---

## Troubleshooting

### Frontend can't reach the backend

**Error:** `Failed to fetch` or `CORS error`

**Solution:**

- Ensure the backend is running on `http://localhost:8000`
- Check that CORS middleware is enabled (it is by default in `main.py`)
- Verify no firewall is blocking port 8000

### MNIST takes a long time to download

**Reason:** First-time download of ~30 MB of data over the network

**Solution:**

- Wait for the initial download to complete
- Subsequent runs will use cached data (instant)
- Check your internet connection if the download fails

### Backend crashes with "port already in use"

**Solution:**

```bash
# Kill the process using port 8000 (Windows PowerShell)
Get-Process | Where-Object {$_.Handles -match 8000} | Stop-Process

# Or use a different port
uvicorn main:app --port 8001 --reload
```

### Training is very slow

**Reason:** Large dataset or many epochs/neurons combined

**Solutions:**

- Reduce `hidden_size` (fewer neurons = faster training)
- Reduce `epochs`
- For MNIST, the backend automatically limits to 20k samples

---

## Project Highlights

✓ **Full-stack implementation** — Frontend and backend from scratch  
✓ **Neural network from NumPy** — No external ML frameworks for the network itself  
✓ **Multiple datasets** — XOR, MNIST, custom CSV support  
✓ **Interactive UI** — Real-time parameter configuration and visualization  
✓ **Educational** — Learn how neural networks work by experimenting  
✓ **Production-ready structure** — Proper error handling, CORS configuration, and modular code

---

## Future Enhancements

Potential improvements:

- [ ] Multi-layer networks (3+ layers)
- [ ] Regularization (L1/L2 dropout)
- [ ] Batch gradient descent with mini-batches
- [ ] More activation functions (tanh, ELU)
- [ ] Model persistence (save/load trained weights)
- [ ] Real-time training progress (WebSockets)
- [ ] Additional built-in datasets (CIFAR-10, fashion-MNIST)

---

## License

This project is provided as-is for educational purposes.

---

## Contact & Support

For issues, questions, or suggestions, please open an issue in the repository or contact the project maintainer.

Happy neural network training! 🧠
