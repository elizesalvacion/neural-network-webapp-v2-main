# Frontend – Neural Network Trainer (React + Vite)

This frontend is a React + Vite single‑page app that talks to the FastAPI backend to train a
simple NumPy neural network and visualize its behavior.

It lets you:

- Choose a **dataset**:
  - XOR (built‑in)
  - MNIST digits (built‑in, loaded by the backend)
  - Uploaded CSV dataset (features + label)
- Configure **hyperparameters**:
  - Activation function: `sigmoid` or `relu`
  - Hidden layer size
  - Number of training epochs
  - Learning rate
- Run training and inspect:
  - Training **loss curve** over epochs
  - Final **accuracy**
  - Sample **predictions**
  - A small summary of **weights** into one hidden neuron and its **activations** on a few samples

---

## How to Run the Frontend

1. Install dependencies (from the `frontend/` folder):

   ```bash
   npm install
   ```

2. Start the dev server:

   ```bash
   npm run dev
   ```

3. Open the URL printed by Vite (usually `http://localhost:5173`).

4. Make sure the **backend** FastAPI server is running on `http://localhost:8000` so that:

   - `POST /upload-dataset` is available for CSV uploads.
   - `POST /train` is available for training on XOR, MNIST, or the uploaded dataset.

---

## Development Notes

- This project was bootstrapped with the standard `create-vite` React template, but the default
  README has been replaced with documentation specific to the Neural Network Trainer UI.
- Styling is done with Tailwind‑style utility classes (compiled by Vite) and a small set of
  component libraries (e.g., Lucide icons, Recharts for charts).

