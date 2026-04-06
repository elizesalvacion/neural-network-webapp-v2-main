import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation="sigmoid"):
        np.random.seed(42)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Scaled initialization for more stable training (similar to Xavier/He)
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Invalid activation")

    # ----------------------------------------------------------
    # AUTO-PREPROCESSING (MNIST or XOR or anything else)
    # ----------------------------------------------------------
    def _preprocess_inputs(self, X):
        # If MNIST-style (N, 28, 28), flatten automatically
        if X.ndim == 3 and X.shape[1] == 28 and X.shape[2] == 28:
            X = X.reshape(X.shape[0], -1)

        # Normalize only if pixel values are large (MNIST)
        if X.max() > 1.0:
            X = X / 255.0

        return X.astype(np.float32)

    def _preprocess_labels(self, y):
        y = np.array(y)

        # Binary (XOR)
        if self.output_size == 1:
            return y.reshape(-1, 1)

        # If labels already one-hot, return as-is
        if y.ndim == 2 and y.shape[1] == self.output_size:
            return y

        # Otherwise convert integer labels → one-hot
        y_onehot = np.zeros((y.size, self.output_size))
        y_onehot[np.arange(y.size), y.astype(int)] = 1
        return y_onehot

    # ----------------------------------------------------------

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2

        if self.output_size == 1:
            self.a2 = sigmoid(self.z2)
        else:
            self.a2 = softmax(self.z2)

        return self.a2

    def train(self, X, y, epochs, lr, return_losses=False):
        # Auto-preprocess MNIST/XOR
        X = self._preprocess_inputs(X)
        y = self._preprocess_labels(y)

        losses = []

        for _ in range(epochs):
            output = self.forward(X)

            # MSE Loss function - quantify the predication err
            if self.output_size == 1:
                loss = np.mean((y - output) ** 2)
                d_output = (output - y)
            else:
                eps = 1e-12
                # Cross-Entropy Loss - suitable for multi-class problems with softmax output
                loss = -np.mean(np.sum(y * np.log(output + eps), axis=1))
                d_output = (output - y)

            losses.append(float(loss))

            d_hidden = (d_output @ self.W2.T) * self.activation_derivative(self.a1)

            self.W2 -= lr * (self.a1.T @ d_output)
            self.b2 -= lr * np.sum(d_output, axis=0, keepdims=True)

            self.W1 -= lr * (X.T @ d_hidden)
            self.b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True)

        if return_losses:
            return losses

    def evaluate(self, X, y, return_values=False):
        """Evaluate accuracy on given data.

        For binary problems (output_size == 1), y is expected to be shape (N, 1)
        with values 0/1. For multi-class, y can be either integer labels of shape
        (N,) or (N, 1), or one-hot vectors of shape (N, output_size).
        """

        # Auto-preprocess inputs only (do not alter labels here)
        X = self._preprocess_inputs(X)
        y = np.array(y)

        out = self.forward(X)

        if self.output_size == 1:
            y = y.reshape(-1, 1)
            preds = np.round(out)
            accuracy = float(np.mean(preds == y))
        else:
            pred_classes = np.argmax(out, axis=1).reshape(-1, 1)

            # If y is one-hot, convert to class indices; otherwise assume integers
            if y.ndim == 2 and y.shape[1] == self.output_size:
                true_classes = np.argmax(y, axis=1).reshape(-1, 1)
            else:
                true_classes = y.astype(int).reshape(-1, 1)

            accuracy = float(np.mean(pred_classes == true_classes))
            preds = pred_classes

        if return_values:
            return accuracy, preds

        return accuracy
