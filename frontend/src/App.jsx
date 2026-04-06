import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Brain, Settings, TrendingDown, CheckCircle } from "lucide-react";

function App() {
  const [activation, setActivation] = useState("sigmoid");
  const [hidden, setHidden] = useState(4);
  const [epochs, setEpochs] = useState(2000);
  const [lr, setLR] = useState(0.1);
  const [result, setResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [datasetType, setDatasetType] = useState("xor"); // "xor" or "uploaded"
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const uploadDataset = async () => {
    if (!file) {
      alert("Please select a CSV file first.");
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/upload-dataset", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        const msg = errData?.detail || "Failed to upload dataset";
        throw new Error(msg);
      }

      const data = await res.json();
      setUploadStatus({
        success: true,
        message: `Uploaded ${data.num_samples} samples with ${data.num_features} features. Train size: ${data.train_size}, Test size: ${data.test_size}.`,
      });
    } catch (error) {
      console.error("Dataset upload failed:", error);
      setUploadStatus({ success: false, message: error.message });
      alert(error.message || "Failed to upload dataset.");
    } finally {
      setIsUploading(false);
    }
  };

  const train = async () => {
    setIsTraining(true);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          activation,
          hidden_size: hidden,
          epochs,
          learning_rate: lr,
          dataset_type: datasetType,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        const msg = errData?.detail || "Training failed";
        throw new Error(msg);
      }

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Training failed:", error);
      alert(
        error.message ||
          "Failed to train. Make sure the backend is running on localhost:8000"
      );
    } finally {
      setIsTraining(false);
    }
  };

  const lossData =
    result?.losses?.map((loss, idx) => ({
      epoch: idx,
      loss: loss,
    })) || [];

  const xorInputs = [
    { a: 0, b: 0, expected: 0 },
    { a: 0, b: 1, expected: 1 },
    { a: 1, b: 0, expected: 1 },
    { a: 1, b: 1, expected: 0 },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto p-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-12 h-12 text-blue-400" />
            <h1 className="text-5xl font-bold text-white">
              Neural Network Trainer
            </h1>
          </div>
          <p className="text-slate-300 text-lg">
            Train a network to solve the XOR problem
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
              <div className="flex items-center gap-2 mb-6">
                <Settings className="w-6 h-6 text-blue-400" />
                <h2 className="text-2xl font-bold text-white">Configuration</h2>
              </div>

              <div className="space-y-5">
                {/* Dataset Type */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Dataset
                  </label>
                  <select
                    value={datasetType}
                    onChange={(e) => {
                      const value = e.target.value;
                      setDatasetType(value);
                      if (value === "mnist") {
                        setEpochs(20);
                      } else if (value === "xor") {
                        setEpochs(2000);
                      }
                    }}
                    className="w-full bg-white/5 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="xor">XOR (built-in)</option>
                    <option value="mnist">MNIST (built-in)</option>
                    <option value="uploaded">Uploaded CSV dataset</option>
                  </select>
                  <p className="text-xs text-slate-400 mt-1">
                    Choose between the built-in XOR dataset or your own CSV.
                  </p>
                </div>

                {/* File Upload (for uploaded dataset) */}
                {datasetType === "uploaded" && (
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Upload CSV Dataset
                    </label>
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => {
                        const selected = e.target.files?.[0] || null;
                        setFile(selected);
                        setUploadStatus(null);
                      }}
                      className="w-full text-slate-200 text-sm"
                    />
                    <button
                      type="button"
                      onClick={uploadDataset}
                      disabled={isUploading || !file}
                      className="mt-3 w-full bg-slate-700 text-white font-semibold py-2 px-4 rounded-lg hover:bg-slate-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isUploading ? "Uploading..." : "Upload Dataset"}
                    </button>
                    {uploadStatus && (
                      <p
                        className={`mt-2 text-xs ${
                          uploadStatus.success
                            ? "text-emerald-300"
                            : "text-red-300"
                        }`}
                      >
                        {uploadStatus.message}
                      </p>
                    )}
                  </div>
                )}

                {/* Activation Function */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Activation Function
                  </label>
                  <select
                    value={activation}
                    onChange={(e) => setActivation(e.target.value)}
                    className="w-full bg-white/5 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="sigmoid">Sigmoid</option>
                    <option value="relu">ReLU</option>
                  </select>
                </div>

                {/* Hidden Size */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Hidden Layer Size
                  </label>
                  <input
                    type="number"
                    value={hidden}
                    onChange={(e) => setHidden(Number(e.target.value))}
                    className="w-full bg-white/5 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="1"
                    max="20"
                  />
                  <p className="text-xs text-slate-400 mt-1">
                    Number of neurons in hidden layer
                  </p>
                </div>

                {/* Epochs */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Training Epochs
                  </label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-full bg-white/5 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="100"
                    step="100"
                  />
                </div>

                {/* Learning Rate */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Learning Rate
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={lr}
                    onChange={(e) => setLR(Number(e.target.value))}
                    className="w-full bg-white/5 border border-white/30 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                    min="0.001"
                    max="1"
                  />
                </div>

                {/* Train Button */}
                <button
                  onClick={train}
                  disabled={
                    isTraining ||
                    (datasetType === "uploaded" && !uploadStatus?.success)
                  }
                  className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-cyan-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                >
                  {isTraining ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Training...
                    </span>
                  ) : (
                    "Start Training"
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                {/* Accuracy Card */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-slate-300 text-sm font-medium mb-1">
                        Model Accuracy
                      </p>
                      <p className="text-5xl font-bold text-white">
                        {(result.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                    <CheckCircle
                      className={`w-16 h-16 ${
                        result.accuracy === 1
                          ? "text-green-400"
                          : "text-yellow-400"
                      }`}
                    />
                  </div>
                </div>

                {/* Loss Curve */}
                <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingDown className="w-6 h-6 text-blue-400" />
                    <h3 className="text-xl font-bold text-white">
                      Training Loss
                    </h3>
                  </div>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={lossData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                      <XAxis
                        dataKey="epoch"
                        stroke="#94a3b8"
                        label={{
                          value: "Epoch",
                          position: "insideBottom",
                          offset: -5,
                          fill: "#94a3b8",
                        }}
                      />
                      <YAxis
                        stroke="#94a3b8"
                        label={{
                          value: "Loss",
                          angle: -90,
                          position: "insideLeft",
                          fill: "#94a3b8",
                        }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1e293b",
                          border: "1px solid #475569",
                          borderRadius: "8px",
                          color: "#fff",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#60a5fa"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-center text-slate-300 text-sm mt-2">
                    Final Loss:{" "}
                    {result.losses[result.losses.length - 1].toFixed(6)}
                  </p>
                </div>

                {/* Weights & Activations Summary */}
                {(result.weights_summary || result.activations_summary) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {result.weights_summary && (
                      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                        <h3 className="text-lg font-bold text-white mb-2">
                          Weights into Hidden Neuron #1
                        </h3>
                        <p className="text-xs text-slate-400 mb-3">
                          Showing the first few input weights feeding into the first hidden neuron.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {result.weights_summary.map((w, idx) => (
                            <div
                              key={idx}
                              className="px-2 py-1 rounded-md bg-slate-900/60 text-xs text-slate-100 font-mono"
                            >
                              w{idx}: {w.toFixed(3)}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.activations_summary && (
                      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                        <h3 className="text-lg font-bold text-white mb-2">
                          Hidden Neuron #1 Activations
                        </h3>
                        <p className="text-xs text-slate-400 mb-3">
                          Activations of the first hidden neuron for the first few evaluation samples.
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {result.activations_summary.map((a, idx) => (
                            <div
                              key={idx}
                              className="px-2 py-1 rounded-md bg-slate-900/60 text-xs text-slate-100 font-mono"
                            >
                              a{idx}: {a.toFixed(3)}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Predictions */}
                {result.dataset_type === "xor" ? (
                  <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                    <h3 className="text-xl font-bold text-white mb-4">
                      XOR Predictions
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      {xorInputs.map((input, idx) => {
                        const prediction = result.predictions[idx][0];
                        const isCorrect =
                          Math.round(prediction) === input.expected;

                        return (
                          <div
                            key={idx}
                            className={`p-4 rounded-lg border-2 ${
                              isCorrect
                                ? "bg-green-500/20 border-green-400"
                                : "bg-red-500/20 border-red-400"
                            }`}
                          >
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-white font-mono text-lg">
                                {input.a} XOR {input.b}
                              </span>
                              <span
                                className={`font-bold ${
                                  isCorrect ? "text-green-400" : "text-red-400"
                                }`}
                              >
                                {isCorrect ? "✓" : "✗"}
                              </span>
                            </div>
                            <div className="text-sm text-slate-300">
                              <div>
                                Expected:{" "}
                                <span className="font-bold text-white">
                                  {input.expected}
                                </span>
                              </div>
                              <div>
                                Predicted:{" "}
                                <span className="font-bold text-white">
                                  {prediction.toFixed(4)}
                                </span>
                              </div>
                              <div>
                                Rounded:{" "}
                                <span className="font-bold text-white">
                                  {Math.round(prediction)}
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-2xl">
                    <h3 className="text-xl font-bold text-white mb-4">
                      Test Predictions (Uploaded Dataset)
                    </h3>
                    <p className="text-slate-300 text-sm mb-4">
                      Showing the first few prediction values from the test set.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {result.predictions.slice(0, 9).map((row, idx) => (
                        <div
                          key={idx}
                          className="p-3 rounded-lg border border-white/20 bg-slate-900/40"
                        >
                          <div className="text-xs text-slate-400 mb-1">
                            Sample #{idx + 1}
                          </div>
                          <div className="text-sm text-white font-mono">
                            {row[0].toFixed(4)}
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-slate-400 mt-4">
                      Total predictions: {result.predictions.length}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 shadow-2xl text-center">
                <Brain className="w-20 h-20 text-blue-400 mx-auto mb-4 opacity-50" />
                <p className="text-slate-300 text-lg">
                  Configure your network and click "Start Training" to begin
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
