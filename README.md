<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/970498f7-008c-487b-b58e-5330f2770ca9" alt="Neural Visualizer" width="100%" />
</p> -->

<h1 align="center">Neural Visualizer</h1>

<p align="center">
  <strong>Interactive deep-learning exploration in your browser</strong><br/>
  Build, visualize, train, and export neural networks — from simple ANNs to Transformers & Diffusion models.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React_19-61DAFB?logo=react&logoColor=black&style=flat-square" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/Three.js-000000?logo=threedotjs&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/License-GPL_v3-blue?style=flat-square" />
</p>

---

## 🎬 Demo

<!-- <p align="center"> -->
https://github.com/user-attachments/assets/fc5b9114-0085-40b6-bdf9-fa3208ce19ff
<!-- </p> -->

---

## ✨ What's New in v3.0

The entire application has been **rebuilt from scratch** — migrated from a single-file Dash/Plotly app to a modern **React + FastAPI** architecture with 15+ interactive visualization tabs, in-browser training, and a polished dark-mode UI.

**Recent additions**
- **Guided Tour** — 20-step interactive tutorial with pulsing element highlights and arrow badges; auto-starts on first visit, re-launchable from the header
- **Input / Output node controls** — configure the exact number of input features (1–16) and output neurons (1–10) per architecture
- **Code Template Editor** — write your own code scaffold using `{{variables}}` that auto-fill from current settings in real time (alongside the existing PyTorch / Keras generators)
- **3D layout fix** — Transformer and other architectures with non-contiguous layer indices now render with even spacing in the 3D view

---

## 🧠 Supported Model Types

| Model | Description |
|---|---|
| **ANN** | Fully-connected feedforward network |
| **CNN** | Convolutional neural network with pooling layers |
| **RNN** | Vanilla recurrent network |
| **LSTM** | Long Short-Term Memory network |
| **GAN** | Generator + Discriminator adversarial pair |
| **Transformer** | Multi-head self-attention encoder |
| **Diffuser** | U-Net style encoder–decoder with time embedding |

---

## 🚀 Features

### Network Building & Visualization
- **Interactive Architecture Builder** — Configure input nodes (1–16), output nodes (1–10), hidden layers (1–5), neurons per layer, activation functions (ReLU, Sigmoid, Tanh, LeakyReLU, ELU, SELU), loss functions, and regularization (L1 / L2 / L1L2) from the sidebar
- **2D & 3D Network Graphs** — Toggle between a D3-powered 2D layout and a fully interactive Three.js 3D view with orbit controls
- **Forward Propagation** — Step-by-step animation of data flowing through each layer with active node/edge highlighting
- **Backpropagation** — Visualize gradient flow in reverse through the network

### Training & Analysis
- **Simulated Training** — Run server-side training with PyTorch; view training curves (loss & accuracy per epoch)
- **Live In-Browser Training (TF.js)** — Train your configured architecture entirely client-side using TensorFlow.js with real-time loss/accuracy canvas charts
- **Decision Boundaries** — See how a trained model partitions the 2D input space for classification tasks
- **Loss Landscape** — Explore the 3D optimization surface computed from the model's parameter space

### Advanced Visualizations
- **Weight Distribution Histograms** — Inspect per-layer weight distributions
- **Layer Activation Heatmaps** — Visualize activations across neurons and layers
- **Attention Heatmaps** — View self-attention weight matrices for Transformer models
- **Network Pruning** — Interactively prune connections and observe the effect on the architecture

### Tools
- **Learning Rate Sweep** — Batch-compare 5 learning rates side-by-side, trained with TF.js
- **Custom Activation Designer** — Draw your own activation function and watch it applied in a mini-network
- **Architecture Comparison** — Side-by-side 3D compare of two different network configurations with stat bars (nodes, edges, params)
- **Code Export** — Auto-generate ready-to-use PyTorch or Keras code from your current configuration; copy to clipboard with one click
- **Code Template Editor** — Switch to the Template tab to write your own scaffold using `{{variables}}` (`{{n_layers}}`, `{{input_nodes}}`, `{{loss_fn_code}}`, etc.) that fill in real time from the active config

### UI & Experience
- **Guided Tour** — 20-step interactive tutorial with pulsing highlights and directional arrow badges on every referenced UI element; auto-launches on first visit, re-accessible via the **Tour** button in the header; keyboard-navigable (← →, Esc)
- **4 Themes** — Dark, Cyberpunk, Matrix, Paper (light mode)
- **Cinema Mode** — Full-screen guided walkthrough of forward propagation with layer-by-layer narration, auto-play, and keyboard navigation
- **Dataset Preview** — Live scatter plot of the selected synthetic dataset
- **Custom Dataset Upload** — Load your own CSV data
- **Grouped Tab Bar** — 15 visualization tabs organized into Network · Analysis · Train · Tools groups
- **Framer Motion Animations** — Smooth tab transitions, status toasts, and micro-interactions throughout

### Datasets
Choose from **4 synthetic datasets** with adjustable noise:
- Circle · Gaussian · XOR · Spiral

---

## 🏗️ Architecture

```
Neural-Visualizer/
├── backend/                # Python FastAPI server
│   ├── main.py             # REST API endpoints
│   ├── models.py           # PyTorch model definitions (7 architectures)
│   ├── compute.py          # Graph building, propagation, training, landscapes
│   ├── datasets.py         # Synthetic dataset generators
│   └── requirements.txt
├── frontend/               # React 19 + TypeScript + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout/          # Header with theme switcher & Tour button
│   │   │   ├── Sidebar/         # NetworkConfig, TrainingConfig panels
│   │   │   ├── Visualizations/  # 15 visualization components
│   │   │   ├── CinemaMode.tsx   # Full-screen guided walkthrough
│   │   │   ├── Tutorial.tsx     # 20-step interactive guided tour
│   │   │   ├── DatasetPreview.tsx
│   │   │   └── DatasetUpload.tsx
│   │   ├── api/            # Axios API client
│   │   ├── store/          # Zustand state management
│   │   ├── contexts/       # Theme context
│   │   └── types/          # TypeScript type definitions
│   └── package.json
├── start.sh                # Launch both servers with one command
├── Visualization.py        # Legacy Dash app (preserved)
└── README.md
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS |
| **3D Rendering** | Three.js, React Three Fiber, Drei |
| **2D Charts** | D3.js, Plotly.js, HTML Canvas |
| **Animations** | Framer Motion |
| **State** | Zustand |
| **Icons** | Lucide React |
| **In-Browser ML** | TensorFlow.js |
| **Backend** | FastAPI, Uvicorn |
| **ML Engine** | PyTorch, scikit-learn, NumPy, SciPy |

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/PeakScripter/Neural-Visualizer.git
cd Neural-Visualizer
```

**2. Set up the backend**
```bash
cd backend
pip install -r requirements.txt
```

**3. Set up the frontend**
```bash
cd frontend
npm install
```

### Running the App

**Option A — Start both servers with one command (Linux/macOS)**
```bash
chmod +x start.sh
./start.sh
```

**Option B — Start each server separately**

Terminal 1 (Backend):
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open your browser at **http://localhost:5173**

> **Note:** The backend runs on port `8000` and the frontend dev server on port `5173`. The frontend proxies API calls to the backend.

---

## 📖 Usage

1. **Follow the Tour** — a 20-step guided tutorial launches automatically on first visit; click **Tour** in the header to reopen it at any time
2. **Select a model type** (ANN, CNN, RNN, LSTM, GAN, Transformer, Diffuser) and configure input nodes, output nodes, hidden layers, neurons, and activations in the sidebar
3. **Click "Build Network"** to generate the architecture graph
4. **Explore tabs** — switch between Architecture, Forward/Backward Propagation, Weights, Activations, Pruning, and more
5. **Toggle 2D/3D** to view the network in an interactive Three.js scene
6. **Configure training parameters** (dataset, noise, learning rate, batch size, epochs) and click **"Simulate Training"**
7. **View results** — training curves, decision boundaries, and loss landscapes
8. **Try Live Training** — train in-browser with TensorFlow.js and watch loss/accuracy update in real-time
9. **Launch Cinema Mode** for a narrated, auto-playing walkthrough of forward propagation
10. **Export code** — generate PyTorch or Keras code, or open the **Template** tab to write and preview your own code scaffold with `{{variables}}`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/build-network` | Build network graph from config |
| `GET` | `/api/network-graph` | Get current network graph |
| `POST` | `/api/forward-propagation` | Compute forward propagation steps |
| `POST` | `/api/backward-propagation` | Compute backward propagation steps |
| `POST` | `/api/decision-boundary` | Compute decision boundary |
| `POST` | `/api/loss-landscape` | Compute loss landscape surface |
| `POST` | `/api/simulate-training` | Run simulated training |
| `POST` | `/api/dataset` | Generate dataset preview |

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
