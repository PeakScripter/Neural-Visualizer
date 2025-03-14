# Neural Network Visualizer
![image](https://github.com/user-attachments/assets/970498f7-008c-487b-b58e-5330f2770ca9)

An interactive Dash application for visualizing neural network architectures, training dynamics, and decision boundaries across multiple model types (ANN, CNN, RNN, LSTM, GAN, Transformer, Diffuser) with real-time forward/backward propagation visualization.

## Features

- **Multiple Model Types**: Visualize different neural network architectures including ANNs, CNNs, RNNs, LSTMs, GANs, Transformers, and Diffusion models
- **Interactive Network Building**: Customize layer sizes, activation functions, and model-specific parameters
- **Forward/Backward Propagation**: Step-by-step visualization of data flow through the network
- **Decision Boundaries**: See how your model classifies data in real-time
- **Loss Landscapes**: Visualize the optimization surface
- **Dataset Generation**: Choose from various synthetic datasets with adjustable noise levels

## ⚠️ Work in Progress

**Real-time Training Visualization** is currently under development. This feature will allow users to observe network parameters and decision boundaries evolving during the training process.

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: dash, plotly, pytorch, numpy, pandas

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python Visualization.py
```

Then open your browser and navigate to `http://localhost:8050`

## Usage

1. Select a model type and problem type
2. Configure network architecture using the sliders and dropdowns
3. Click "Build Network" to visualize the architecture
4. Explore forward and backward propagation using the step sliders
5. View decision boundaries and loss landscapes in their respective tabs

## License

This project is licensed under the GNU - see the LICENSE file for details.

