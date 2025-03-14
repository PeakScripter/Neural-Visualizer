import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL, ClientsideFunction
import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.exceptions import PreventUpdate
from collections import OrderedDict
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
import json
import time

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # For deployment

# Add near the top of the file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define dataset generation functions
def generate_circle_data(n_samples=200):
    X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    return X, y

def generate_gaussian_data(n_samples=200):
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42)
    return X, y

def generate_xor_data(n_samples=200):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=1)
    # Transform to make it more XOR-like
    mask = np.logical_or(np.logical_and(X[:, 0] > 0, X[:, 1] > 0), 
                         np.logical_and(X[:, 0] < 0, X[:, 1] < 0))
    y = mask.astype(int)
    return X, y

def generate_spiral_data(n_samples=200):
    n_samples_per_class = n_samples // 2
    n_classes = 2
    n_turns = 2
    
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_classes):
        ix = range(n_samples_per_class * i, n_samples_per_class * (i + 1))
        r = np.linspace(0.0, 1, n_samples_per_class)
        t = np.linspace(i * n_turns, (i + 1) * n_turns, n_samples_per_class) + np.random.randn(n_samples_per_class) * 0.1
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = i
    
    return X, y

# Define the neural network models
class CustomNN(nn.Module):
    def __init__(self, layers, activations):
        super(CustomNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations  # Store activations as instance variable
        input_size = 2  # Assuming 2D input for decision boundary visualization
        for neurons, activation in zip(layers, activations):
            self.layers.append(nn.Linear(input_size, neurons))
            input_size = neurons
        self.layers.append(nn.Linear(input_size, 1))  # Output layer (binary classification)
    
    def forward(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):  # Use self.activations
            x = layer(x)
            if activation == "ReLU":
                x = torch.relu(x)
            elif activation == "Sigmoid":
                x = torch.sigmoid(x)
            elif activation == "Tanh":
                x = torch.tanh(x)
            else:
                x = torch.relu(x)  # Default
        return torch.sigmoid(self.layers[-1](x))  # Apply sigmoid to output for binary classification

class CustomCNN(nn.Module):
    def __init__(self, layers, activations):
        super(CustomCNN, self).__init__()
        self.activations = activations
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        # Convolutional layers
        in_channels = 1  # Grayscale input
        for i, neurons in enumerate(layers):
            self.conv_layers.append(nn.Conv2d(in_channels, neurons, kernel_size=3, padding=1))
            in_channels = neurons
        
        # Fully connected layers
        self.fc_layers.append(nn.Linear(layers[-1] * 4 * 4, 64))  # Assuming 4x4 feature maps after pooling
        self.fc_layers.append(nn.Linear(64, 1))
    
    def forward(self, x):
        # Reshape input to [batch_size, channels, height, width]
        if x.dim() == 2:
            x = x.view(-1, 1, 8, 8)  # Reshape 2D input to image format (8x8)
        
        # Apply convolutional layers with pooling
        for i, (conv, activation) in enumerate(zip(self.conv_layers, self.activations)):
            x = conv(x)
            if activation == "ReLU":
                x = F.relu(x)
            elif activation == "Sigmoid":
                x = torch.sigmoid(x)
            elif activation == "Tanh":
                x = torch.tanh(x)
            else:
                x = F.relu(x)
            
            # Apply max pooling after each conv layer
            if i < len(self.conv_layers) - 1:  # No pooling after last conv layer
                x = F.max_pool2d(x, 2)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc_layers[0](x))
        x = torch.sigmoid(self.fc_layers[1](x))
        
        return x

class CustomRNN(nn.Module):
    def __init__(self, layers, activations):
        super(CustomRNN, self).__init__()
        self.activations = activations
        hidden_size = layers[0]
        self.rnn = nn.RNN(input_size=2, hidden_size=hidden_size, num_layers=len(layers), batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Reshape input to [batch_size, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply RNN
        x, _ = self.rnn(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layer with sigmoid
        x = torch.sigmoid(self.fc(x))
        
        return x

class CustomLSTM(nn.Module):
    def __init__(self, layers, activations):
        super(CustomLSTM, self).__init__()
        self.activations = activations
        hidden_size = layers[0]
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=len(layers), batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Reshape input to [batch_size, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layer with sigmoid
        x = torch.sigmoid(self.fc(x))
        
        return x

class CustomGAN(nn.Module):
    def __init__(self, layers, activations):
        super(CustomGAN, self).__init__()
        self.activations = activations
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(10, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0], layers[1] if len(layers) > 1 else 64),
            nn.ReLU(),
            nn.Linear(layers[1] if len(layers) > 1 else 64, 2),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(2, layers[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(layers[0], layers[1] if len(layers) > 1 else 64),
            nn.LeakyReLU(0.2),
            nn.Linear(layers[1] if len(layers) > 1 else 64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # For visualization purposes, we'll use the discriminator
        return self.discriminator(x)
    
    def generate(self, z):
        return self.generator(z)

class CustomTransformer(nn.Module):
    def __init__(self, layers, activations):
        super(CustomTransformer, self).__init__()
        self.activations = activations
        d_model = layers[0]
        nhead = 2  # Number of attention heads
        
        # Embedding layer
        self.embedding = nn.Linear(2, d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=len(layers))
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # Reshape input to [batch_size, seq_len, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Take the output from the last position
        x = x[:, -1, :]
        
        # Apply fully connected layer with sigmoid
        x = torch.sigmoid(self.fc(x))
        
        return x

class CustomDiffuser(nn.Module):
    def __init__(self, layers, activations):
        super(CustomDiffuser, self).__init__()
        self.activations = activations
        
        # U-Net inspired architecture (simplified)
        # Encoder
        self.encoder1 = nn.Linear(2, layers[0])
        self.encoder2 = nn.Linear(layers[0], layers[1] if len(layers) > 1 else 64)
        
        # Bottleneck
        self.bottleneck = nn.Linear(layers[1] if len(layers) > 1 else 64, layers[1] if len(layers) > 1 else 64)
        
        # Decoder
        self.decoder1 = nn.Linear(layers[1] if len(layers) > 1 else 64, layers[0])
        self.decoder2 = nn.Linear(layers[0], 2)
        
        # Time embedding
        self.time_embed = nn.Linear(1, layers[0])
        
        # Output layer
        self.output = nn.Linear(2, 1)
    
    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.size(0), 1)
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Encoder
        e1 = F.relu(self.encoder1(x))
        e1 = e1 + t_emb  # Add time embedding
        e2 = F.relu(self.encoder2(e1))
        
        # Bottleneck
        b = F.relu(self.bottleneck(e2))
        
        # Decoder with skip connections
        d1 = F.relu(self.decoder1(b))
        d1 = d1 + e1  # Skip connection
        d2 = F.relu(self.decoder2(d1))
        
        # For decision boundary visualization
        out = torch.sigmoid(self.output(d2))
        
        return out

# Global variables to store network state
network_state = {
    'nodes': [],
    'edges': [],
    'node_activations': {},
    'node_gradients': {},
    'weights': {},
    'biases': {},
    'layers': [],
    'activations': [],
    'n_layers': 3,
    'model_type': 'ANN'  # Default model type
}

# Define model-specific parameters
model_specific_params = {
    'ANN': {
        'losses': ['Binary Cross Entropy', 'Mean Squared Error', 'Hinge Loss'],
        'default_loss': 'Binary Cross Entropy',
        'regularizations': ['None', 'L1', 'L2', 'L1L2'],
        'default_regularization': 'None'
    },
    'CNN': {
        'losses': ['Binary Cross Entropy', 'Categorical Cross Entropy', 'Mean Squared Error'],
        'default_loss': 'Categorical Cross Entropy',
        'kernel_sizes': [1, 3, 5, 7],
        'default_kernel': 3,
        'pool_sizes': [2, 3],
        'default_pool': 2,
        'regularizations': ['None', 'L1', 'L2', 'L1L2'],
        'default_regularization': 'None'
    },
    'RNN': {
        'losses': ['Binary Cross Entropy', 'Mean Squared Error', 'Sequence Loss'],
        'default_loss': 'Sequence Loss',
        'sequence_lengths': [1, 3, 5, 10],
        'default_seq_len': 3,
        'regularizations': ['None', 'L1', 'L2', 'L1L2'],
        'default_regularization': 'None'
    },
    'LSTM': {
        'losses': ['Binary Cross Entropy', 'Mean Squared Error', 'Sequence Loss'],
        'default_loss': 'Sequence Loss',
        'sequence_lengths': [1, 3, 5, 10],
        'default_seq_len': 3,
        'regularizations': ['None', 'L1', 'L2', 'L1L2'],
        'default_regularization': 'None'
    },
    'GAN': {
        'losses': ['Binary Cross Entropy', 'Wasserstein Loss', 'Least Squares'],
        'default_loss': 'Binary Cross Entropy',
        'latent_dims': [10, 32, 64, 100],
        'default_latent_dim': 10,
        'regularizations': ['None', 'Gradient Penalty', 'Spectral Norm'],
        'default_regularization': 'None'
    },
    'Transformer': {
        'losses': ['Binary Cross Entropy', 'Cross Entropy', 'Masked Language Model Loss'],
        'default_loss': 'Cross Entropy',
        'num_heads': [1, 2, 4, 8],
        'default_heads': 2,
        'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.5],
        'default_dropout': 0.1,
        'regularizations': ['None', 'L1', 'L2', 'L1L2'],
        'default_regularization': 'None'
    },
    'Diffuser': {
        'losses': ['Mean Squared Error', 'L1 Loss', 'Huber Loss'],
        'default_loss': 'Mean Squared Error',
        'time_steps': [10, 50, 100, 1000],
        'default_time_steps': 100,
        'noise_schedules': ['Linear', 'Cosine', 'Quadratic'],
        'default_schedule': 'Linear',
        'regularizations': ['None', 'L2', 'Gradient Clipping'],
        'default_regularization': 'None'
    }
}

# Add global settings for training and data
global_settings = {
    'problem_types': ['Classification', 'Regression'],
    'default_problem': 'Classification',
    'datasets': ['Circle', 'Gaussian', 'XOR', 'Spiral'],
    'default_dataset': 'Circle',
    'batch_sizes': [1, 8, 16, 32, 64, 128],
    'default_batch_size': 32,
    'learning_rates': [0.0001, 0.001, 0.01, 0.1],
    'default_learning_rate': 0.01,
    'regularization_rates': [0.0001, 0.001, 0.01, 0.1],
    'default_reg_rate': 0.01,
    'noise_levels': [0, 5, 10, 15, 20],
    'default_noise': 10
}

# Add a global variable to track training state after the other global variables
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'loss_history': [],
    'accuracy_history': []
}

# Define the app layout with Bootstrap components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Neural Network Visualizer", className="text-center my-4"),
            html.P("An interactive 3D visualization tool for neural networks", className="text-center lead mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        # Left sidebar for network configuration
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Network Configuration"),
                dbc.CardBody([
                    html.Label("Model Type:"),
                    dcc.Dropdown(
                        id='model-type-dropdown',
                        options=[
                            {'label': 'Artificial Neural Network (ANN)', 'value': 'ANN'},
                            {'label': 'Convolutional Neural Network (CNN)', 'value': 'CNN'},
                            {'label': 'Recurrent Neural Network (RNN)', 'value': 'RNN'},
                            {'label': 'Long Short-Term Memory (LSTM)', 'value': 'LSTM'},
                            {'label': 'Generative Adversarial Network (GAN)', 'value': 'GAN'},
                            {'label': 'Transformer', 'value': 'Transformer'},
                            {'label': 'Diffusion Model', 'value': 'Diffuser'}
                        ],
                        value='ANN',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("Loss Function:"),
                    dcc.Dropdown(
                        id='loss-function-dropdown',
                        options=[
                            {'label': 'Binary Cross Entropy', 'value': 'Binary Cross Entropy'},
                            {'label': 'Mean Squared Error', 'value': 'Mean Squared Error'}
                        ],
                        value='Binary Cross Entropy',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Div(id='model-specific-params', children=[]),
                    html.Label("Number of Layers:"),
                    dcc.Slider(
                        id='n-layers-slider',
                        min=1,
                        max=5,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 6)},
                        className="mb-4"
                    ),
                    html.Div(id='layer-controls', children=[]),
                    dbc.Button('Build Network', id='build-button', color="primary", className="mt-3 w-100")
                ])
            ]),
            
            dbc.Card([
                dbc.CardHeader("Training Parameters"),
                dbc.CardBody([
                    html.Label("Problem Type:"),
                    dcc.Dropdown(
                        id='problem-type-dropdown',
                        options=[
                            {'label': 'Classification', 'value': 'Classification'},
                            {'label': 'Regression', 'value': 'Regression'}
                        ],
                        value='Classification',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("Dataset:"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[
                            {'label': 'Circle', 'value': 'Circle'},
                            {'label': 'Gaussian', 'value': 'Gaussian'},
                            {'label': 'XOR', 'value': 'XOR'},
                            {'label': 'Spiral', 'value': 'Spiral'}
                        ],
                        value='Circle',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("Noise Level:"),
                    dcc.Slider(
                        id='noise-slider',
                        min=0,
                        max=20,
                        step=5,
                        value=10,
                        marks={i: f"{i}%" for i in [0, 5, 10, 15, 20]},
                        className="mb-3"
                    ),
                    html.Label("Batch Size:"),
                    dcc.Dropdown(
                        id='batch-size-dropdown',
                        options=[
                            {'label': '1 (SGD)', 'value': 1},
                            {'label': '8', 'value': 8},
                            {'label': '16', 'value': 16},
                            {'label': '32', 'value': 32},
                            {'label': '64', 'value': 64},
                            {'label': '128', 'value': 128}
                        ],
                        value=32,
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("Learning Rate:"),
                    dcc.Slider(
                        id='learning-rate-slider',
                        min=-4,
                        max=-1,
                        step=0.1,
                        value=-2,
                        marks={i: f"1e{i}" for i in range(-4, 0)},
                        className="mb-3"
                    ),
                    html.Label("Regularization:"),
                    dcc.Dropdown(
                        id='regularization-dropdown',
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'L1', 'value': 'L1'},
                            {'label': 'L2', 'value': 'L2'},
                            {'label': 'L1L2', 'value': 'L1L2'}
                        ],
                        value='None',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("Regularization Rate:"),
                    dcc.Slider(
                        id='reg-rate-slider',
                        min=-4,
                        max=-1,
                        step=0.1,
                        value=-2,
                        marks={i: f"1e{i}" for i in range(-4, 0)},
                        className="mb-3",
                        disabled=True  # Initially disabled
                    ),
                    html.Label("Epochs:"),
                    dcc.Slider(
                        id='epochs-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 11, 2)},
                        className="mb-3"
                    ),
                    dbc.Button('Simulate Training', id='train-button', color="success", className="w-100")
                ])
            ], className="mt-3")
        ], width=3),
        
        # Main content area
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dcc.Graph(id='network-graph', style={'height': '70vh'})
                ], label="Network Architecture", tab_id="tab-architecture"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Propagation Step:"),
                            dcc.Slider(
                                id='forward-step-slider',
                                min=0,
                                max=4,
                                step=1,
                                value=0,
                                marks={i: f'Step {i+1}' for i in range(5)},
                                className="mb-3"
                            ),
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='forward-graph', style={'height': '60vh'})
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Forward Propagation Details"),
                                dbc.CardBody(id='forward-explanation')
                            ], className="h-100")
                        ], width=4)
                    ])
                ], label="Forward Propagation", tab_id="tab-forward"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Backpropagation Step:"),
                            dcc.Slider(
                                id='backward-step-slider',
                                min=0,
                                max=4,
                                step=1,
                                value=0,
                                marks={i: f'Step {i+1}' for i in range(5)},
                                className="mb-3"
                            ),
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='backward-graph', style={'height': '60vh'})
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Backpropagation Details"),
                                dbc.CardBody(id='backward-explanation')
                            ], className="h-100")
                        ], width=4)
                    ])
                ], label="Backpropagation", tab_id="tab-backward"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='decision-boundary', style={'height': '70vh'})
                        ], width=12)
                    ])
                ], label="Decision Boundary", tab_id="tab-decision"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='loss-landscape', style={'height': '70vh'})
                        ], width=12)
                    ])
                ], label="Loss Landscape", tab_id="tab-loss"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='training-curve', style={'height': '70vh'})
                        ], width=12)
                    ])
                ], label="Training Curve", tab_id="tab-training"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            # Training controls
                            dbc.Card([
                                dbc.CardHeader("Training Controls"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Learning Rate:"),
                                            dcc.Slider(
                                                id='realtime-learning-rate-slider',
                                                min=-4,
                                                max=-1,
                                                step=1,
                                                marks={i: f"10^{i}" for i in range(-4, 0)},
                                                value=-2  # Default 0.01
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Epochs:"),
                                            dcc.Slider(
                                                id='realtime-epochs-slider',
                                                min=10,
                                                max=1000,
                                                step=10,
                                                marks={i: str(i) for i in [10, 100, 500, 1000]},
                                                value=100
                                            )
                                        ], width=6)
                                    ]),
                                    html.Div(className="mb-3"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("Start Training", id="start-training-button", color="success", className="w-100"),
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Button("Stop Training", id="stop-training-button", color="danger", className="w-100"),
                                        ], width=6)
                                    ])
                                ])
                            ]),
                            
                            # Real-time training visualization
                            dbc.Card([
                                dbc.CardHeader("Training Progress"),
                                dbc.CardBody([
                                    html.Div(id="training-status", children=[
                                        html.P("Not training", id="training-message"),
                                        dbc.Progress(id="training-progress", value=0, striped=True, animated=True)
                                    ]),
                                    html.Div(className="mb-3"),
                                    dcc.Graph(id="training-metrics", style={"height": "300px"}),
                                    # Hidden div for storing training state
                                    html.Div(id="is-training", style={"display": "none"}, **{"data-json": "false"}),
                                    # Interval component for updating visualizations
                                    dcc.Interval(id="training-interval", interval=100, disabled=True)
                                ])
                            ], className="mt-3"),
                            
                            # Real-time decision boundary visualization
                            dbc.Card([
                                dbc.CardHeader("Real-time Decision Boundary"),
                                dbc.CardBody([
                                    dcc.Graph(id="realtime-decision-boundary", style={"height": "400px"})
                                ])
                            ], className="mt-3"),
                            
                            # Add this to the network architecture tab layout
                            html.Div([
                                dbc.Button(
                                    "View Training Animation", 
                                    id="view-training-button", 
                                    color="info", 
                                    className="mt-3"
                                )
                            ])
                        ], width=12)
                    ])
                ], label="Real-time Training", tab_id="tab-realtime-training")
            ], id="tabs", active_tab="tab-architecture")
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Node Inspector"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Layer:"),
                            dcc.Dropdown(id='layer-selector', placeholder="Select Layer")
                        ], width=6),
                        dbc.Col([
                            html.Label("Select Neuron:"),
                            dcc.Dropdown(id='neuron-selector', placeholder="Select Neuron")
                        ], width=6)
                    ]),
                    html.Div(id='node-details', className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mt-3 mb-5")
], fluid=True)

# Callback to generate layer controls based on number of layers
@app.callback(
    Output('layer-controls', 'children'),
    [Input('n-layers-slider', 'value'),
     Input('model-type-dropdown', 'value')]
)
def update_layer_controls(n_layers, model_type):
    controls = []
    
    for i in range(n_layers):
        layer_controls = [
            html.Label(f"Layer {i+1} Neurons:"),
            dcc.Dropdown(
                id={'type': 'layer-neurons', 'index': i+1},
                options=[{'label': str(n), 'value': n} for n in [8, 16, 32, 64, 128]],
                value=32,
                clearable=False
            ),
            html.Label(f"Layer {i+1} Activation:"),
            dcc.Dropdown(
                id={'type': 'layer-activation', 'index': i+1},
                options=[
                    {'label': 'ReLU', 'value': 'ReLU'},
                    {'label': 'Sigmoid', 'value': 'Sigmoid'},
                    {'label': 'Tanh', 'value': 'Tanh'},
                    {'label': 'LeakyReLU', 'value': 'LeakyReLU'},
                    {'label': 'ELU', 'value': 'ELU'},
                    {'label': 'SELU', 'value': 'SELU'}
                ],
                value='ReLU',
                clearable=False
            ),
            html.Label(f"Layer {i+1} Weight Initialization:"),
            dcc.Dropdown(
                id={'type': 'layer-weight-init', 'index': i+1},
                options=[
                    {'label': 'Xavier/Glorot', 'value': 'glorot_uniform'},
                    {'label': 'He (for ReLU)', 'value': 'he_uniform'},
                    {'label': 'Normal', 'value': 'normal'},
                    {'label': 'Uniform', 'value': 'uniform'},
                    {'label': 'Zeros', 'value': 'zeros'},
                    {'label': 'Ones', 'value': 'ones'}
                ],
                value='glorot_uniform',
                clearable=False
            )
        ]
        
        # Add model-specific layer parameters
        if model_type == 'CNN':
            if i < n_layers - 1:  # All but the last layer
                layer_controls.extend([
                    html.Label(f"Layer {i+1} Kernel Size:"),
                    dcc.Dropdown(
                        id={'type': 'layer-kernel-size', 'index': i+1},
                        options=[{'label': f"{k}x{k}", 'value': k} for k in [1, 3, 5, 7]],
                        value=3,
                        clearable=False
                    ),
                    html.Label(f"Layer {i+1} Stride:"),
                    dcc.Dropdown(
                        id={'type': 'layer-stride', 'index': i+1},
                        options=[{'label': str(s), 'value': s} for s in [1, 2]],
                        value=1,
                        clearable=False
                    ),
                    html.Label(f"Layer {i+1} Pooling:"),
                    dcc.Dropdown(
                        id={'type': 'layer-pooling', 'index': i+1},
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'Max Pooling', 'value': 'Max'},
                            {'label': 'Average Pooling', 'value': 'Avg'}
                        ],
                        value='Max',
                        clearable=False
                    )
                ])
        
        elif model_type == 'Transformer':
            layer_controls.extend([
                html.Label(f"Layer {i+1} Attention Heads:"),
                dcc.Dropdown(
                    id={'type': 'layer-attention-heads', 'index': i+1},
                    options=[{'label': str(h), 'value': h} for h in [1, 2, 4, 8]],
                    value=2,
                    clearable=False
                ),
                html.Label(f"Layer {i+1} Dropout:"),
                dcc.Dropdown(
                    id={'type': 'layer-dropout', 'index': i+1},
                    options=[{'label': str(d), 'value': d} for d in [0.0, 0.1, 0.2, 0.3, 0.5]],
                    value=0.1,
                    clearable=False
                )
            ])
        
        elif model_type == 'Diffuser':
            if i < n_layers // 2:  # Encoder layers
                layer_controls.extend([
                    html.Label(f"Encoder {i+1} Normalization:"),
                    dcc.Dropdown(
                        id={'type': 'layer-norm', 'index': i+1},
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'Batch Norm', 'value': 'Batch'},
                            {'label': 'Layer Norm', 'value': 'Layer'}
                        ],
                        value='Batch',
                        clearable=False
                    )
                ])
            else:  # Decoder layers
                layer_controls.extend([
                    html.Label(f"Decoder {i+1-n_layers//2} Skip Connection:"),
                    dcc.Dropdown(
                        id={'type': 'layer-skip', 'index': i+1},
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'Add', 'value': 'Add'},
                            {'label': 'Concatenate', 'value': 'Concat'}
                        ],
                        value='Add',
                        clearable=False
                    )
                ])
        
        controls.extend(layer_controls)
        controls.append(html.Hr())
    
    return controls

# Function to build the network
def build_network(n_layers, neurons, activations, model_type='ANN', model_params=None):
    # Reset network state
    network_state['nodes'] = []
    network_state['edges'] = []
    network_state['node_activations'] = {}
    network_state['node_gradients'] = {}
    network_state['weights'] = {}
    network_state['biases'] = {}
    network_state['layers'] = neurons[:n_layers]
    network_state['activations'] = activations[:n_layers]
    network_state['n_layers'] = n_layers
    network_state['model_type'] = model_type
    
    # Store model-specific parameters
    if model_params:
        network_state['model_params'] = model_params
    
    # Generate random weights and biases for visualization
    np.random.seed(42)  # For reproducibility
    
    # Input layer values (sample input)
    input_values = [0.5, -0.3]  # Sample input for visualization
    input_neurons = 2
    
    # Input layer
    for i in range(input_neurons):
        network_state['nodes'].append({
            'x': 0, 
            'y': i-(input_neurons-1)/2, 
            'z': 0,
            'name': f"Input {i+1}",
            'color': "blue",
            'layer_type': "input",
            'neuron_idx': i,
            'layer_idx': 0,
            'value': input_values[i]
        })
        network_state['node_activations'][(0, i)] = input_values[i]
        network_state['node_gradients'][(0, i)] = 0.1  # Simulated gradient
    
    # Model-specific architecture
    if model_type == 'CNN':
        # For CNN, we'll visualize convolutional layers differently
        z_pos = 1
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights and bias for this layer
            if layer_idx == 0:
                # First conv layer connects to input
                kernel_size = 3
                w = np.random.randn(neuron_count, 1, kernel_size, kernel_size) * 0.1
                b = np.random.randn(neuron_count) * 0.1
            else:
                # Other conv layers connect to previous conv layer
                kernel_size = 3
                w = np.random.randn(neuron_count, neurons[layer_idx-1], kernel_size, kernel_size) * 0.1
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][layer_idx+1] = w
            network_state['biases'][layer_idx+1] = b
            
            # Create feature map nodes (simplified)
            for i in range(neuron_count):
                # Simulate activation
                a_val = np.random.rand()
                network_state['node_activations'][(layer_idx+1, i)] = a_val
                network_state['node_gradients'][(layer_idx+1, i)] = np.random.rand() * 0.2
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': 0,
                    'name': f"Conv{layer_idx+1}F{i+1}",
                    'color': "green",
                    'layer_type': "conv",
                    'neuron_idx': i,
                    'layer_idx': layer_idx+1,
                    'activation': activation,
                    'a_val': float(a_val),
                    'kernel_size': kernel_size
                })
                
                # Connect to previous layer (simplified)
                if layer_idx == 0:
                    for j in range(input_neurons):
                        weight_val = float(np.mean(w[i]))  # Average weight for visualization
                        network_state['edges'].append({
                            'source': j,
                            'target': input_neurons + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                else:
                    prev_layer_start = input_neurons + sum(neurons[:layer_idx-1])
                    for j in range(neurons[layer_idx-1]):
                        weight_val = float(np.mean(w[i, j]))  # Average weight for visualization
                        network_state['edges'].append({
                            'source': prev_layer_start + j,
                            'target': input_neurons + sum(neurons[:layer_idx]) + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
            z_pos += 1
        
        # Add FC layers
        fc_neurons = 64
        fc_layer_idx = n_layers + 1
        
        # Create FC layer
        w_fc = np.random.randn(fc_neurons, neurons[n_layers-1] * 4 * 4) * 0.1  # Assuming 4x4 feature maps
        b_fc = np.random.randn(fc_neurons) * 0.1
        
        network_state['weights'][fc_layer_idx] = w_fc
        network_state['biases'][fc_layer_idx] = b_fc
        
        for i in range(fc_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][(fc_layer_idx, i)] = a_val
            network_state['node_gradients'][(fc_layer_idx, i)] = np.random.rand() * 0.2
            
            # Create node
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(fc_neurons-1)/4,  # Compress y-axis for FC layer
                'z': 0,
                'name': f"FC{i+1}",
                'color': "purple",
                'layer_type': "fc",
                'neuron_idx': i,
                'layer_idx': fc_layer_idx,
                'activation': "ReLU",
                'a_val': float(a_val)
            })
            
            # Connect to last conv layer (simplified)
            last_conv_start = input_neurons + sum(neurons[:n_layers-1])
            for j in range(neurons[n_layers-1]):
                weight_val = float(np.mean(w_fc[i, j*16:(j+1)*16]))  # Average weight for visualization
                network_state['edges'].append({
                    'source': last_conv_start + j,
                    'target': input_neurons + sum(neurons) + i,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
        
        z_pos += 1
        
        # Output layer
        output_neurons = 1
        output_layer_idx = fc_layer_idx + 1
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, fc_neurons) * 0.1
        b_out = np.random.randn(output_neurons) * 0.1
        
        network_state['weights'][output_layer_idx] = w_out
        network_state['biases'][output_layer_idx] = b_out
        
        for i in range(output_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][(output_layer_idx, i)] = a_val
            network_state['node_gradients'][(output_layer_idx, i)] = 0.5
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0,
                'name': f"Output",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': output_layer_idx,
                'activation': "Sigmoid",
                'a_val': float(a_val)
            })
            
            # Connect to FC layer
            fc_start = input_neurons + sum(neurons) 
            for j in range(fc_neurons):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': fc_start + j,
                    'target': fc_start + fc_neurons + i,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    elif model_type in ['RNN', 'LSTM']:
        # For RNN/LSTM, we'll visualize with time steps
        z_pos = 1
        time_steps = 3  # Number of time steps to visualize
        
        # Create recurrent layers
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights
            if layer_idx == 0:
                # Input to hidden
                w_ih = np.random.randn(neuron_count, input_neurons) * 0.5
                # Hidden to hidden (recurrent)
                w_hh = np.random.randn(neuron_count, neuron_count) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            else:
                # Layer to layer
                w_ih = np.random.randn(neuron_count, neurons[layer_idx-1]) * 0.5
                # Hidden to hidden (recurrent)
                w_hh = np.random.randn(neuron_count, neuron_count) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][f"{layer_idx+1}_ih"] = w_ih
            network_state['weights'][f"{layer_idx+1}_hh"] = w_hh
            network_state['biases'][layer_idx+1] = b
            
            # Create nodes for each time step
            for t in range(time_steps):
                for i in range(neuron_count):
                    # Simulate activation
                    a_val = np.random.rand()
                    network_state['node_activations'][(layer_idx+1, i, t)] = a_val
                    network_state['node_gradients'][(layer_idx+1, i, t)] = np.random.rand() * 0.2
                    
                    # Create node
                    node_type = "lstm" if model_type == "LSTM" else "rnn"
                    network_state['nodes'].append({
                        'x': z_pos + t*0.3,  # Offset for time steps
                        'y': i-(neuron_count-1)/2, 
                        'z': t*0.2,  # Use z for time dimension
                        'name': f"{model_type}{layer_idx+1}N{i+1}T{t+1}",
                        'color': "green",
                        'layer_type': node_type,
                        'neuron_idx': i,
                        'layer_idx': layer_idx+1,
                        'time_step': t,
                        'activation': activation,
                        'a_val': float(a_val)
                    })
                    
                    # Connect to previous layer at same time step
                    if layer_idx == 0 and t == 0:
                        # First time step, first layer connects to input
                        for j in range(input_neurons):
                            weight_val = float(w_ih[i, j])
                            network_state['edges'].append({
                                'source': j,
                                'target': input_neurons + i,
                                'value': 1,
                                'weight': weight_val,
                                'color': 'red' if weight_val < 0 else 'green',
                                'width': abs(weight_val) * 3
                            })
                    elif layer_idx > 0 and t == 0:
                        # First time step, higher layers
                        prev_layer_start = input_neurons + (layer_idx-1) * neuron_count * time_steps
                        curr_node_idx = input_neurons + layer_idx * neuron_count * time_steps + i
                        for j in range(neurons[layer_idx-1]):
                            prev_node_idx = prev_layer_start + j
                            weight_val = float(w_ih[i, j])
                            network_state['edges'].append({
                                'source': prev_node_idx,
                                'target': curr_node_idx,
                                'value': 1,
                                'weight': weight_val,
                                'color': 'red' if weight_val < 0 else 'green',
                                'width': abs(weight_val) * 3
                            })
                    
                    # Recurrent connections (from previous time step)
                    if t > 0:
                        # Connect to same neuron at previous time step
                        prev_time_idx = input_neurons + layer_idx * neuron_count * time_steps + i + (t-1) * neuron_count
                        curr_node_idx = input_neurons + layer_idx * neuron_count * time_steps + i + t * neuron_count
                        weight_val = float(w_hh[i, i])
                        network_state['edges'].append({
                            'source': prev_time_idx,
                            'target': curr_node_idx,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'orange',  # Highlight recurrent connections
                            'width': abs(weight_val) * 3,
                            'dash': 'dash'  # Dashed line for recurrent
                        })
            
            z_pos += 1
        
        # Output layer
        output_neurons = 1
        output_layer_idx = n_layers + 1
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, neurons[n_layers-1]) * 0.5
        b_out = np.random.randn(output_neurons) * 0.1
        
        network_state['weights'][output_layer_idx] = w_out
        network_state['biases'][output_layer_idx] = b_out
        
        for i in range(output_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][(output_layer_idx, i)] = a_val
            network_state['node_gradients'][(output_layer_idx, i)] = 0.5
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0,
                'name': f"Output",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': output_layer_idx,
                'activation': "Sigmoid",
                'a_val': float(a_val)
            })
            
            # Connect to last hidden layer (last time step)
            last_hidden_start = input_neurons + (n_layers-1) * neuron_count * time_steps + (time_steps-1) * neuron_count
            for j in range(neurons[n_layers-1]):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': last_hidden_start + j,
                    'target': input_neurons + n_layers * neuron_count * time_steps + i,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    elif model_type == 'GAN':
        # For GAN, we'll visualize both generator and discriminator
        z_pos = 1
        
        # Generator
        gen_input_size = 10
        gen_nodes_start = input_neurons
        
        # Create generator input nodes (latent space)
        for i in range(gen_input_size):
            if i < 5:  # Only show a few nodes to avoid clutter
                z_val = np.random.randn()
                network_state['nodes'].append({
                    'x': -1,  # Place before input
                    'y': i-(min(5, gen_input_size)-1)/2, 
                    'z': 0,
                    'name': f"z{i+1}",
                    'color': "purple",
                    'layer_type': "latent",
                    'neuron_idx': i,
                    'layer_idx': -1,
                    'value': float(z_val)
                })
        
        # Generator hidden layers
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights and bias
            if layer_idx == 0:
                w = np.random.randn(neuron_count, gen_input_size) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            else:
                w = np.random.randn(neuron_count, neurons[layer_idx-1]) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][f"gen_{layer_idx+1}"] = w
            network_state['biases'][f"gen_{layer_idx+1}"] = b
            
            for i in range(neuron_count):
                # Simulate activation
                a_val = np.random.rand()
                network_state['node_activations'][(f"gen_{layer_idx+1}", i)] = a_val
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': -0.5,  # Place generator below
                    'name': f"G{layer_idx+1}N{i+1}",
                    'color': "lightblue",
                    'layer_type': "generator",
                    'neuron_idx': i,
                    'layer_idx': f"gen_{layer_idx+1}",
                    'activation': activation,
                    'a_val': float(a_val)
                })
            
            z_pos += 0.5
        
        # Generator output (2D for visualization)
        gen_output_size = 2
        for i in range(gen_output_size):
            # Simulate activation
            a_val = np.random.rand() * 2 - 1  # tanh output
            network_state['node_activations'][(f"gen_output", i)] = a_val
            
            # Create node
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(gen_output_size-1)/2, 
                'z': -0.5,
                'name': f"G_out{i+1}",
                'color': "blue",
                'layer_type': "generator_output",
                'neuron_idx': i,
                'layer_idx': "gen_output",
                'activation': "Tanh",
                'a_val': float(a_val)
            })
        
        # Connect generator nodes
        # (simplified - not showing all connections to avoid clutter)
        
        # Reset for discriminator
        z_pos = 1
        
        # Discriminator hidden layers
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights and bias
            if layer_idx == 0:
                w = np.random.randn(neuron_count, input_neurons) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            else:
                w = np.random.randn(neuron_count, neurons[layer_idx-1]) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][layer_idx+1] = w
            network_state['biases'][layer_idx+1] = b
            
            for i in range(neuron_count):
                # Simulate activation
                a_val = np.random.rand()
                network_state['node_activations'][(layer_idx+1, i)] = a_val
                network_state['node_gradients'][(layer_idx+1, i)] = np.random.rand() * 0.2
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': 0.5,  # Place discriminator above
                    'name': f"D{layer_idx+1}N{i+1}",
                    'color': "green",
                    'layer_type': "discriminator",
                    'neuron_idx': i,
                    'layer_idx': layer_idx+1,
                    'activation': activation,
                    'a_val': float(a_val)
                })
                
                # Connect to previous layer
                if layer_idx == 0:
                    for j in range(input_neurons):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': j,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                else:
                    disc_prev_layer_start = len(network_state['nodes']) - neuron_count - neurons[layer_idx-1]
                    for j in range(neurons[layer_idx-1]):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': disc_prev_layer_start + j,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
            
            z_pos += 0.5
        
        # Discriminator output
        output_neurons = 1
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, neurons[n_layers-1]) * 0.5
        b_out = np.random.randn(output_neurons) * 0.1
        
        network_state['weights'][n_layers+1] = w_out
        network_state['biases'][n_layers+1] = b_out
        
        for i in range(output_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][(n_layers+1, i)] = a_val
            network_state['node_gradients'][(n_layers+1, i)] = 0.5
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0.5,
                'name': f"D_out",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': n_layers+1,
                'activation': "Sigmoid",
                'a_val': float(a_val)
            })
            
            # Connect to last discriminator layer
            disc_last_layer_start = len(network_state['nodes']) - output_neurons - neurons[n_layers-1]
            for j in range(neurons[n_layers-1]):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': disc_last_layer_start + j,
                    'target': len(network_state['nodes'])-1,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    elif model_type == 'Transformer':
        # For Transformer, we'll visualize attention mechanism
        z_pos = 1
        
        # Create transformer layers
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights
            if layer_idx == 0:
                # Input embedding
                w_emb = np.random.randn(neuron_count, input_neurons) * 0.5
                b_emb = np.random.randn(neuron_count) * 0.1
                network_state['weights'][f"emb_{layer_idx+1}"] = w_emb
                network_state['biases'][f"emb_{layer_idx+1}"] = b_emb
                
                # Create embedding nodes
                for i in range(neuron_count):
                    # Simulate activation
                    a_val = np.random.rand()
                    network_state['node_activations'][(f"emb_{layer_idx+1}", i)] = a_val
                    
                    # Create node
                    network_state['nodes'].append({
                        'x': z_pos, 
                        'y': i-(neuron_count-1)/2, 
                        'z': 0,
                        'name': f"Emb{i+1}",
                        'color': "purple",
                        'layer_type': "embedding",
                        'neuron_idx': i,
                        'layer_idx': f"emb_{layer_idx+1}",
                        'activation': activation,
                        'a_val': float(a_val)
                    })
                    
                    # Connect to input
                    for j in range(input_neurons):
                        weight_val = float(w_emb[i, j])
                        network_state['edges'].append({
                            'source': j,
                            'target': input_neurons + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                
                z_pos += 1
                
                # Attention heads
                n_heads = 2
                for h in range(n_heads):
                    # Create attention nodes
                    for i in range(neuron_count):
                        # Simulate activation
                        a_val = np.random.rand()
                        network_state['node_activations'][(f"attn_{layer_idx+1}_{h}", i)] = a_val
                        
                        # Create node
                        network_state['nodes'].append({
                            'x': z_pos, 
                            'y': i-(neuron_count-1)/2, 
                            'z': h*0.5 - 0.25,  # Offset heads in z direction
                            'name': f"A{layer_idx+1}H{h+1}N{i+1}",
                            'color': "orange",
                            'layer_type': "attention",
                            'neuron_idx': i,
                            'layer_idx': f"attn_{layer_idx+1}_{h}",
                            'head': h,
                            'a_val': float(a_val)
                        })
                    
                    # Add attention connections (simplified)
                    attn_start = len(network_state['nodes']) - neuron_count
                    emb_start = input_neurons
                    
                    # Show some attention connections
                    for i in range(neuron_count):
                        for j in range(min(3, neuron_count)):  # Limit connections for clarity
                            weight_val = np.random.rand() * 0.5
                            network_state['edges'].append({
                                'source': emb_start + j,
                                'target': attn_start + i,
                                'value': 1,
                                'weight': weight_val,
                                'color': 'orange',
                                'width': weight_val * 3,
                                'dash': 'dot'  # Dotted line for attention
                            })
                
                z_pos += 1
                
                # Feed-forward nodes
                for i in range(neuron_count):
                    # Simulate activation
                    a_val = np.random.rand()
                    network_state['node_activations'][(f"ff_{layer_idx+1}", i)] = a_val
                    
                    # Create node
                    network_state['nodes'].append({
                        'x': z_pos, 
                        'y': i-(neuron_count-1)/2, 
                        'z': 0,
                        'name': f"FF{layer_idx+1}N{i+1}",
                        'color': "green",
                        'layer_type': "feedforward",
                        'neuron_idx': i,
                        'layer_idx': f"ff_{layer_idx+1}",
                        'activation': activation,
                        'a_val': float(a_val)
                    })
                    
                    # Connect from attention heads (simplified)
                    ff_idx = len(network_state['nodes']) - 1
                    for h in range(n_heads):
                        attn_idx = input_neurons + neuron_count + h * neuron_count + i
                        weight_val = np.random.rand() * 0.5
                        network_state['edges'].append({
                            'source': attn_idx,
                            'target': ff_idx,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'green',
                            'width': weight_val * 3
                        })
            
            z_pos += 1
        
        # Output layer
        output_neurons = 1
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, neurons[n_layers-1]) * 0.5
        b_out = np.random.randn(output_neurons) * 0.1
        
        network_state['weights'][n_layers+1] = w_out
        network_state['biases'][n_layers+1] = b_out
        
        for i in range(output_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][(n_layers+1, i)] = a_val
            network_state['node_gradients'][(n_layers+1, i)] = 0.5
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0,
                'name': f"Output",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': n_layers+1,
                'activation': "Sigmoid",
                'a_val': float(a_val)
            })
            
            # Connect to last transformer layer
            last_transformer_start = len(network_state['nodes']) - output_neurons - neurons[n_layers-1]
            for j in range(neurons[n_layers-1]):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': last_transformer_start + j,
                    'target': len(network_state['nodes'])-1,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    elif model_type == 'Diffuser':
        # For Diffusion model, we'll visualize the U-Net like structure
        z_pos = 1
        
        # Encoder
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers//2 + n_layers%2], activations[:n_layers//2 + n_layers%2])):
            # Generate random weights and bias
            if layer_idx == 0:
                w = np.random.randn(neuron_count, input_neurons) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            else:
                w = np.random.randn(neuron_count, neurons[layer_idx-1]) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][f"enc_{layer_idx+1}"] = w
            network_state['biases'][f"enc_{layer_idx+1}"] = b
            
            for i in range(neuron_count):
                # Simulate activation
                a_val = np.random.rand()
                network_state['node_activations'][(f"enc_{layer_idx+1}", i)] = a_val
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': 0,
                    'name': f"Enc{layer_idx+1}N{i+1}",
                    'color': "teal",
                    'layer_type': "encoder",
                    'neuron_idx': i,
                    'layer_idx': f"enc_{layer_idx+1}",
                    'activation': activation,
                    'a_val': float(a_val)
                })
                
                # Connect to previous layer
                if layer_idx == 0:
                    for j in range(input_neurons):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': j,
                            'target': input_neurons + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                else:
                    prev_layer_start = input_neurons + sum([neurons[k] for k in range(layer_idx-1)])
                    for j in range(neurons[layer_idx-1]):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': prev_layer_start + j,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
            
            z_pos += 1
        
        # Bottleneck
        bottleneck_size = neurons[n_layers//2] if n_layers > 1 else 64
        bottleneck_idx = z_pos
        
        # Create bottleneck nodes
        for i in range(bottleneck_size):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][("bottleneck", i)] = a_val
            
            # Create node
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(bottleneck_size-1)/2, 
                'z': 0,
                'name': f"Bottleneck{i+1}",
                'color': "purple",
                'layer_type': "bottleneck",
                'neuron_idx': i,
                'layer_idx': "bottleneck",
                'activation': "ReLU",
                'a_val': float(a_val)
            })
            
            # Connect from last encoder layer
            last_enc_start = len(network_state['nodes']) - bottleneck_size - neurons[n_layers//2 - 1 + n_layers%2]
            for j in range(neurons[n_layers//2 - 1 + n_layers%2]):
                weight_val = np.random.randn() * 0.5
                network_state['edges'].append({
                    'source': last_enc_start + j,
                    'target': len(network_state['nodes'])-1,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
        
        z_pos += 1
        
        # Decoder (in reverse order)
        for layer_idx in range(n_layers//2 - 1, -1, -1):
            neuron_count = neurons[layer_idx]
            activation = activations[layer_idx]
            
            # Generate random weights and bias
            if layer_idx == n_layers//2 - 1:
                w = np.random.randn(neuron_count, bottleneck_size) * 0.5
            else:
                w = np.random.randn(neuron_count, neurons[layer_idx+1]) * 0.5
            b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][f"dec_{layer_idx+1}"] = w
            network_state['biases'][f"dec_{layer_idx+1}"] = b
            
            for i in range(neuron_count):
                # Simulate activation
                a_val = np.random.rand()
                network_state['node_activations'][(f"dec_{layer_idx+1}", i)] = a_val
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': 0,
                    'name': f"Dec{layer_idx+1}N{i+1}",
                    'color': "orange",
                    'layer_type': "decoder",
                    'neuron_idx': i,
                    'layer_idx': f"dec_{layer_idx+1}",
                    'activation': activation,
                    'a_val': float(a_val)
                })
                
                # Connect from previous layer
                if layer_idx == n_layers//2 - 1:
                    # Connect from bottleneck
                    bottleneck_start = len(network_state['nodes']) - neuron_count - bottleneck_size
                    for j in range(bottleneck_size):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': bottleneck_start + j,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                else:
                    # Connect from previous decoder layer
                    prev_dec_start = len(network_state['nodes']) - neuron_count - neurons[layer_idx+1]
                    for j in range(neurons[layer_idx+1]):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': prev_dec_start + j,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                
                # Skip connections (from encoder to decoder)
                if layer_idx < n_layers//2:
                    # Find corresponding encoder layer
                    enc_layer_idx = layer_idx
                    enc_start = input_neurons + sum([neurons[k] for k in range(enc_layer_idx)])
                    
                    # Add skip connection
                    if i < neurons[enc_layer_idx]:  # Only if there's a matching neuron
                        weight_val = 0.8  # Strong skip connection
                        network_state['edges'].append({
                            'source': enc_start + i,
                            'target': len(network_state['nodes'])-1,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'blue',  # Highlight skip connections
                            'width': 2,
                            'dash': 'dash'  # Dashed line for skip connections
                        })
            
            z_pos += 1
        
        # Output layer
        output_neurons = 1
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, neurons[0]) * 0.5
        b_out = np.random.randn(output_neurons) * 0.1
        
        network_state['weights']["output"] = w_out
        network_state['biases']["output"] = b_out
        
        for i in range(output_neurons):
            # Simulate activation
            a_val = np.random.rand()
            network_state['node_activations'][("output", i)] = a_val
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0,
                'name': f"Output",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': "output",
                'activation': "Sigmoid",
                'a_val': float(a_val)
            })
            
            # Connect from last decoder layer
            last_dec_start = len(network_state['nodes']) - output_neurons - neurons[0]
            for j in range(neurons[0]):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': last_dec_start + j,
                    'target': len(network_state['nodes'])-1,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    else:  # Default ANN
        # Hidden layers
        z_pos = 1
        for layer_idx, (neuron_count, activation) in enumerate(zip(neurons[:n_layers], activations[:n_layers])):
            # Generate random weights and bias for this layer
            if layer_idx == 0:
                # First hidden layer connects to input
                w = np.random.randn(neuron_count, input_neurons) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            else:
                # Other hidden layers connect to previous hidden layer
                w = np.random.randn(neuron_count, neurons[layer_idx-1]) * 0.5
                b = np.random.randn(neuron_count) * 0.1
            
            network_state['weights'][layer_idx+1] = w
            network_state['biases'][layer_idx+1] = b
            
            # Calculate activations for this layer (forward pass simulation)
            if layer_idx == 0:
                # First hidden layer gets input from input layer
                prev_activations = np.array(input_values)
            else:
                # Other layers get input from previous hidden layer
                prev_activations = np.array([network_state['node_activations'][(layer_idx, j)] for j in range(neurons[layer_idx-1])])
            
            # Calculate pre-activation (z) and activation (a) values
            for i in range(neuron_count):
                z_val = np.dot(w[i], prev_activations) + b[i]
                
                # Apply activation function
                if activation == "ReLU":
                    a_val = max(0, z_val)
                elif activation == "Sigmoid":
                    a_val = 1 / (1 + np.exp(-z_val))
                elif activation == "Tanh":
                    a_val = np.tanh(z_val)
                else:  # Default to ReLU
                    a_val = max(0, z_val)
                
                # Store only the computed values, not function references
                network_state['node_activations'][(layer_idx+1, i)] = a_val
                network_state['node_gradients'][(layer_idx+1, i)] = np.random.rand() * 0.2
                
                # Create node
                network_state['nodes'].append({
                    'x': z_pos, 
                    'y': i-(neuron_count-1)/2, 
                    'z': 0,
                    'name': f"L{layer_idx+1}N{i+1}",
                    'color': "green",
                    'layer_type': "hidden",
                    'neuron_idx': i,
                    'layer_idx': layer_idx+1,
                    'activation': activation,
                    'z_val': float(z_val),
                    'a_val': float(a_val),
                    'weights': w[i].tolist(),
                    'bias': float(b[i])
                })
                
                # Connect to previous layer
                if layer_idx == 0:  # First hidden layer connects to input
                    for j in range(input_neurons):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': j,
                            'target': input_neurons + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
                else:  # Other hidden layers connect to previous hidden layer
                    prev_layer_start = input_neurons + sum(neurons[:layer_idx-1])
                    for j in range(neurons[layer_idx-1]):
                        weight_val = float(w[i, j])
                        network_state['edges'].append({
                            'source': prev_layer_start + j,
                            'target': input_neurons + sum(neurons[:layer_idx]) + i,
                            'value': 1,
                            'weight': weight_val,
                            'color': 'red' if weight_val < 0 else 'green',
                            'width': abs(weight_val) * 3
                        })
            z_pos += 1
        
        # Output layer
        output_neurons = 1
        output_start_idx = input_neurons + sum(neurons[:n_layers])
        
        # Generate random weights and bias for output layer
        w_out = np.random.randn(output_neurons, neurons[n_layers-1]) * 0.5
        b_out = np.random.randn(output_neurons) * 0.1
        network_state['weights'][n_layers+1] = w_out
        network_state['biases'][n_layers+1] = b_out
        
        # Calculate output layer activation
        prev_activations = np.array([network_state['node_activations'][(n_layers, j)] for j in range(neurons[n_layers-1])])
        
        for i in range(output_neurons):
            z_val = np.dot(w_out[i], prev_activations) + b_out[i]
            a_val = 1 / (1 + np.exp(-z_val))  # Sigmoid for output
            
            # Store activation
            network_state['node_activations'][(n_layers+1, i)] = a_val
            network_state['node_gradients'][(n_layers+1, i)] = 0.5  # Higher gradient at output
            
            network_state['nodes'].append({
                'x': z_pos, 
                'y': i-(output_neurons-1)/2, 
                'z': 0,
                'name': f"Output",
                'color': "red",
                'layer_type': "output",
                'neuron_idx': i,
                'layer_idx': n_layers+1,
                'activation': "Sigmoid",
                'z_val': float(z_val),
                'a_val': float(a_val),
                'weights': w_out[i].tolist(),
                'bias': float(b_out[i])
            })
            
            # Connect to last hidden layer
            last_hidden_start = input_neurons + sum(neurons[:n_layers-1])
            for j in range(neurons[n_layers-1]):
                weight_val = float(w_out[i, j])
                network_state['edges'].append({
                    'source': last_hidden_start + j,
                    'target': output_start_idx + i,
                    'value': 1,
                    'weight': weight_val,
                    'color': 'red' if weight_val < 0 else 'green',
                    'width': abs(weight_val) * 3
                })
    
    # Create the appropriate model
    if model_type == 'CNN':
        model = CustomCNN(neurons, activations).to(device)
    elif model_type == 'RNN':
        model = CustomRNN(neurons, activations).to(device)
    elif model_type == 'LSTM':
        model = CustomLSTM(neurons, activations).to(device)
    elif model_type == 'GAN':
        model = CustomGAN(neurons, activations).to(device)
    elif model_type == 'Transformer':
        model = CustomTransformer(neurons, activations).to(device)
    elif model_type == 'Diffuser':
        model = CustomDiffuser(neurons, activations).to(device)
    else:  # Default ANN
        model = CustomNN(neurons, activations).to(device)
    
    network_state['model'] = model
    
    return network_state

# Callback to build network and update all visualizations
@app.callback(
    [Output('network-graph', 'figure'),
     Output('forward-step-slider', 'max'),
     Output('forward-step-slider', 'marks'),
     Output('backward-step-slider', 'max'),
     Output('backward-step-slider', 'marks'),
     Output('layer-selector', 'options')],
    Input('build-button', 'n_clicks'),
    [State('model-type-dropdown', 'value'),
     State('n-layers-slider', 'value'),
     State({'type': 'layer-neurons', 'index': ALL}, 'value'),
     State({'type': 'layer-activation', 'index': ALL}, 'value'),
     State('loss-function-dropdown', 'value'),
     State('problem-type-dropdown', 'value'),
     State('regularization-dropdown', 'value'),
     State('reg-rate-slider', 'value'),
     State('batch-size-dropdown', 'value')],
    prevent_initial_call=True
)
def update_network(n_clicks, model_type, n_layers, neurons, activations, loss_function, 
                  problem_type, regularization, reg_rate, batch_size):
    if n_clicks is None:
        raise PreventUpdate
    
    # Collect model-specific parameters
    model_params = {
        'loss_function': loss_function,
        'problem_type': problem_type,
        'regularization': regularization,
        'reg_rate': 10 ** reg_rate if regularization != 'None' else 0,
        'batch_size': batch_size
    }
    
    # Build network with the selected model type and parameters
    build_network(n_layers, neurons, activations, model_type, model_params)
    
    # Create network visualization
    fig = create_network_figure()
    
    # Update slider ranges for forward and backward propagation
    if model_type == 'CNN':
        num_layers = n_layers + 3  # Input + conv layers + fc + output
    elif model_type in ['RNN', 'LSTM']:
        num_layers = n_layers + 2  # Input + recurrent layers + output
    elif model_type == 'Transformer':
        num_layers = n_layers * 3 + 1  # Input + (embedding + attention + ff) per layer
    elif model_type == 'GAN':
        num_layers = n_layers + 2  # Input + hidden + output (discriminator only)
    elif model_type == 'Diffuser':
        num_layers = n_layers + 3  # Input + encoder + bottleneck + decoder + output
    else:
        num_layers = n_layers + 2  # Input + hidden + output
    
    # Use integers for slider marks
    forward_marks = {i: f'Step {i+1}' for i in range(num_layers)}
    backward_marks = {i: f'Step {i+1}' for i in range(num_layers)}
    
    # Update layer selector options based on model type
    if model_type == 'CNN':
        layer_options = [{'label': 'Input Layer', 'value': 'input'}]
        for i in range(n_layers):
            layer_options.append({'label': f'Conv Layer {i+1}', 'value': f'conv_{i+1}'})
        layer_options.append({'label': 'FC Layer', 'value': 'fc'})
        layer_options.append({'label': 'Output Layer', 'value': 'output'})
    elif model_type in ['RNN', 'LSTM']:
        layer_options = [{'label': 'Input Layer', 'value': 'input'}]
        for i in range(n_layers):
            layer_options.append({'label': f'{model_type} Layer {i+1}', 'value': f'{model_type.lower()}_{i+1}'})
        layer_options.append({'label': 'Output Layer', 'value': 'output'})
    elif model_type == 'Transformer':
        layer_options = [{'label': 'Input Layer', 'value': 'input'}]
        for i in range(n_layers):
            layer_options.append({'label': f'Embedding Layer {i+1}', 'value': f'embedding_{i+1}'})
            layer_options.append({'label': f'Attention Layer {i+1}', 'value': f'attention_{i+1}'})
            layer_options.append({'label': f'Feed-Forward Layer {i+1}', 'value': f'ff_{i+1}'})
        layer_options.append({'label': 'Output Layer', 'value': 'output'})
    elif model_type == 'GAN':
        layer_options = [
            {'label': 'Latent Space', 'value': 'latent'},
            {'label': 'Input Layer (Real Data)', 'value': 'input'}
        ]
        for i in range(n_layers):
            layer_options.append({'label': f'Generator Layer {i+1}', 'value': f'gen_{i+1}'})
        layer_options.append({'label': 'Generator Output', 'value': 'gen_output'})
        for i in range(n_layers):
            layer_options.append({'label': f'Discriminator Layer {i+1}', 'value': f'disc_{i+1}'})
        layer_options.append({'label': 'Discriminator Output', 'value': 'output'})
    elif model_type == 'Diffuser':
        layer_options = [{'label': 'Input Layer', 'value': 'input'}]
        for i in range(n_layers//2 + n_layers%2):
            layer_options.append({'label': f'Encoder Layer {i+1}', 'value': f'enc_{i+1}'})
        layer_options.append({'label': 'Bottleneck', 'value': 'bottleneck'})
        for i in range(n_layers//2):
            layer_options.append({'label': f'Decoder Layer {i+1}', 'value': f'dec_{i+1}'})
        layer_options.append({'label': 'Output Layer', 'value': 'output'})
    else:  # Default ANN
        layer_options = [{'label': 'Input Layer', 'value': 'input'}]
        for i in range(n_layers):
            layer_options.append({'label': f'Hidden Layer {i+1}', 'value': f'hidden_{i+1}'})
        layer_options.append({'label': 'Output Layer', 'value': 'output'})
    
    return fig, num_layers-1, forward_marks, num_layers-1, backward_marks, layer_options

# Function to create network visualization
def create_network_figure():
    if not network_state['nodes']:
        # Return empty figure if network not built yet
        return go.Figure()
    
    nodes = network_state['nodes']
    edges = network_state['edges']
    
    # Create 3D network visualization
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_z = [node['z'] for node in nodes]
    node_color = [node['color'] for node in nodes]
    node_text = [node.get('name', '') for node in nodes]
    
    # Create custom hover text
    node_customdata = []
    for node in nodes:
        layer_idx = node.get('layer_idx', -1)
        neuron_idx = node.get('neuron_idx', -1)
        
        if layer_idx == 0:  # Input layer
            hover_text = f"{node.get('name', '')}<br>Value: {node.get('value', 0):.4f}"
        else:  # Hidden or output layer
            activation = node.get('activation', 'N/A')
            a_val = node.get('a_val', 0)
            hover_text = f"{node.get('name', '')}<br>Activation: {activation}<br>Value: {a_val:.4f}"
        
        node_customdata.append([hover_text])
    
    # Create edges as lines
    edge_x = []
    edge_y = []
    edge_z = []
    edge_color = []
    edge_width = []
    
    for edge in edges:
        source_node = nodes[edge['source']]
        target_node = nodes[edge['target']]
        
        edge_x.extend([source_node['x'], target_node['x'], None])
        edge_y.extend([source_node['y'], target_node['y'], None])
        edge_z.extend([source_node['z'], target_node['z'], None])
        
        # Add color and width for this edge segment
        edge_color.extend([edge.get('color', '#888'), edge.get('color', '#888'), '#888'])
        edge_width.extend([edge.get('width', 1), edge.get('width', 1), 1])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges with varying colors and widths
    for i in range(0, len(edge_x), 3):
        fig.add_trace(go.Scatter3d(
            x=edge_x[i:i+3],
            y=edge_y[i:i+3],
            z=edge_z[i:i+3],
            line=dict(
                color=edge_color[i],
                width=edge_width[i]
            ),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=12,
            color=node_color,
            line=dict(width=1, color='#000')
        ),
        text=node_text,
        hovertemplate='%{customdata[0]}<extra></extra>',
        customdata=node_customdata
    ))
    
    # Layout
    fig.update_layout(
        title='Interactive 3D Neural Network Architecture',
        scene=dict(
            xaxis=dict(title='Layer', showticklabels=False),
            yaxis=dict(title='Neuron', showticklabels=False),
            zaxis=dict(title='', showticklabels=False),
            camera=dict(eye=dict(x=1.75, y=1.75, z=1.75))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    
    return fig

# Function for step-by-step visualization
def create_step_visualization(current_step, direction="forward"):
    if not network_state['nodes']:
        # Return empty figure if network not built yet
        return go.Figure()
    
    # Create a copy of nodes to modify for visualization
    nodes = []
    for node in network_state['nodes']:
        # Create a deep copy to avoid modifying the original
        node_copy = dict(node)
        nodes.append(node_copy)
    
    edges = network_state['edges']
    node_activations = network_state['node_activations']
    node_gradients = network_state['node_gradients']
    
    # Get unique layer indices ahead of time to avoid function calls in list comprehensions
    unique_layer_indices = set()
    for node in nodes:
        layer_idx = node.get('layer_idx', -1)
        unique_layer_indices.add(layer_idx)
    max_layer_idx = max(unique_layer_indices)
    
    # Determine which nodes and edges to highlight based on current step
    if direction == "forward":
        # For forward propagation, we highlight layer by layer from input to output
        active_layer_idx = current_step
        
        # Update node colors based on activation state
        for i, node in enumerate(nodes):
            layer_idx = node.get('layer_idx', -1)
            
            if layer_idx < active_layer_idx:
                # Already processed layers
                nodes[i]['color'] = 'lightblue' if layer_idx == 0 else 'lightgreen'
                nodes[i]['size'] = 10
            elif layer_idx == active_layer_idx:
                # Current active layer
                nodes[i]['color'] = 'yellow'
                nodes[i]['size'] = 15
            else:
                # Not yet processed layers
                nodes[i]['color'] = 'gray'
                nodes[i]['size'] = 8
    else:
        # For backward propagation, we highlight layer by layer from output to input
        active_layer_idx = max_layer_idx - current_step
        
        # Update node colors based on gradient state
        for i, node in enumerate(nodes):
            layer_idx = node.get('layer_idx', -1)
            
            if layer_idx > active_layer_idx:
                # Already processed layers (gradients have flowed through)
                nodes[i]['color'] = 'orange' if layer_idx == max_layer_idx else 'pink'
                nodes[i]['size'] = 10
            elif layer_idx == active_layer_idx:
                # Current active layer
                nodes[i]['color'] = 'red'
                nodes[i]['size'] = 15
            else:
                # Not yet processed layers
                nodes[i]['color'] = 'gray'
                nodes[i]['size'] = 8
    
    # Create 3D network visualization
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_z = [node['z'] for node in nodes]
    node_color = [node['color'] for node in nodes]
    node_size = [node.get('size', 10) for node in nodes]
    node_text = [node.get('name', '') for node in nodes]
    
    # Create custom hover text based on direction
    node_customdata = []
    for node in nodes:
        layer_idx = node.get('layer_idx', -1)
        neuron_idx = node.get('neuron_idx', -1)
        
        if direction == "forward":
            if layer_idx <= active_layer_idx:
                # Show activation for processed nodes
                activation = node_activations.get((layer_idx, neuron_idx), 0)
                hover_text = f"{node.get('name', '')}<br>Value: {activation:.4f}"
                if layer_idx > 0:  # Not input layer
                    hover_text += f"<br>Activation: {node.get('activation', 'N/A')}"
            else:
                hover_text = f"{node.get('name', '')}<br>Not activated yet"
        else:
            if layer_idx >= active_layer_idx:
                # Show gradient for processed nodes
                gradient = node_gradients.get((layer_idx, neuron_idx), 0)
                hover_text = f"{node.get('name', '')}<br>Gradient: {gradient:.4f}"
            else:
                hover_text = f"{node.get('name', '')}<br>No gradient yet"
        
        node_customdata.append([hover_text])
    
    # Create edges as lines
    edge_x = []
    edge_y = []
    edge_z = []
    edge_color = []
    edge_width = []
    
    for edge in edges:
        source_node = nodes[edge['source']]
        target_node = nodes[edge['target']]
        source_layer = source_node.get('layer_idx', -1)
        target_layer = target_node.get('layer_idx', -1)
        
        # Determine edge color and width based on propagation direction and step
        if direction == "forward":
            if source_layer < active_layer_idx and target_layer <= active_layer_idx:
                # Signal has passed through this edge
                color = edge.get('color', 'blue')
                width = edge.get('width', 2)
            elif source_layer == active_layer_idx - 1 and target_layer == active_layer_idx:
                # Signal is currently passing through this edge
                color = 'yellow'
                width = 4
            else:
                # Signal has not reached this edge yet
                color = 'lightgray'
                width = 1
        else:
            if source_layer >= active_layer_idx and target_layer > active_layer_idx:
                # Gradient has passed through this edge
                color = 'red'
                width = 2
            elif source_layer == active_layer_idx and target_layer == active_layer_idx + 1:
                # Gradient is currently passing through this edge
                color = 'orange'
                width = 4
            else:
                # Gradient has not reached this edge yet
                color = 'lightgray'
                width = 1
        
        edge_x.extend([source_node['x'], target_node['x'], None])
        edge_y.extend([source_node['y'], target_node['y'], None])
        edge_z.extend([source_node['z'], target_node['z'], None])
        edge_color.extend([color, color, color])
        edge_width.extend([width, width, width])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for i in range(0, len(edge_x), 3):
        fig.add_trace(go.Scatter3d(
            x=edge_x[i:i+3],
            y=edge_y[i:i+3],
            z=edge_z[i:i+3],
            line=dict(
                color=edge_color[i],
                width=edge_width[i]
            ),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='#000')
        ),
        text=node_text,
        hovertemplate='%{customdata[0]}<extra></extra>',
        customdata=node_customdata
    ))
    
    # Add title based on direction and step
    if direction == "forward":
        title = f"Forward Propagation - Step {current_step+1}: "
        if current_step == 0:
            title += "Input Layer"
        elif current_step == max_layer_idx:
            title += "Output Layer"
        else:
            title += f"Hidden Layer {current_step}"
    else:
        title = f"Backward Propagation - Step {current_step+1}: "
        if active_layer_idx == max_layer_idx:
            title += "Output Layer"
        elif active_layer_idx == 0:
            title += "Input Layer"
        else:
            title += f"Hidden Layer {active_layer_idx}"
    
    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Layer', showticklabels=False),
            yaxis=dict(title='Neuron', showticklabels=False),
            zaxis=dict(title='', showticklabels=False),
            camera=dict(eye=dict(x=1.75, y=1.75, z=1.75))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    
    return fig

# Callback for forward propagation visualization
@app.callback(
    [Output('forward-graph', 'figure'),
     Output('forward-explanation', 'children')],
    [Input('forward-step-slider', 'value')]
)
def update_forward_visualization(step):
    if not network_state['nodes']:
        # Return empty figure if network not built yet
        return go.Figure(), "Build a network first to visualize forward propagation."
    
    # Create visualization for current step
    fig = create_step_visualization(step, "forward")
    
    # Generate explanation text
    num_layers = network_state['n_layers'] + 2  # Input + hidden + output
    
    if step == 0:
        explanation = [
            html.H5("Input Layer"),
            html.P("The network receives input values. These are the initial activations."),
            html.H6("Input values:"),
            html.Ul([
                html.Li(f"Input {i+1}: {network_state['node_activations'].get((0, i), 0):.4f}")
                for i in range(2)  # Assuming 2 input neurons
            ])
        ]
    elif step == num_layers - 1:
        explanation = [
            html.H5("Output Layer"),
            html.P("The final output is computed using the activations from the last hidden layer."),
            html.H6("Output calculation:")
        ]
        
        # Find output node
        output_node = None
        for node in network_state['nodes']:
            if node.get('layer_idx') == num_layers - 1:
                output_node = node
                break
        
        if output_node:
            weights_val = output_node.get('weights', [])
            bias_val = output_node.get('bias', 0)
            prev_layer_idx = num_layers - 2
            prev_values = [network_state['node_activations'].get((prev_layer_idx, j), 0) 
                          for j in range(network_state['layers'][-1])]
            
            calculation = "z = "
            for i, (w, x) in enumerate(zip(weights_val, prev_values)):
                calculation += f"{w:.4f} × {x:.4f}"
                if i < len(weights_val) - 1:
                    calculation += " + "
            
            calculation += f" + {bias_val:.4f} (bias)"
            
            z_val = output_node.get('z_val', 0)
            a_val = output_node.get('a_val', 0)
            
            explanation.extend([
                html.P(calculation),
                html.P(f"z = {z_val:.4f}"),
                html.P(f"a = sigmoid(z) = 1/(1+e^(-{z_val:.4f})) = {a_val:.4f}"),
                html.P(f"Final output: {a_val:.4f}", className="font-weight-bold")
            ])
    else:
        explanation = [
            html.H5(f"Hidden Layer {step}"),
            html.P("Each neuron computes a weighted sum of inputs from the previous layer, adds a bias, and applies an activation function."),
            html.H6(f"Layer {step} activations:"),
            html.Ul([
                html.Li(f"Neuron {i+1}: {network_state['node_activations'].get((step, i), 0):.4f}")
                for i in range(network_state['layers'][step-1])
            ])
        ]
    
    return fig, explanation

# Callback for backward propagation visualization
@app.callback(
    [Output('backward-graph', 'figure'),
     Output('backward-explanation', 'children')],
    [Input('backward-step-slider', 'value')]
)
def update_backward_visualization(step):
    if not network_state['nodes']:
        # Return empty figure if network not built yet
        return go.Figure(), "Build a network first to visualize backpropagation."
    
    # Create visualization for current step
    fig = create_step_visualization(step, "backward")
    
    # Generate explanation text
    num_layers = network_state['n_layers'] + 2  # Input + hidden + output
    reverse_idx = num_layers - step - 1
    
    if step == 0:
        explanation = [
            html.H5("Output Layer Gradients"),
            html.P("Backpropagation starts at the output layer. The gradient is computed based on the difference between predicted and actual values."),
            html.H6("Output gradient:"),
            html.P(f"Gradient: {network_state['node_gradients'].get((num_layers-1, 0), 0):.4f}")
        ]
    elif step == num_layers - 1:
        explanation = [
            html.H5("Input Layer Gradients"),
            html.P("The gradients have propagated all the way back to the input layer."),
            html.H6("Input gradients:"),
            html.Ul([
                html.Li(f"Input {i+1}: {network_state['node_gradients'].get((0, i), 0):.4f}")
                for i in range(2)  # Assuming 2 input neurons
            ])
        ]
    else:
        explanation = [
            html.H5(f"Hidden Layer {reverse_idx} Gradients"),
            html.P("Gradients flow backward through the network. Each neuron receives gradients from the next layer, which are used to update weights."),
            html.H6(f"Layer {reverse_idx} gradients:"),
            html.Ul([
                html.Li(f"Neuron {i+1}: {network_state['node_gradients'].get((reverse_idx, i), 0):.4f}")
                for i in range(network_state['layers'][reverse_idx-1])
            ])
        ]
    
    # Add weight update explanation
    if step > 0:
        explanation.extend([
            html.H5("Weight Updates"),
            html.P("Weights are updated using the formula: weight -= learning_rate * gradient * input"),
            html.P("Example for one connection:"),
            html.P("Δw = -0.01 * gradient * activation"),
            html.P("The negative gradient direction minimizes the loss function.")
        ])
    
    return fig, explanation

# Callback to update neuron selector based on layer selection
@app.callback(
    [Output('neuron-selector', 'options'),
     Output('neuron-selector', 'value')],
    [Input('layer-selector', 'value')]
)
def update_neuron_selector(selected_layer):
    if not selected_layer or not network_state['nodes']:
        return [], None
    
    if selected_layer == 'input':
        neuron_count = 2  # Assuming 2 input neurons
        options = [{'label': f'Input {i+1}', 'value': i} for i in range(neuron_count)]
    elif selected_layer == 'output':
        neuron_count = 1  # Assuming 1 output neuron
        options = [{'label': 'Output', 'value': 0}]
    else:
        # Extract hidden layer number
        layer_num = int(selected_layer.split('_')[1]) - 1
        neuron_count = network_state['layers'][layer_num]
        options = [{'label': f'Neuron {i+1}', 'value': i} for i in range(neuron_count)]
    
    return options, 0  # Default to first neuron

# Callback to display node details
@app.callback(
    Output('node-details', 'children'),
    [Input('layer-selector', 'value'),
     Input('neuron-selector', 'value')]
)
def update_node_details(layer_type, neuron_idx):
    if not layer_type or neuron_idx is None or not network_state['nodes']:
        return "Select a layer and neuron to see details."
    
    # Determine layer index
    if layer_type == 'input':
        layer_idx = 0
    elif layer_type == 'output':
        layer_idx = network_state['n_layers'] + 1
    else:
        layer_idx = int(layer_type.split('_')[1])
    
    # Find the selected node
    selected_node = None
    for node in network_state['nodes']:
        if node.get('layer_idx') == layer_idx and node.get('neuron_idx') == neuron_idx:
            selected_node = node
            break
    
    if not selected_node:
        return "Node not found."
    
    # Generate details based on node type
    if layer_idx == 0:  # Input layer
        details = [
            html.H5(f"Input {neuron_idx+1}"),
            html.P(f"Value: {selected_node.get('value', 'N/A')}")
        ]
    else:  # Hidden or output layer
        details = [
            html.H5(f"{selected_node.get('layer_type', 'Unknown').capitalize()} Layer Neuron {neuron_idx+1}"),
            html.P(f"Activation Function: {selected_node.get('activation', 'N/A')}"),
            html.H6("Forward Pass Calculation:"),
        ]
        
        # Get weights and bias
        weights_val = selected_node.get('weights', [])
        bias_val = selected_node.get('bias', 0)
        
        # Get inputs from previous layer
        if layer_idx == 1:  # First hidden layer
            prev_values = [network_state['node_activations'].get((0, j), 0) for j in range(2)]
            prev_layer_name = "Input"  # Define this variable for first hidden layer
        else:  # Other hidden layers or output
            prev_layer_idx = layer_idx - 1
            prev_neuron_count = 2 if prev_layer_idx == 0 else network_state['layers'][prev_layer_idx-1]
            prev_values = [network_state['node_activations'].get((prev_layer_idx, j), 0) for j in range(prev_neuron_count)]
            prev_layer_name = "Hidden" if prev_layer_idx < network_state['n_layers'] else "Output"
        
        # Display the calculation
        calculation = "z = "
        for i, (w, x) in enumerate(zip(weights_val, prev_values)):
            calculation += f"{w:.4f} × {x:.4f}"
            if i < len(weights_val) - 1:
                calculation += " + "
        
        calculation += f" + {bias_val:.4f} (bias)"
        
        z_val = selected_node.get('z_val', 0)
        a_val = selected_node.get('a_val', 0)
        activation_func = selected_node.get('activation', 'ReLU')
        
        details.extend([
            html.P(calculation),
            html.P(f"z = {z_val:.4f}"),
        ])
        
        # Activation calculation
        if activation_func == "ReLU":
            details.append(html.P(f"a = max(0, z) = max(0, {z_val:.4f}) = {a_val:.4f}"))
        elif activation_func == "Sigmoid":
            details.append(html.P(f"a = sigmoid(z) = 1/(1+e^(-{z_val:.4f})) = {a_val:.4f}"))
        elif activation_func == "Tanh":
            details.append(html.P(f"a = tanh(z) = tanh({z_val:.4f}) = {a_val:.4f}"))
        else:
            details.append(html.P(f"a = {activation_func}({z_val:.4f}) = {a_val:.4f}"))
        
        # Show backward pass (gradient) information
        gradient_val = network_state['node_gradients'].get((layer_idx, neuron_idx), 0)
        details.extend([
            html.H6("Backward Pass (Gradient):"),
            html.P(f"Gradient at this node: {gradient_val:.4f}"),
            html.H6("Weight Updates:"),
            html.P(f"Using learning rate: 0.01")
        ])
        
        # Show weight updates
        for i, w in enumerate(weights_val):
            update = 0.01 * gradient_val * prev_values[i]
            details.extend([
                html.P(f"Δw{i+1} = -0.01 × {gradient_val:.4f} × {prev_values[i]:.4f} = -{update:.6f}"),
                html.P(f"w{i+1}_new = {w:.4f} - {update:.6f} = {w - update:.4f}")
            ])
    
    return details

# Complete the decision boundary visualization function
@app.callback(
    Output('decision-boundary', 'figure'),
    [Input('tabs', 'active_tab'),
     Input('build-button', 'n_clicks')],
    [State('model-type-dropdown', 'value'),
     State('dataset-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('problem-type-dropdown', 'value')]
)
def update_decision_boundary(active_tab, n_clicks, model_type, dataset_type, noise_level, problem_type):
    if active_tab != 'tab-decision' or not network_state['nodes']:
        raise PreventUpdate
    
    try:
        # Create appropriate model based on model type
        if model_type == 'CNN':
            model = CustomCNN(network_state['layers'], network_state['activations'])
        elif model_type == 'RNN':
            model = CustomRNN(network_state['layers'], network_state['activations'])
        elif model_type == 'LSTM':
            model = CustomLSTM(network_state['layers'], network_state['activations'])
        elif model_type == 'GAN':
            model = CustomGAN(network_state['layers'], network_state['activations'])
        elif model_type == 'Transformer':
            model = CustomTransformer(network_state['layers'], network_state['activations'])
        elif model_type == 'Diffuser':
            model = CustomDiffuser(network_state['layers'], network_state['activations'])
        else:  # Default ANN
            model = CustomNN(network_state['layers'], network_state['activations'])
        
        # Generate dataset based on selection
        n_samples = 200
        noise_factor = noise_level / 100.0  # Convert percentage to factor
        
        if dataset_type == 'Circle':
            X, y = make_circles(n_samples=n_samples, noise=noise_factor, factor=0.5, random_state=42)
        elif dataset_type == 'Gaussian':
            X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise_factor*2, random_state=42)
        elif dataset_type == 'XOR':
            X, y = generate_xor_data(n_samples=n_samples)
            # Add noise
            if noise_factor > 0:
                X += np.random.normal(0, noise_factor, X.shape)
        elif dataset_type == 'Spiral':
            X, y = generate_spiral_data(n_samples=n_samples)
            # Add noise
            if noise_factor > 0:
                X += np.random.normal(0, noise_factor, X.shape)
        else:
            # Default to circles
            X, y = make_circles(n_samples=n_samples, noise=noise_factor, factor=0.5, random_state=42)
        
        # Create grid for decision boundary
        resolution = 100
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # Prepare input for model
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        
        # Get predictions
        with torch.no_grad():
            if model_type == 'Diffuser':
                z = model(grid_tensor, torch.zeros(grid_tensor.size(0), 1)).detach().cpu().numpy().reshape(xx.shape)
            else:
                z = model(grid_tensor).detach().cpu().numpy().reshape(xx.shape)
        
        # Create figure with a 3D surface plot
        fig = go.Figure()
        
        # Add decision boundary surface
        fig.add_trace(
            go.Surface(
                x=np.linspace(x_min, x_max, resolution),
                y=np.linspace(y_min, y_max, resolution),
                z=z,
                colorscale='RdBu',
                opacity=0.8,
                showscale=True,
                contours={
                    "z": {"show": True, "start": 0.5, "end": 0.5, "size": 0.5, "color":"white"}
                }
            )
        )
        
        # Add data points
        if problem_type == 'Classification':
            # Add scatter points for each class
            for class_idx in np.unique(y):
                mask = y == class_idx
                fig.add_trace(
                    go.Scatter3d(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        z=np.ones(np.sum(mask)) * (0 if class_idx == 0 else 1),
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='blue' if class_idx == 0 else 'red',
                            opacity=0.8
                        ),
                        name=f'Class {class_idx}'
                    )
                )
        else:  # Regression
            # For regression, color points by their value
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=y,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=y,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name='Data Points'
                )
            )
        
        # Add title based on model type and dataset
        title = f'3D Decision Boundary - {model_type} on {dataset_type} Dataset'
        
        # Improve layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Prediction',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating decision boundary: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

# Callback to create loss landscape visualization
@app.callback(
    Output('loss-landscape', 'figure'),
    [Input('tabs', 'active_tab')]
)
def update_loss_landscape(active_tab):
    if active_tab != 'tab-loss':
        raise PreventUpdate
    
    try:
        # Create a simpler grid for the loss landscape
        resolution = 30
        x_vals = np.linspace(-2, 2, resolution)
        y_vals = np.linspace(-2, 2, resolution)
        
        # Create meshgrid
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Compute loss function (simulated)
        Z = np.sin(X) ** 2 + np.cos(Y) ** 2
        
        # Create figure
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(
            go.Surface(
                x=x_vals,
                y=y_vals,
                z=Z,
                colorscale='Viridis',
                opacity=0.9
            )
        )
        
        # Add a point to represent current position in loss landscape
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[Z[resolution//2, resolution//2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                ),
                name='Current Position'
            )
        )
        
        # Add a path showing optimization trajectory
        # Simulated gradient descent path
        path_x = np.linspace(1.5, 0, 10)
        path_y = np.linspace(1.5, 0, 10)
        path_z = [np.sin(x) ** 2 + np.cos(y) ** 2 for x, y in zip(path_x, path_y)]
        
        fig.add_trace(
            go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=4, color='red'),
                name='Optimization Path'
            )
        )
        
        # Improve layout
        fig.update_layout(
            title='Interactive 3D Loss Landscape',
            scene=dict(
                xaxis_title='Weight 1',
                yaxis_title='Weight 2',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.75, y=1.75, z=1.75))
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating loss landscape: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

# Update the training curve visualization to include regularization effects
@app.callback(
    Output('training-curve', 'figure'),
    [Input('tabs', 'active_tab'),
     Input('train-button', 'n_clicks')],
    [State('learning-rate-slider', 'value'),
     State('epochs-slider', 'value'),
     State('regularization-dropdown', 'value'),
     State('reg-rate-slider', 'value'),
     State('batch-size-dropdown', 'value')]
)
def update_training_curve(active_tab, n_clicks, learning_rate_exp, epochs, regularization, reg_rate_exp, batch_size):
    if active_tab != 'tab-training' or n_clicks is None:
        raise PreventUpdate
    
    # Convert learning rate from exponent
    learning_rate = 10 ** learning_rate_exp
    
    # Convert regularization rate from exponent if applicable
    reg_rate = 0
    if regularization != 'None':
        reg_rate = 10 ** reg_rate_exp
    
    # Simulate training with different parameters
    np.random.seed(42)  # For reproducibility
    
    # Base loss depends on learning rate
    base_loss = 1.0 / (learning_rate * 10 + 1)
    
    # Batch size affects convergence speed and noise
    batch_factor = 1.0 / np.sqrt(batch_size)
    
    # Regularization affects final loss
    reg_factor = 1.0 + (reg_rate * 5 if regularization != 'None' else 0)
    
    # Generate loss values with these factors
    loss_values = []
    for epoch in range(epochs):
        # Decay factor based on epoch
        decay = 1.0 / (epoch + 1)
        
        # Noise based on batch size (smaller batches = more noise)
        noise = np.random.uniform(-0.05, 0.05) * batch_factor
        
        # Regularization adds a small constant to loss
        reg_loss = reg_factor * 0.01 if epoch > 0 else 0
        
        # Combine factors for final loss
        loss = base_loss * decay + noise + reg_loss
        loss_values.append(loss)
    
    # Create 3D visualization of training progress
    fig = go.Figure()
    
    # Add the main loss curve as a 3D line
    fig.add_trace(go.Scatter3d(
        x=list(range(1, epochs+1)),
        y=[0] * epochs,
        z=loss_values,
        mode='lines+markers',
        line=dict(color='blue', width=5),
        marker=dict(size=8, color=loss_values, colorscale='Viridis'),
        name='Loss',
        hovertemplate='Epoch: %{x}<br>Loss: %{z:.4f}<extra></extra>'
    ))
    
    # Add vertical lines to ground plane for better 3D effect
    for i, loss in enumerate(loss_values):
        fig.add_trace(go.Scatter3d(
            x=[i+1, i+1],
            y=[0, 0],
            z=[0, loss],
            mode='lines',
            line=dict(color='rgba(0,0,100,0.3)', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add a semi-transparent surface to show potential loss landscape
    x = np.linspace(1, epochs, 20)
    y = np.linspace(-0.5, 0.5, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create a surface that mimics the loss curve but with variations in y-direction
    Z = np.array([[base_loss / (x_val) + 0.05 * np.sin(y_val * np.pi) + 
                   (reg_factor * 0.01 if x_val > 1 else 0) for x_val in x] for y_val in y])
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',
        opacity=0.3,
        showscale=False,
        hoverinfo='none'
    ))
    
    # Add training parameters to title
    title = '3D Training Loss Visualization<br>'
    title += f'Learning Rate: {learning_rate:.4f}, Batch Size: {batch_size}, '
    title += f'Regularization: {regularization}'
    if regularization != 'None':
        title += f' ({reg_rate:.4f})'
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='',
            zaxis_title='Loss',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            yaxis=dict(showticklabels=False)
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    return fig

# Add callback to update loss function options based on model type
@app.callback(
    Output('loss-function-dropdown', 'options'),
    Output('loss-function-dropdown', 'value'),
    Input('model-type-dropdown', 'value')
)
def update_loss_function_options(model_type):
    losses = model_specific_params[model_type]['losses']
    default_loss = model_specific_params[model_type]['default_loss']
    
    options = [{'label': loss, 'value': loss} for loss in losses]
    return options, default_loss

# Add callback to show model-specific parameters
@app.callback(
    Output('model-specific-params', 'children'),
    Input('model-type-dropdown', 'value')
)
def update_model_specific_params(model_type):
    params_ui = []
    
    if model_type == 'CNN':
        params_ui.extend([
            html.Label("Kernel Size:"),
            dcc.Dropdown(
                id='cnn-kernel-size',
                options=[{'label': f"{k}x{k}", 'value': k} for k in model_specific_params['CNN']['kernel_sizes']],
                value=model_specific_params['CNN']['default_kernel'],
                clearable=False,
                className="mb-2"
            ),
            html.Label("Pooling Size:"),
            dcc.Dropdown(
                id='cnn-pool-size',
                options=[{'label': f"{p}x{p}", 'value': p} for p in model_specific_params['CNN']['pool_sizes']],
                value=model_specific_params['CNN']['default_pool'],
                clearable=False,
                className="mb-3"
            )
        ])
    
    elif model_type in ['RNN', 'LSTM']:
        params_ui.extend([
            html.Label("Sequence Length:"),
            dcc.Dropdown(
                id='rnn-seq-length',
                options=[{'label': str(s), 'value': s} for s in model_specific_params[model_type]['sequence_lengths']],
                value=model_specific_params[model_type]['default_seq_len'],
                clearable=False,
                className="mb-3"
            )
        ])
    
    elif model_type == 'GAN':
        params_ui.extend([
            html.Label("Latent Dimension:"),
            dcc.Dropdown(
                id='gan-latent-dim',
                options=[{'label': str(d), 'value': d} for d in model_specific_params['GAN']['latent_dims']],
                value=model_specific_params['GAN']['default_latent_dim'],
                clearable=False,
                className="mb-3"
            )
        ])
    
    elif model_type == 'Transformer':
        params_ui.extend([
            html.Label("Number of Attention Heads:"),
            dcc.Dropdown(
                id='transformer-num-heads',
                options=[{'label': str(h), 'value': h} for h in model_specific_params['Transformer']['num_heads']],
                value=model_specific_params['Transformer']['default_heads'],
                clearable=False,
                className="mb-2"
            ),
            html.Label("Dropout Rate:"),
            dcc.Dropdown(
                id='transformer-dropout',
                options=[{'label': str(d), 'value': d} for d in model_specific_params['Transformer']['dropout_rates']],
                value=model_specific_params['Transformer']['default_dropout'],
                clearable=False,
                className="mb-3"
            )
        ])
    
    elif model_type == 'Diffuser':
        params_ui.extend([
            html.Label("Diffusion Time Steps:"),
            dcc.Dropdown(
                id='diffuser-time-steps',
                options=[{'label': str(t), 'value': t} for t in model_specific_params['Diffuser']['time_steps']],
                value=model_specific_params['Diffuser']['default_time_steps'],
                clearable=False,
                className="mb-2"
            ),
            html.Label("Noise Schedule:"),
            dcc.Dropdown(
                id='diffuser-noise-schedule',
                options=[{'label': s, 'value': s} for s in model_specific_params['Diffuser']['noise_schedules']],
                value=model_specific_params['Diffuser']['default_schedule'],
                clearable=False,
                className="mb-3"
            )
        ])
    
    return params_ui

# Add callback to update regularization dropdown based on model type
@app.callback(
    Output('regularization-dropdown', 'options'),
    Output('regularization-dropdown', 'value'),
    Input('model-type-dropdown', 'value')
)
def update_regularization_options(model_type):
    regularizations = model_specific_params[model_type]['regularizations']
    default_reg = model_specific_params[model_type]['default_regularization']
    
    options = [{'label': reg, 'value': reg} for reg in regularizations]
    return options, default_reg

# Add callback to update the dataset visualization when parameters change
@app.callback(
    Output('dataset-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('noise-slider', 'value'),
     Input('problem-type-dropdown', 'value')]
)
def update_dataset_visualization(dataset_type, noise_level, problem_type):
    # Generate dataset based on selection
    n_samples = 200
    noise_factor = noise_level / 100.0  # Convert percentage to factor
    
    if dataset_type == 'Circle':
        X, y = make_circles(n_samples=n_samples, noise=noise_factor, factor=0.5, random_state=42)
    elif dataset_type == 'Gaussian':
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise_factor*2, random_state=42)
    elif dataset_type == 'XOR':
        X, y = generate_xor_data(n_samples=n_samples)
        # Add noise
        if noise_factor > 0:
            X += np.random.normal(0, noise_factor, X.shape)
    elif dataset_type == 'Spiral':
        X, y = generate_spiral_data(n_samples=n_samples)
        # Add noise
        if noise_factor > 0:
            X += np.random.normal(0, noise_factor, X.shape)
    else:
        # Default to circles
        X, y = make_circles(n_samples=n_samples, noise=noise_factor, factor=0.5, random_state=42)
    
    # Create figure
    if problem_type == 'Classification':
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str),
                         color_discrete_sequence=['#636EFA', '#EF553B'],
                         labels={'color': 'Class', 'x': 'Feature 1', 'y': 'Feature 2'},
                         title=f'{dataset_type} Dataset with {noise_level}% Noise')
    else:  # Regression
        # For regression, we'll use the distance from origin as target
        y_reg = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y_reg,
                         color_continuous_scale='Viridis',
                         labels={'color': 'Value', 'x': 'Feature 1', 'y': 'Feature 2'},
                         title=f'{dataset_type} Dataset with {noise_level}% Noise (Regression)')
    
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# Callback to update training progress UI elements
@app.callback(
    [Output("training-message", "children"),
     Output("training-progress", "value")],
    [Input("training-interval", "n_intervals")],
    [State("is-training", "data-json")]
)
def update_training_progress(n_intervals, is_training_json):
    if not is_training_json or is_training_json == "false":
        return "Not training", 0
    
    training_state_local = json.loads(is_training_json)
    progress = (training_state_local["current_epoch"] / training_state_local["total_epochs"]) * 100
    
    return f"Training: Epoch {training_state_local['current_epoch']}/{training_state_local['total_epochs']}", progress

# Callback for training control buttons
@app.callback(
    [Output("training-interval", "disabled"),
     Output("is-training", "data-json")],
    [Input("start-training-button", "n_clicks"),
     Input("stop-training-button", "n_clicks")],
    [State("realtime-epochs-slider", "value"),
     State("realtime-learning-rate-slider", "value"),
     State("is-training", "data-json")]
)
def control_training(start_clicks, stop_clicks, epochs, learning_rate_exp, is_training_json):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, "false"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "start-training-button" and start_clicks:
        # Start training
        training_state["is_training"] = True
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["loss_history"] = []
        training_state["accuracy_history"] = []
        
        # Convert learning rate from exponent
        learning_rate = 10 ** learning_rate_exp
        training_state["learning_rate"] = learning_rate
        
        return False, json.dumps(training_state)
    
    elif button_id == "stop-training-button" and stop_clicks:
        # Stop training
        training_state["is_training"] = False
        return True, "false"
    
    # Default - no change
    return dash.no_update, dash.no_update

# Modify the training_step callback to include network graph updates
@app.callback(
    [Output("training-metrics", "figure"),
     Output("realtime-decision-boundary", "figure"),
     Output("network-graph", "figure", allow_duplicate=True),  # This is the key output
     Output("is-training", "data-json", allow_duplicate=True)],
    [Input("training-interval", "n_intervals")],
    [State("is-training", "data-json"),
     State("model-type-dropdown", "value"),
     State("dataset-dropdown", "value"),
     State("problem-type-dropdown", "value"),
     State("noise-slider", "value"),
     State("n-layers-slider", "value"),
     State({"type": "layer-neurons", "index": dash.dependencies.ALL}, "value"),
     State({"type": "layer-activation", "index": dash.dependencies.ALL}, "value")],
    prevent_initial_call=True
)
def training_step(n_intervals, is_training_json, model_type, dataset_type, problem_type, 
                 noise_level, n_layers, neurons, activations):
    if not is_training_json or is_training_json == "false":
        raise dash.exceptions.PreventUpdate
    
    training_state_local = json.loads(is_training_json)
    
    if not training_state_local["is_training"]:
        raise dash.exceptions.PreventUpdate
    
    # Increment epoch
    training_state_local["current_epoch"] += 1
    current_epoch = training_state_local["current_epoch"]
    total_epochs = training_state_local["total_epochs"]
    
    # Create metrics figure - THIS WAS MISSING
    loss = 1.0 / (current_epoch + 1) + np.random.normal(0, 0.05)
    accuracy = min(0.99, 0.5 + 0.5 * (current_epoch / total_epochs) + np.random.normal(0, 0.02))
    
    # Update history
    loss_history = training_state_local.get("loss_history", [])
    accuracy_history = training_state_local.get("accuracy_history", [])
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    training_state_local["loss_history"] = loss_history
    training_state_local["accuracy_history"] = accuracy_history
    
    # Create metrics figure
    metrics_fig = go.Figure()
    epochs = list(range(1, len(loss_history) + 1))
    
    metrics_fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_history,
        mode="lines+markers",
        name="Loss",
        line=dict(color="red", width=2)
    ))
    
    metrics_fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracy_history,
        mode="lines+markers",
        name="Accuracy",
        line=dict(color="green", width=2)
    ))
    
    metrics_fig.update_layout(
        title=f"Training Metrics - Epoch {current_epoch}/{total_epochs}",
        xaxis_title="Epoch",
        yaxis_title="Value",
        plot_bgcolor="white"
    )
    
    # Create decision boundary figure - THIS WAS MISSING
    decision_fig = go.Figure()
    
    # Create a dynamic visualization that clearly changes
    theta = np.linspace(0, 2*np.pi, 100)
    progress = current_epoch / total_epochs
    
    # Outer circle (fixed reference)
    decision_fig.add_trace(go.Scatter(
        x=np.cos(theta), 
        y=np.sin(theta),
        mode="lines",
        line=dict(color="lightgray", width=1),
        name="Target"
    ))
    
    # Progress circle (grows with training)
    decision_fig.add_trace(go.Scatter(
        x=progress * np.cos(theta), 
        y=progress * np.sin(theta),
        mode="lines",
        line=dict(color="blue", width=2),
        fill="toself",
        name="Progress"
    ))
    
    decision_fig.update_layout(
        title=f"Decision Boundary - Epoch {current_epoch}/{total_epochs}",
        xaxis=dict(range=[-1.2, 1.2], title="Feature 1"),
        yaxis=dict(range=[-1.2, 1.2], title="Feature 2"),
        plot_bgcolor="white"
    )
    
    # Now, update the network visualization to show animation
    # This is the key part for animating the model architecture
    
    # 1. Update node activations based on training progress
    for node in network_state['nodes']:
        # Generate activation values that change with training
        if 'neuron_idx' in node:
            # Create a wave pattern that changes with epoch
            wave_factor = np.sin(current_epoch / 5 + node.get('neuron_idx', 0) / 2)
            activation_value = 0.5 + 0.5 * wave_factor  # Range from 0 to 1
            
            # Store the activation value
            node['a_val'] = activation_value
            
            # Update node color based on activation
            # Bright red for high activation, blue for low
            r = int(255 * activation_value)
            b = int(255 * (1 - activation_value))
            node['color'] = f'rgb({r}, 100, {b})'
            
            # Update node size based on activation
            base_size = node.get('base_size', 15)  # Store original size if not already stored
            if 'base_size' not in node:
                node['base_size'] = base_size
            
            # Make node size pulse with activation
            size_factor = 1 + 0.5 * activation_value
            node['size'] = base_size * size_factor
    
    # 2. Update edge weights based on training progress
    for edge in network_state.get('edges', []):
        # Generate weight values that converge with training
        source_idx = edge.get('source_idx', 0)
        target_idx = edge.get('target_idx', 0)
        
        # Initial random weight if not set
        if 'initial_weight' not in edge:
            edge['initial_weight'] = np.random.normal(0, 1)
        
        # Target weight (where training would converge to)
        if 'target_weight' not in edge:
            edge['target_weight'] = np.random.normal(0, 0.5)
        
        # Calculate current weight based on training progress
        progress = min(1.0, current_epoch / (total_epochs * 0.7))  # Converge by 70% of training
        current_weight = edge['initial_weight'] + progress * (edge['target_weight'] - edge['initial_weight'])
        
        # Add some oscillation for visual effect
        oscillation = 0.1 * np.sin(current_epoch / 3 + source_idx + target_idx)
        current_weight += oscillation
        
        # Store the weight
        edge['weight'] = current_weight
        
        # Update edge width based on weight magnitude
        width = 1 + 3 * abs(current_weight)
        edge['width'] = width
        
        # Update edge color based on weight sign
        if current_weight > 0:
            # Positive weights: blue, stronger = darker
            intensity = int(200 - 150 * min(1, abs(current_weight)))
            edge['color'] = f'rgba(0, 0, {intensity}, 0.8)'
        else:
            # Negative weights: red, stronger = darker
            intensity = int(200 - 150 * min(1, abs(current_weight)))
            edge['color'] = f'rgba({intensity}, 0, 0, 0.8)'
    
    # 3. Create the updated network figure
    network_fig = create_network_figure()
    
    # 4. Update the layout to show training progress
    network_fig.update_layout(
        title=f"Network Architecture - Training Epoch {current_epoch}/{total_epochs}",
        scene=dict(
            annotations=[
                dict(
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    z=0,
                    text=f"Training Progress: {current_epoch}/{total_epochs}",
                    xanchor="center",
                    xshift=0,
                    opacity=0.7
                )
            ]
        )
    )
    
    # Check if training is complete
    if current_epoch >= total_epochs:
        training_state_local["is_training"] = False
    
    return metrics_fig, decision_fig, network_fig, json.dumps(training_state_local)

# Add a callback to link the button to the training tab
@app.callback(
    Output("tabs", "active_tab"),
    [Input("view-training-button", "n_clicks")],
    prevent_initial_call=True
)
def switch_to_training_tab(n_clicks):
    if n_clicks:
        return "tab-realtime-training"
    raise PreventUpdate

# Add to your layout in the real-time training tab
html.Div([
    html.Label("Animation Speed:"),
    dcc.Slider(
        id="animation-speed-slider",
        min=100,
        max=2000,
        step=100,
        marks={i: f"{i}ms" for i in range(100, 2001, 300)},
        value=500
    )
], className="mt-3"),

# Add a callback to control animation speed
@app.callback(
    Output("training-interval", "interval"),
    [Input("animation-speed-slider", "value")]
)
def update_animation_speed(value):
    return value

# Run the app with suppress_callback_exceptions=True
if __name__ == '__main__':
    app.config.suppress_callback_exceptions = True
    app.run_server(debug=True, host='0.0.0.0')
