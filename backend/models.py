import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomNN(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.activations = activations
        input_size = 2
        for neurons in layers:
            self.fc_layers.append(nn.Linear(input_size, neurons))
            input_size = neurons
        self.fc_layers.append(nn.Linear(input_size, 1))

    def forward(self, x):
        for layer, act in zip(self.fc_layers[:-1], self.activations):
            x = layer(x)
            if act == "ReLU":
                x = torch.relu(x)
            elif act == "Sigmoid":
                x = torch.sigmoid(x)
            elif act == "Tanh":
                x = torch.tanh(x)
            elif act == "LeakyReLU":
                x = F.leaky_relu(x, 0.2)
            elif act == "ELU":
                x = F.elu(x)
            else:
                x = torch.relu(x)
        return torch.sigmoid(self.fc_layers[-1](x))


class CustomCNN(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for neurons in layers:
            self.conv_layers.append(nn.Conv2d(in_channels, neurons, kernel_size=3, padding=1))
            in_channels = neurons
        self.fc1 = nn.Linear(layers[-1] * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 8, 8)
        for i, (conv, act) in enumerate(zip(self.conv_layers, self.activations)):
            x = conv(x)
            if act == "ReLU":
                x = F.relu(x)
            elif act == "Sigmoid":
                x = torch.sigmoid(x)
            elif act == "Tanh":
                x = torch.tanh(x)
            else:
                x = F.relu(x)
            if i < len(self.conv_layers) - 1:
                x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class CustomRNN(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        hidden_size = layers[0]
        self.rnn = nn.RNN(input_size=2, hidden_size=hidden_size, num_layers=len(layers), batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class CustomLSTM(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        hidden_size = layers[0]
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=len(layers), batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class CustomGAN(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        gen_layers = [nn.Linear(10, layers[0]), nn.ReLU()]
        if len(layers) > 1:
            gen_layers += [nn.Linear(layers[0], layers[1]), nn.ReLU()]
        gen_layers += [nn.Linear(layers[-1], 2), nn.Tanh()]
        self.generator = nn.Sequential(*gen_layers)

        disc_layers = [nn.Linear(2, layers[0]), nn.LeakyReLU(0.2)]
        if len(layers) > 1:
            disc_layers += [nn.Linear(layers[0], layers[1]), nn.LeakyReLU(0.2)]
        disc_layers += [nn.Linear(layers[-1], 1), nn.Sigmoid()]
        self.discriminator = nn.Sequential(*disc_layers)

    def forward(self, x):
        return self.discriminator(x)

    def generate(self, z):
        return self.generator(z)


class CustomTransformer(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        d_model = layers[0]
        nhead = 2
        self.embedding = nn.Linear(2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=len(layers))
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class CustomDiffuser(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.activations = activations
        h = layers[0]
        h2 = layers[1] if len(layers) > 1 else 64
        self.encoder1 = nn.Linear(2, h)
        self.encoder2 = nn.Linear(h, h2)
        self.bottleneck = nn.Linear(h2, h2)
        self.decoder1 = nn.Linear(h2, h)
        self.decoder2 = nn.Linear(h, 2)
        self.time_embed = nn.Linear(1, h)
        self.output = nn.Linear(2, 1)

    def forward(self, x, t=None):
        if t is None:
            t = torch.zeros(x.size(0), 1)
        t_emb = self.time_embed(t)
        e1 = F.relu(self.encoder1(x)) + t_emb
        e2 = F.relu(self.encoder2(e1))
        b = F.relu(self.bottleneck(e2))
        d1 = F.relu(self.decoder1(b)) + e1
        d2 = F.relu(self.decoder2(d1))
        return torch.sigmoid(self.output(d2))


MODEL_CLASSES = {
    'ANN': CustomNN,
    'CNN': CustomCNN,
    'RNN': CustomRNN,
    'LSTM': CustomLSTM,
    'GAN': CustomGAN,
    'Transformer': CustomTransformer,
    'Diffuser': CustomDiffuser,
}
