import { useState } from 'react';
import { Copy, Check, Code2 } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';

function act(activations: string[], i: number): string {
  const m: Record<string, string> = {
    ReLU: 'nn.ReLU()', Sigmoid: 'nn.Sigmoid()', Tanh: 'nn.Tanh()',
    LeakyReLU: 'nn.LeakyReLU(0.2)', ELU: 'nn.ELU()',
  };
  return m[activations[i]] ?? 'nn.ReLU()';
}

function loss(lossFn: string): string {
  const m: Record<string, string> = {
    'Binary Cross Entropy': 'nn.BCELoss()',
    'Mean Squared Error': 'nn.MSELoss()',
    'Categorical Cross Entropy': 'nn.CrossEntropyLoss()',
    'Hinge Loss': 'nn.HingeEmbeddingLoss()',
    'Sequence Loss': 'nn.CrossEntropyLoss()',
    'Wasserstein Loss': '# Wasserstein: use critic output mean directly',
    'L1 Loss': 'nn.L1Loss()',
    'Huber Loss': 'nn.HuberLoss()',
  };
  return m[lossFn] ?? 'nn.BCELoss()';
}

function generateANN(n_layers: number, neurons: number[], activations: string[], lossFn: string, regType: string, regRate: number): string {
  const layers: string[] = ['nn.Linear(2, ' + neurons[0] + ')', act(activations, 0)];
  for (let i = 1; i < n_layers; i++) {
    layers.push('nn.Linear(' + neurons[i - 1] + ', ' + neurons[i] + ')');
    layers.push(act(activations, i));
  }
  layers.push('nn.Linear(' + neurons[n_layers - 1] + ', 1)', 'nn.Sigmoid()');
  const reg = regType !== 'None'
    ? '\n    # Regularization: ' + regType + ' weight_decay=' + regRate + '\n    # Pass weight_decay=' + regRate + ' to optimizer'
    : '';
  const wd = regType !== 'None' ? ', weight_decay=' + regRate : '';
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'model = nn.Sequential(',
    '    ' + layers.join(',\n    '),
    ')',
    '',
    'criterion = ' + loss(lossFn),
    'optimizer = torch.optim.Adam(model.parameters(), lr=0.01' + wd + ')' + reg,
    '',
    '# Training loop',
    'def train(X, y, epochs=50):',
    '    for epoch in range(epochs):',
    '        optimizer.zero_grad()',
    '        pred = model(X)',
    '        loss_val = criterion(pred.squeeze(), y.float())',
    '        loss_val.backward()',
    '        optimizer.step()',
  ].join('\n');
}

function generateCNN(n_layers: number, neurons: number[], activations: string[], lossFn: string, regType: string, regRate: number): string {
  const convLines: string[] = [];
  for (let i = 0; i < n_layers; i++) {
    const inC = i === 0 ? 1 : Math.max(1, neurons[i - 1] >> 3);
    const outC = Math.max(1, neurons[i] >> 3);
    convLines.push('            nn.Conv2d(' + inC + ', ' + outC + ', kernel_size=3, padding=1),');
    convLines.push('            ' + act(activations, i) + ',');
    convLines.push('            nn.MaxPool2d(2),');
  }
  const wd = regType !== 'None' ? ', weight_decay=' + regRate : '';
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class CNN(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        self.conv_layers = nn.Sequential(',
    convLines.join('\n'),
    '        )',
    '        self.fc = nn.Sequential(',
    '            nn.Flatten(),',
    '            nn.Linear(' + Math.max(1, neurons[n_layers - 1] >> 3) + ', 64),',
    '            nn.ReLU(),',
    '            nn.Linear(64, 1),',
    '            nn.Sigmoid(),',
    '        )',
    '    def forward(self, x):',
    '        return self.fc(self.conv_layers(x))',
    '',
    'model = CNN()',
    'criterion = ' + loss(lossFn),
    'optimizer = torch.optim.Adam(model.parameters(), lr=0.001' + wd + ')',
  ].join('\n');
}

function generateRNN(modelType: string, n_layers: number, neurons: number[], lossFn: string, regType: string, regRate: number): string {
  const cell = modelType === 'LSTM' ? 'nn.LSTM' : 'nn.RNN';
  const wd = regType !== 'None' ? ', weight_decay=' + regRate : '';
  const drop = n_layers > 1 ? 0.2 : 0;
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class ' + modelType + 'Model(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        self.rnn = ' + cell + '(',
    '            input_size=2,',
    '            hidden_size=' + neurons[0] + ',',
    '            num_layers=' + n_layers + ',',
    '            batch_first=True,',
    '            dropout=' + drop + ',',
    '        )',
    '        self.fc = nn.Linear(' + neurons[0] + ', 1)',
    '        self.sigmoid = nn.Sigmoid()',
    '',
    '    def forward(self, x):',
    '        out, _ = self.rnn(x)',
    '        return self.sigmoid(self.fc(out[:, -1, :]))',
    '',
    'model = ' + modelType + 'Model()',
    'criterion = ' + loss(lossFn),
    'optimizer = torch.optim.Adam(model.parameters(), lr=0.001' + wd + ')',
  ].join('\n');
}

function generateTransformer(n_layers: number, neurons: number[], lossFn: string, regType: string, regRate: number): string {
  const wd = regType !== 'None' ? ', weight_decay=' + regRate : '';
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class TransformerModel(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        self.embed = nn.Linear(2, ' + neurons[0] + ')',
    '        encoder_layer = nn.TransformerEncoderLayer(',
    '            d_model=' + neurons[0] + ', nhead=4,',
    '            dim_feedforward=' + (neurons[0] * 4) + ', dropout=0.1, batch_first=True,',
    '        )',
    '        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=' + n_layers + ')',
    '        self.fc = nn.Linear(' + neurons[0] + ', 1)',
    '        self.sigmoid = nn.Sigmoid()',
    '',
    '    def forward(self, x):',
    '        x = self.embed(x.unsqueeze(1))',
    '        x = self.encoder(x)',
    '        return self.sigmoid(self.fc(x.mean(dim=1)))',
    '',
    'model = TransformerModel()',
    'criterion = ' + loss(lossFn),
    'optimizer = torch.optim.Adam(model.parameters(), lr=0.0001' + wd + ')',
  ].join('\n');
}

function generateGAN(n_layers: number, neurons: number[]): string {
  const genHidden: string[] = [];
  for (let i = 0; i < n_layers - 1; i++) {
    genHidden.push('            nn.Linear(' + neurons[i] + ', ' + neurons[i + 1] + '), nn.ReLU(),');
  }
  const discHidden: string[] = [];
  for (let i = 0; i < n_layers - 1; i++) {
    discHidden.push('            nn.Linear(' + neurons[i] + ', ' + neurons[i + 1] + '), nn.LeakyReLU(0.2),');
  }
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class Generator(nn.Module):',
    '    def __init__(self, latent_dim=10):',
    '        super().__init__()',
    '        self.net = nn.Sequential(',
    '            nn.Linear(latent_dim, ' + neurons[0] + '), nn.ReLU(),',
    ...genHidden,
    '            nn.Linear(' + neurons[n_layers - 1] + ', 2), nn.Tanh(),',
    '        )',
    '    def forward(self, z): return self.net(z)',
    '',
    'class Discriminator(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        self.net = nn.Sequential(',
    '            nn.Linear(2, ' + neurons[0] + '), nn.LeakyReLU(0.2),',
    ...discHidden,
    '            nn.Linear(' + neurons[n_layers - 1] + ', 1), nn.Sigmoid(),',
    '        )',
    '    def forward(self, x): return self.net(x)',
    '',
    'G, D = Generator(), Discriminator()',
    'criterion = nn.BCELoss()',
    'opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))',
    'opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))',
  ].join('\n');
}

function generateDiffuser(n_layers: number, neurons: number[]): string {
  const half = Math.ceil(n_layers / 2);
  const encLines: string[] = [];
  for (let i = 0; i < half; i++) {
    const inN = i === 0 ? 2 : neurons[i - 1];
    encLines.push('            nn.Linear(' + inN + ', ' + neurons[i] + '), nn.SiLU(),');
  }
  const decLines: string[] = [];
  for (let i = 0; i < Math.floor(n_layers / 2); i++) {
    const li = half + i;
    const outN = li < n_layers - 1 ? neurons[li] : 2;
    decLines.push('            nn.Linear(' + neurons[li - 1] + ', ' + outN + '), nn.SiLU(),');
  }
  const midN = neurons[Math.floor(n_layers / 2)] ?? neurons[0];
  return [
    'import torch',
    'import torch.nn as nn',
    '',
    'class Diffuser(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        # Encoder',
    '        self.encoder = nn.Sequential(',
    ...encLines,
    '        )',
    '        # Time embedding',
    '        self.time_embed = nn.Linear(1, ' + midN + ')',
    '        # Decoder',
    '        self.decoder = nn.Sequential(',
    ...decLines,
    '        )',
    '',
    '    def forward(self, x, t):',
    '        h = self.encoder(x) + self.time_embed(t.unsqueeze(-1).float())',
    '        return self.decoder(h)',
    '',
    'model = Diffuser()',
    'criterion = nn.MSELoss()',
    'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)',
  ].join('\n');
}

function generatePyTorch(modelType: string, n_layers: number, neurons: number[], activations: string[], lossFn: string, regType: string, regRate: number): string {
  if (modelType === 'ANN') return generateANN(n_layers, neurons, activations, lossFn, regType, regRate);
  if (modelType === 'CNN') return generateCNN(n_layers, neurons, activations, lossFn, regType, regRate);
  if (modelType === 'RNN' || modelType === 'LSTM') return generateRNN(modelType, n_layers, neurons, lossFn, regType, regRate);
  if (modelType === 'Transformer') return generateTransformer(n_layers, neurons, lossFn, regType, regRate);
  if (modelType === 'GAN') return generateGAN(n_layers, neurons);
  return generateDiffuser(n_layers, neurons);
}

function generateKeras(n_layers: number, neurons: number[], activations: string[]): string {
  const denseLines: string[] = [];
  for (let i = 0; i < n_layers; i++) {
    const actName = (activations[i] ?? 'relu').toLowerCase();
    denseLines.push('    keras.layers.Dense(' + neurons[i] + ", activation='" + actName + "'),");
  }
  return [
    'import tensorflow as tf',
    'from tensorflow import keras',
    '',
    'model = keras.Sequential([',
    '    keras.layers.Input(shape=(2,)),',
    ...denseLines,
    '    keras.layers.Dense(1, activation=\'sigmoid\'),',
    '])',
    '',
    'model.compile(',
    '    optimizer=keras.optimizers.Adam(learning_rate=0.01),',
    "    loss='binary_crossentropy',",
    "    metrics=['accuracy'],",
    ')',
    '',
    '# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)',
  ].join('\n');
}

export function ExportCode() {
  const { networkConfig } = useNetworkStore();
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<'pytorch' | 'keras'>('pytorch');

  const neurons = networkConfig.neurons.slice(0, networkConfig.n_layers);
  const activations = networkConfig.activations.slice(0, networkConfig.n_layers);

  const code = generatePyTorch(
    networkConfig.model_type, networkConfig.n_layers,
    neurons, activations, networkConfig.loss_fn,
    networkConfig.reg_type, networkConfig.reg_rate,
  );

  const kerasCode = generateKeras(networkConfig.n_layers, neurons, activations);

  const display = tab === 'pytorch' ? code : kerasCode;

  const copy = () => {
    navigator.clipboard.writeText(display);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Code2 size={16} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Export — {networkConfig.model_type}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {(['pytorch', 'keras'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="text-xs px-3 py-1 rounded-lg border transition-all"
              style={{
                background: tab === t ? 'var(--accent)' : 'transparent',
                borderColor: tab === t ? 'var(--accent)' : 'var(--border-soft)',
                color: tab === t ? '#fff' : 'var(--text-muted)',
              }}
            >
              {t === 'pytorch' ? 'PyTorch' : 'Keras'}
            </button>
          ))}
          <button
            onClick={copy}
            className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-lg border transition-all"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border-soft)', color: 'var(--text-muted)' }}
          >
            {copied ? <Check size={12} style={{ color: '#10b981' }} /> : <Copy size={12} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      <div
        className="flex-1 rounded-xl overflow-auto border font-mono text-xs leading-relaxed p-4"
        style={{ background: '#0d1117', borderColor: 'var(--border)', color: '#e6edf3' }}
      >
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{display}</pre>
      </div>

      <p className="text-xs text-center" style={{ color: 'var(--text-faint)' }}>
        Generated from current network config · {networkConfig.n_layers} layers · {neurons.join('→')} neurons
      </p>
    </div>
  );
}
