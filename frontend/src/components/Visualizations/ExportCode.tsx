import { useState, useMemo } from 'react';
import { Copy, Check, Code2, FileCode, RotateCcw } from 'lucide-react';
import { useNetworkStore } from '../../store/networkStore';
import type { NetworkConfig, TrainingConfig } from '../../types';

// ── Activation / loss helpers ────────────────────────────────────────────────
function actPy(a: string): string {
  const m: Record<string, string> = {
    ReLU: 'nn.ReLU()', Sigmoid: 'nn.Sigmoid()', Tanh: 'nn.Tanh()',
    LeakyReLU: 'nn.LeakyReLU(0.2)', ELU: 'nn.ELU()', SELU: 'nn.SELU()',
  };
  return m[a] ?? 'nn.ReLU()';
}

function lossPy(l: string): string {
  const m: Record<string, string> = {
    'Binary Cross Entropy': 'nn.BCELoss()',
    'Mean Squared Error': 'nn.MSELoss()',
    'Categorical Cross Entropy': 'nn.CrossEntropyLoss()',
    'Hinge Loss': 'nn.HingeEmbeddingLoss()',
    'Sequence Loss': 'nn.CrossEntropyLoss()',
    'Wasserstein Loss': '# Wasserstein: use critic output mean directly',
    'L1 Loss': 'nn.L1Loss()',
    'Huber Loss': 'nn.HuberLoss()',
    'Cross Entropy': 'nn.CrossEntropyLoss()',
    'Masked Language Model Loss': 'nn.CrossEntropyLoss()',
    'Least Squares': 'nn.MSELoss()',
    'Gradient Penalty': 'nn.BCELoss()  # + gradient penalty term',
  };
  return m[l] ?? 'nn.BCELoss()';
}

// ── Code generators ──────────────────────────────────────────────────────────
function generateANN(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const acts = cfg.activations.slice(0, cfg.n_layers);
  const inp = cfg.input_nodes;
  const out = cfg.output_nodes;
  const wd = cfg.reg_type !== 'None' ? `, weight_decay=${cfg.reg_rate}` : '';
  const layers: string[] = [`nn.Linear(${inp}, ${neurons[0]})`, actPy(acts[0])];
  for (let i = 1; i < cfg.n_layers; i++) {
    layers.push(`nn.Linear(${neurons[i - 1]}, ${neurons[i]})`, actPy(acts[i]));
  }
  layers.push(`nn.Linear(${neurons[cfg.n_layers - 1]}, ${out})`, out > 1 ? 'nn.Softmax(dim=1)' : 'nn.Sigmoid()');
  return [
    'import torch', 'import torch.nn as nn', '',
    'model = nn.Sequential(', '    ' + layers.join(',\n    '), ')', '',
    `criterion = ${lossPy(cfg.loss_fn)}`,
    `optimizer = torch.optim.Adam(model.parameters(), lr=0.01${wd})`, '',
    '# Training loop', 'def train(X, y, epochs=50):',
    '    for epoch in range(epochs):',
    '        optimizer.zero_grad()',
    '        pred = model(X)',
    '        loss_val = criterion(pred.squeeze(), y.float())',
    '        loss_val.backward()', '        optimizer.step()',
  ].join('\n');
}

function generateCNN(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const acts = cfg.activations.slice(0, cfg.n_layers);
  const wd = cfg.reg_type !== 'None' ? `, weight_decay=${cfg.reg_rate}` : '';
  const convLines: string[] = [];
  for (let i = 0; i < cfg.n_layers; i++) {
    const inC = i === 0 ? 1 : Math.max(1, neurons[i - 1] >> 3);
    const outC = Math.max(1, neurons[i] >> 3);
    convLines.push(`            nn.Conv2d(${inC}, ${outC}, kernel_size=3, padding=1),`);
    convLines.push(`            ${actPy(acts[i])},`);
    convLines.push('            nn.MaxPool2d(2),');
  }
  return [
    'import torch', 'import torch.nn as nn', '',
    'class CNN(nn.Module):',
    '    def __init__(self):', '        super().__init__()',
    '        self.conv_layers = nn.Sequential(', convLines.join('\n'), '        )',
    '        self.fc = nn.Sequential(',
    '            nn.Flatten(),',
    `            nn.Linear(${Math.max(1, neurons[cfg.n_layers - 1] >> 3)}, 64),`,
    '            nn.ReLU(),',
    `            nn.Linear(64, ${cfg.output_nodes}),`,
    cfg.output_nodes > 1 ? '            nn.Softmax(dim=1),' : '            nn.Sigmoid(),',
    '        )',
    '    def forward(self, x):', '        return self.fc(self.conv_layers(x))', '',
    'model = CNN()',
    `criterion = ${lossPy(cfg.loss_fn)}`,
    `optimizer = torch.optim.Adam(model.parameters(), lr=0.001${wd})`,
  ].join('\n');
}

function generateRNN(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const cell = cfg.model_type === 'LSTM' ? 'nn.LSTM' : 'nn.RNN';
  const wd = cfg.reg_type !== 'None' ? `, weight_decay=${cfg.reg_rate}` : '';
  return [
    'import torch', 'import torch.nn as nn', '',
    `class ${cfg.model_type}Model(nn.Module):`,
    '    def __init__(self):', '        super().__init__()',
    `        self.rnn = ${cell}(`,
    `            input_size=${cfg.input_nodes},`,
    `            hidden_size=${neurons[0]},`,
    `            num_layers=${cfg.n_layers},`,
    '            batch_first=True,',
    `            dropout=${cfg.n_layers > 1 ? 0.2 : 0},`,
    '        )',
    `        self.fc = nn.Linear(${neurons[0]}, ${cfg.output_nodes})`,
    `        self.out_act = ${cfg.output_nodes > 1 ? 'nn.Softmax(dim=1)' : 'nn.Sigmoid()'}`, '',
    '    def forward(self, x):',
    '        out, _ = self.rnn(x)',
    '        return self.out_act(self.fc(out[:, -1, :]))', '',
    `model = ${cfg.model_type}Model()`,
    `criterion = ${lossPy(cfg.loss_fn)}`,
    `optimizer = torch.optim.Adam(model.parameters(), lr=0.001${wd})`,
  ].join('\n');
}

function generateTransformer(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const wd = cfg.reg_type !== 'None' ? `, weight_decay=${cfg.reg_rate}` : '';
  return [
    'import torch', 'import torch.nn as nn', '',
    'class TransformerModel(nn.Module):',
    '    def __init__(self):', '        super().__init__()',
    `        self.embed = nn.Linear(${cfg.input_nodes}, ${neurons[0]})`,
    '        encoder_layer = nn.TransformerEncoderLayer(',
    `            d_model=${neurons[0]}, nhead=4,`,
    `            dim_feedforward=${neurons[0] * 4}, dropout=0.1, batch_first=True,`,
    '        )',
    `        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=${cfg.n_layers})`,
    `        self.fc = nn.Linear(${neurons[0]}, ${cfg.output_nodes})`,
    `        self.out_act = ${cfg.output_nodes > 1 ? 'nn.Softmax(dim=1)' : 'nn.Sigmoid()'}`, '',
    '    def forward(self, x):',
    '        x = self.embed(x.unsqueeze(1))',
    '        x = self.encoder(x)',
    '        return self.out_act(self.fc(x.mean(dim=1)))', '',
    'model = TransformerModel()',
    `criterion = ${lossPy(cfg.loss_fn)}`,
    `optimizer = torch.optim.Adam(model.parameters(), lr=0.0001${wd})`,
  ].join('\n');
}

function generateGAN(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const genHidden = neurons.slice(0, -1).map((n, i) =>
    `            nn.Linear(${n}, ${neurons[i + 1]}), nn.ReLU(),`
  );
  const discHidden = neurons.slice(0, -1).map((n, i) =>
    `            nn.Linear(${n}, ${neurons[i + 1]}), nn.LeakyReLU(0.2),`
  );
  return [
    'import torch', 'import torch.nn as nn', '',
    'class Generator(nn.Module):',
    '    def __init__(self, latent_dim=10):', '        super().__init__()',
    '        self.net = nn.Sequential(',
    `            nn.Linear(latent_dim, ${neurons[0]}), nn.ReLU(),`,
    ...genHidden,
    `            nn.Linear(${neurons[neurons.length - 1]}, ${cfg.output_nodes}), nn.Tanh(),`,
    '        )',
    '    def forward(self, z): return self.net(z)', '',
    'class Discriminator(nn.Module):',
    '    def __init__(self):', '        super().__init__()',
    '        self.net = nn.Sequential(',
    `            nn.Linear(${cfg.input_nodes}, ${neurons[0]}), nn.LeakyReLU(0.2),`,
    ...discHidden,
    `            nn.Linear(${neurons[neurons.length - 1]}, 1), nn.Sigmoid(),`,
    '        )',
    '    def forward(self, x): return self.net(x)', '',
    'G, D = Generator(), Discriminator()',
    'criterion = nn.BCELoss()',
    'opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))',
    'opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))',
  ].join('\n');
}

function generateDiffuser(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const half = Math.ceil(cfg.n_layers / 2);
  const encLines = Array.from({ length: half }, (_, i) => {
    const inN = i === 0 ? cfg.input_nodes : neurons[i - 1];
    return `            nn.Linear(${inN}, ${neurons[i]}), nn.SiLU(),`;
  });
  const decLines = Array.from({ length: Math.floor(cfg.n_layers / 2) }, (_, i) => {
    const li = half + i;
    const outN = li < cfg.n_layers - 1 ? neurons[li] : cfg.output_nodes;
    return `            nn.Linear(${neurons[li - 1]}, ${outN}), nn.SiLU(),`;
  });
  const midN = neurons[Math.floor(cfg.n_layers / 2)] ?? neurons[0];
  return [
    'import torch', 'import torch.nn as nn', '',
    'class Diffuser(nn.Module):',
    '    def __init__(self):', '        super().__init__()',
    '        self.encoder = nn.Sequential(', ...encLines, '        )',
    `        self.time_embed = nn.Linear(1, ${midN})`,
    '        self.decoder = nn.Sequential(', ...decLines, '        )', '',
    '    def forward(self, x, t):',
    '        h = self.encoder(x) + self.time_embed(t.unsqueeze(-1).float())',
    '        return self.decoder(h)', '',
    'model = Diffuser()',
    'criterion = nn.MSELoss()',
    'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)',
  ].join('\n');
}

function generateKeras(cfg: NetworkConfig): string {
  const neurons = cfg.neurons.slice(0, cfg.n_layers);
  const acts = cfg.activations.slice(0, cfg.n_layers);
  const denseLines = neurons.map((n, i) =>
    `    keras.layers.Dense(${n}, activation='${(acts[i] ?? 'relu').toLowerCase()}'),`
  );
  const outAct = cfg.output_nodes > 1 ? 'softmax' : 'sigmoid';
  return [
    'import tensorflow as tf', 'from tensorflow import keras', '',
    'model = keras.Sequential([',
    `    keras.layers.Input(shape=(${cfg.input_nodes},)),`,
    ...denseLines,
    `    keras.layers.Dense(${cfg.output_nodes}, activation='${outAct}'),`,
    '])', '',
    'model.compile(',
    '    optimizer=keras.optimizers.Adam(learning_rate=0.01),',
    "    loss='binary_crossentropy',",
    "    metrics=['accuracy'],",
    ')', '',
    '# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)',
  ].join('\n');
}

function generateCode(cfg: NetworkConfig): string {
  switch (cfg.model_type) {
    case 'ANN': return generateANN(cfg);
    case 'CNN': return generateCNN(cfg);
    case 'RNN': case 'LSTM': return generateRNN(cfg);
    case 'Transformer': return generateTransformer(cfg);
    case 'GAN': return generateGAN(cfg);
    default: return generateDiffuser(cfg);
  }
}

// ── Template fill ────────────────────────────────────────────────────────────
function fillTemplate(template: string, nc: NetworkConfig, tc: TrainingConfig): string {
  const neurons = nc.neurons.slice(0, nc.n_layers);
  const acts = nc.activations.slice(0, nc.n_layers);
  const actNameMap: Record<string, string> = {
    ReLU: 'ReLU', Sigmoid: 'Sigmoid', Tanh: 'Tanh',
    LeakyReLU: 'LeakyReLU', ELU: 'ELU', SELU: 'SELU',
  };
  const vars: Record<string, string> = {
    model_type:    nc.model_type,
    n_layers:      String(nc.n_layers),
    input_nodes:   String(nc.input_nodes),
    output_nodes:  String(nc.output_nodes),
    neurons:       neurons.join(', '),
    neurons_list:  '[' + neurons.join(', ') + ']',
    neurons_0:     String(neurons[0] ?? 32),
    neurons_last:  String(neurons[neurons.length - 1] ?? 32),
    activations:   acts.join(', '),
    activation_0:  actNameMap[acts[0]] ?? 'ReLU',
    loss_fn:       nc.loss_fn,
    loss_fn_code:  lossPy(nc.loss_fn),
    reg_type:      nc.reg_type,
    reg_rate:      String(nc.reg_rate),
    learning_rate: String(tc.learning_rate),
    batch_size:    String(tc.batch_size),
    epochs:        String(tc.epochs),
    dataset:       tc.dataset,
  };
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => vars[key] ?? `{{${key}}}`);
}

const DEFAULT_TEMPLATE = `# {{model_type}} Neural Network
# Architecture: {{n_layers}} hidden layers
# Input: {{input_nodes}} features  →  Output: {{output_nodes}} neurons

import torch
import torch.nn as nn

class {{model_type}}Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear({{input_nodes}}, {{neurons_0}}),
            nn.{{activation_0}}(),
            # hidden layers: {{neurons}}
            nn.Linear({{neurons_last}}, {{output_nodes}}),
        )

    def forward(self, x):
        return self.layers(x)

model = {{model_type}}Network()
criterion = {{loss_fn_code}}
optimizer = torch.optim.Adam(
    model.parameters(),
    lr={{learning_rate}},
)

# Training: {{epochs}} epochs · batch {{batch_size}} · dataset {{dataset}}
# Regularization: {{reg_type}} (rate: {{reg_rate}})
`;

const VARIABLES_HELP = [
  ['{{model_type}}', 'ANN, CNN, RNN…'],
  ['{{n_layers}}', 'hidden layer count'],
  ['{{input_nodes}}', 'input features'],
  ['{{output_nodes}}', 'output neurons'],
  ['{{neurons}}', 'comma list e.g. 32, 64'],
  ['{{neurons_0}}', 'first layer size'],
  ['{{neurons_last}}', 'last layer size'],
  ['{{activation_0}}', 'first activation'],
  ['{{activations}}', 'all activations'],
  ['{{loss_fn_code}}', 'nn.BCELoss() etc.'],
  ['{{learning_rate}}', 'from training cfg'],
  ['{{batch_size}}', 'from training cfg'],
  ['{{epochs}}', 'from training cfg'],
];

// ── Main component ───────────────────────────────────────────────────────────
type Tab = 'pytorch' | 'keras' | 'template';

export function ExportCode() {
  const { networkConfig, trainingConfig } = useNetworkStore();
  const [copied, setCopied] = useState(false);
  const [tab, setTab] = useState<Tab>('pytorch');
  const [userTemplate, setUserTemplate] = useState(DEFAULT_TEMPLATE);

  const neurons = networkConfig.neurons.slice(0, networkConfig.n_layers);

  const pytorchCode = useMemo(() => generateCode(networkConfig), [networkConfig]);
  const kerasCode   = useMemo(() => generateKeras(networkConfig), [networkConfig]);
  const filledCode  = useMemo(
    () => fillTemplate(userTemplate, networkConfig, trainingConfig),
    [userTemplate, networkConfig, trainingConfig],
  );

  const displayCode = tab === 'pytorch' ? pytorchCode : tab === 'keras' ? kerasCode : filledCode;

  const copy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const TABS: { id: Tab; label: string }[] = [
    { id: 'pytorch',  label: 'PyTorch' },
    { id: 'keras',    label: 'Keras' },
    { id: 'template', label: 'Template' },
  ];

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Header row */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Code2 size={16} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Export — {networkConfig.model_type}
          </span>
          <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
            {networkConfig.input_nodes} in → {neurons.join('→')} → {networkConfig.output_nodes} out
          </span>
        </div>
        <div className="flex items-center gap-2">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className="text-xs px-3 py-1 rounded-lg border transition-all"
              style={{
                background: tab === t.id ? 'var(--accent)' : 'transparent',
                borderColor: tab === t.id ? 'var(--accent)' : 'var(--border-soft)',
                color: tab === t.id ? '#fff' : 'var(--text-muted)',
              }}
            >
              {t.label}
            </button>
          ))}
          <button
            onClick={() => copy(displayCode)}
            className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-lg border transition-all"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border-soft)', color: 'var(--text-muted)' }}
          >
            {copied ? <Check size={12} style={{ color: '#10b981' }} /> : <Copy size={12} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      </div>

      {tab === 'template' ? (
        /* ── Template editor ─────────────────────────────────────────── */
        <div className="flex-1 min-h-0 flex gap-3">
          {/* Left: editable template */}
          <div className="flex flex-col gap-1.5" style={{ flex: '0 0 48%' }}>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                <FileCode size={11} className="inline mr-1" />Edit template
              </span>
              <button
                onClick={() => setUserTemplate(DEFAULT_TEMPLATE)}
                className="flex items-center gap-1 text-xs px-2 py-0.5 rounded border transition-all"
                style={{ borderColor: 'var(--border-soft)', color: 'var(--text-faint)' }}
              >
                <RotateCcw size={10} /> Reset
              </button>
            </div>
            <textarea
              className="flex-1 font-mono text-xs rounded-xl border p-3 resize-none leading-relaxed"
              style={{
                background: '#0d1117', borderColor: 'var(--border)',
                color: '#e6edf3', outline: 'none', minHeight: 0,
              }}
              value={userTemplate}
              onChange={e => setUserTemplate(e.target.value)}
              spellCheck={false}
            />
            {/* Variables reference */}
            <div className="rounded-lg border p-2" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
              <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>Available variables</p>
              <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                {VARIABLES_HELP.map(([v, desc]) => (
                  <div key={v} className="flex gap-1.5 items-baseline">
                    <code className="text-xs" style={{ color: '#a78bfa' }}>{v}</code>
                    <span className="text-xs truncate" style={{ color: 'var(--text-faint)' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: filled output */}
          <div className="flex flex-col gap-1.5 flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Filled output (live preview)
              </span>
              <button
                onClick={() => copy(filledCode)}
                className="flex items-center gap-1 text-xs px-2 py-0.5 rounded border transition-all"
                style={{ borderColor: 'var(--border-soft)', color: 'var(--text-faint)' }}
              >
                <Copy size={10} /> Copy
              </button>
            </div>
            <div
              className="flex-1 rounded-xl overflow-auto border font-mono text-xs leading-relaxed p-3"
              style={{ background: '#0d1117', borderColor: 'var(--border)', color: '#e6edf3', minHeight: 0 }}
            >
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{filledCode}</pre>
            </div>
          </div>
        </div>
      ) : (
        /* ── Regular code view ───────────────────────────────────────── */
        <div
          className="flex-1 rounded-xl overflow-auto border font-mono text-xs leading-relaxed p-4"
          style={{ background: '#0d1117', borderColor: 'var(--border)', color: '#e6edf3' }}
        >
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{displayCode}</pre>
        </div>
      )}

      <p className="text-xs text-center flex-shrink-0" style={{ color: 'var(--text-faint)' }}>
        {networkConfig.input_nodes} input → {neurons.join('→')} → {networkConfig.output_nodes} output ·{' '}
        {networkConfig.n_layers} layers · {networkConfig.model_type}
      </p>
    </div>
  );
}
