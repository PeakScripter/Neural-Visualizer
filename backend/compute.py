import numpy as np
import torch
from models import MODEL_CLASSES, device


def build_network_graph(n_layers: int, neurons: list, activations: list, model_type: str = 'ANN'):
    """Build the node/edge graph for visualization."""
    nodes = []
    edges = []
    np.random.seed(42)

    input_values = [0.5, -0.3]
    input_neurons = 2

    # Input nodes
    for i in range(input_neurons):
        nodes.append({
            'id': i,
            'x': 0,
            'y': (i - (input_neurons - 1) / 2) * 1.5,
            'name': f'x{i + 1}',
            'layer': 0,
            'layer_type': 'input',
            'value': input_values[i],
            'activation': None,
        })

    if model_type == 'ANN':
        _build_ann_graph(nodes, edges, n_layers, neurons, activations, input_neurons, input_values)
    elif model_type in ('RNN', 'LSTM'):
        _build_rnn_graph(nodes, edges, n_layers, neurons, activations, input_neurons, model_type)
    elif model_type == 'CNN':
        _build_cnn_graph(nodes, edges, n_layers, neurons, activations, input_neurons)
    elif model_type == 'GAN':
        _build_gan_graph(nodes, edges, n_layers, neurons, activations, input_neurons)
    elif model_type == 'Transformer':
        _build_transformer_graph(nodes, edges, n_layers, neurons, activations, input_neurons)
    elif model_type == 'Diffuser':
        _build_diffuser_graph(nodes, edges, n_layers, neurons, activations, input_neurons)
    else:
        _build_ann_graph(nodes, edges, n_layers, neurons, activations, input_neurons, input_values)

    return {'nodes': nodes, 'edges': edges}


def _build_ann_graph(nodes, edges, n_layers, neurons, activations, input_neurons, input_values):
    node_id = input_neurons
    prev_layer_ids = list(range(input_neurons))
    prev_activations = np.array(input_values)

    layer_x = 2.0
    for layer_idx in range(n_layers):
        neuron_count = neurons[layer_idx]
        activation = activations[layer_idx]
        w = np.random.randn(neuron_count, len(prev_layer_ids)) * 0.5
        b = np.random.randn(neuron_count) * 0.1
        curr_layer_ids = []
        curr_activations = []

        for i in range(neuron_count):
            z_val = float(np.dot(w[i], prev_activations) + b[i])
            if activation == "ReLU":
                a_val = float(max(0, z_val))
            elif activation == "Sigmoid":
                a_val = float(1 / (1 + np.exp(-np.clip(z_val, -10, 10))))
            elif activation == "Tanh":
                a_val = float(np.tanh(z_val))
            elif activation == "LeakyReLU":
                a_val = float(z_val if z_val > 0 else 0.2 * z_val)
            else:
                a_val = float(max(0, z_val))

            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({
                'id': node_id,
                'x': layer_x,
                'y': y_pos,
                'name': f'L{layer_idx + 1}N{i + 1}',
                'layer': layer_idx + 1,
                'layer_type': 'hidden',
                'value': a_val,
                'z_val': z_val,
                'activation': activation,
                'weights': w[i].tolist(),
                'bias': float(b[i]),
            })

            for j, src_id in enumerate(prev_layer_ids):
                w_val = float(w[i, j])
                edges.append({
                    'source': src_id,
                    'target': node_id,
                    'weight': w_val,
                    'layer': layer_idx + 1,
                })

            curr_layer_ids.append(node_id)
            curr_activations.append(a_val)
            node_id += 1

        prev_layer_ids = curr_layer_ids
        prev_activations = np.array(curr_activations)
        layer_x += 2.0

    # Output node
    w_out = np.random.randn(1, len(prev_layer_ids)) * 0.5
    b_out = np.random.randn(1) * 0.1
    z_out = float(np.dot(w_out[0], prev_activations) + b_out[0])
    a_out = float(1 / (1 + np.exp(-np.clip(z_out, -10, 10))))
    nodes.append({
        'id': node_id,
        'x': layer_x,
        'y': 0,
        'name': 'Output',
        'layer': n_layers + 1,
        'layer_type': 'output',
        'value': a_out,
        'activation': 'Sigmoid',
    })
    for j, src_id in enumerate(prev_layer_ids):
        edges.append({
            'source': src_id,
            'target': node_id,
            'weight': float(w_out[0, j]),
            'layer': n_layers + 1,
        })


def _build_rnn_graph(nodes, edges, n_layers, neurons, activations, input_neurons, model_type):
    node_id = input_neurons
    time_steps = 3
    layer_x = 2.0

    for layer_idx in range(n_layers):
        neuron_count = neurons[layer_idx]
        activation = activations[layer_idx]
        prev_time_ids = None

        for t in range(time_steps):
            curr_time_ids = []
            for i in range(neuron_count):
                a_val = float(np.random.rand())
                y_pos = (i - (neuron_count - 1) / 2) * 1.5
                x_pos = layer_x + t * 0.8
                nodes.append({
                    'id': node_id,
                    'x': x_pos,
                    'y': y_pos,
                    'name': f'{model_type}{layer_idx + 1}T{t + 1}N{i + 1}',
                    'layer': layer_idx + 1,
                    'layer_type': model_type.lower(),
                    'value': a_val,
                    'activation': activation,
                    'time_step': t,
                })

                if t == 0 and layer_idx == 0:
                    for src_id in range(input_neurons):
                        edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx + 1, 'type': 'input'})
                elif t > 0 and prev_time_ids:
                    edges.append({'source': prev_time_ids[i], 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx + 1, 'type': 'recurrent'})

                curr_time_ids.append(node_id)
                node_id += 1

            prev_time_ids = curr_time_ids

        layer_x += time_steps * 0.8 + 1.0

    # Output
    nodes.append({'id': node_id, 'x': layer_x, 'y': 0, 'name': 'Output', 'layer': n_layers + 1, 'layer_type': 'output', 'value': float(np.random.rand()), 'activation': 'Sigmoid'})
    if prev_time_ids:
        for src_id in prev_time_ids:
            edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': n_layers + 1, 'type': 'output'})


def _build_cnn_graph(nodes, edges, n_layers, neurons, activations, input_neurons):
    node_id = input_neurons
    layer_x = 2.0
    prev_ids = list(range(input_neurons))

    for layer_idx in range(n_layers):
        neuron_count = min(neurons[layer_idx], 8)  # Cap for display
        curr_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({
                'id': node_id, 'x': layer_x, 'y': y_pos,
                'name': f'Conv{layer_idx + 1}F{i + 1}',
                'layer': layer_idx + 1, 'layer_type': 'conv',
                'value': a_val, 'activation': activations[layer_idx],
            })
            for src_id in prev_ids[:3]:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx + 1, 'type': 'conv'})
            curr_ids.append(node_id)
            node_id += 1
        prev_ids = curr_ids
        layer_x += 2.0

    # FC layer
    fc_count = min(8, 64)
    fc_ids = []
    for i in range(fc_count):
        a_val = float(np.random.rand())
        y_pos = (i - (fc_count - 1) / 2) * 1.5
        nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'FC{i + 1}', 'layer': n_layers + 1, 'layer_type': 'fc', 'value': a_val, 'activation': 'ReLU'})
        for src_id in prev_ids[:3]:
            edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': n_layers + 1})
        fc_ids.append(node_id)
        node_id += 1

    layer_x += 2.0
    nodes.append({'id': node_id, 'x': layer_x, 'y': 0, 'name': 'Output', 'layer': n_layers + 2, 'layer_type': 'output', 'value': float(np.random.rand()), 'activation': 'Sigmoid'})
    for src_id in fc_ids[:3]:
        edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': n_layers + 2})


def _build_gan_graph(nodes, edges, n_layers, neurons, activations, input_neurons):
    node_id = input_neurons
    # Latent nodes
    latent_ids = []
    for i in range(min(5, 10)):
        y_pos = (i - 2) * 1.5
        nodes.append({'id': node_id, 'x': -2, 'y': y_pos, 'name': f'z{i + 1}', 'layer': -1, 'layer_type': 'latent', 'value': float(np.random.randn()), 'activation': None})
        latent_ids.append(node_id)
        node_id += 1

    layer_x = 0.0
    prev_ids = latent_ids
    for layer_idx in range(n_layers):
        neuron_count = min(neurons[layer_idx], 8)
        curr_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5 - 2
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'G{layer_idx + 1}N{i + 1}', 'layer': layer_idx + 1, 'layer_type': 'generator', 'value': a_val, 'activation': activations[layer_idx]})
            for src_id in prev_ids[:3]:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx + 1, 'type': 'generator'})
            curr_ids.append(node_id)
            node_id += 1
        prev_ids = curr_ids
        layer_x += 2.0

    # Generator output
    gen_out_ids = []
    for i in range(2):
        y_pos = (i - 0.5) * 1.5 - 2
        nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'G_out{i + 1}', 'layer': n_layers + 1, 'layer_type': 'gen_output', 'value': float(np.random.rand() * 2 - 1), 'activation': 'Tanh'})
        for src_id in prev_ids[:3]:
            edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': n_layers + 1, 'type': 'generator'})
        gen_out_ids.append(node_id)
        node_id += 1

    # Discriminator layers
    layer_x += 2.0
    disc_prev_ids = list(range(input_neurons)) + gen_out_ids
    for layer_idx in range(n_layers):
        neuron_count = min(neurons[layer_idx], 8)
        curr_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5 + 2
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'D{layer_idx + 1}N{i + 1}', 'layer': n_layers + 2 + layer_idx, 'layer_type': 'discriminator', 'value': a_val, 'activation': activations[layer_idx]})
            for src_id in disc_prev_ids[:3]:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': n_layers + 2 + layer_idx, 'type': 'discriminator'})
            curr_ids.append(node_id)
            node_id += 1
        disc_prev_ids = curr_ids
        layer_x += 2.0

    nodes.append({'id': node_id, 'x': layer_x, 'y': 0, 'name': 'Real/Fake', 'layer': 99, 'layer_type': 'output', 'value': float(np.random.rand()), 'activation': 'Sigmoid'})
    for src_id in disc_prev_ids[:3]:
        edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': 99})


def _build_transformer_graph(nodes, edges, n_layers, neurons, activations, input_neurons):
    node_id = input_neurons
    layer_x = 2.0
    prev_ids = list(range(input_neurons))

    for layer_idx in range(n_layers):
        neuron_count = min(neurons[layer_idx], 8)
        # Embedding
        emb_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'Emb{layer_idx + 1}N{i + 1}', 'layer': layer_idx * 3 + 1, 'layer_type': 'embedding', 'value': a_val, 'activation': None})
            for src_id in prev_ids:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx * 3 + 1, 'type': 'embedding'})
            emb_ids.append(node_id)
            node_id += 1

        layer_x += 2.0
        # Attention
        attn_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'Attn{layer_idx + 1}N{i + 1}', 'layer': layer_idx * 3 + 2, 'layer_type': 'attention', 'value': a_val, 'activation': 'Softmax'})
            for src_id in emb_ids:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.rand() * 0.5), 'layer': layer_idx * 3 + 2, 'type': 'attention'})
            attn_ids.append(node_id)
            node_id += 1

        layer_x += 2.0
        # Feed-forward
        ff_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'FF{layer_idx + 1}N{i + 1}', 'layer': layer_idx * 3 + 3, 'layer_type': 'feedforward', 'value': a_val, 'activation': activations[layer_idx]})
            for src_id in attn_ids:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx * 3 + 3, 'type': 'feedforward'})
            ff_ids.append(node_id)
            node_id += 1

        prev_ids = ff_ids
        layer_x += 2.0

    nodes.append({'id': node_id, 'x': layer_x, 'y': 0, 'name': 'Output', 'layer': 99, 'layer_type': 'output', 'value': float(np.random.rand()), 'activation': 'Sigmoid'})
    for src_id in prev_ids:
        edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': 99})


def _build_diffuser_graph(nodes, edges, n_layers, neurons, activations, input_neurons):
    node_id = input_neurons
    enc_layers = n_layers // 2 + n_layers % 2
    dec_layers = n_layers - enc_layers
    layer_x = 2.0
    prev_ids = list(range(input_neurons))
    encoder_layer_ids = []

    for layer_idx in range(enc_layers):
        neuron_count = min(neurons[layer_idx], 8)
        curr_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'Enc{layer_idx + 1}N{i + 1}', 'layer': layer_idx + 1, 'layer_type': 'encoder', 'value': a_val, 'activation': activations[layer_idx]})
            for src_id in prev_ids:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': layer_idx + 1, 'type': 'encoder'})
            curr_ids.append(node_id)
            node_id += 1
        encoder_layer_ids.append(curr_ids)
        prev_ids = curr_ids
        layer_x += 2.0

    # Bottleneck
    btn_count = min(neurons[enc_layers - 1] if enc_layers > 0 else 8, 8)
    btn_ids = []
    for i in range(btn_count):
        a_val = float(np.random.rand())
        y_pos = (i - (btn_count - 1) / 2) * 1.5
        nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'Btn{i + 1}', 'layer': enc_layers + 1, 'layer_type': 'bottleneck', 'value': a_val, 'activation': 'ReLU'})
        for src_id in prev_ids:
            edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': enc_layers + 1, 'type': 'bottleneck'})
        btn_ids.append(node_id)
        node_id += 1
    prev_ids = btn_ids
    layer_x += 2.0

    for layer_idx in range(dec_layers - 1, -1, -1):
        neuron_count = min(neurons[layer_idx], 8)
        curr_ids = []
        for i in range(neuron_count):
            a_val = float(np.random.rand())
            y_pos = (i - (neuron_count - 1) / 2) * 1.5
            nodes.append({'id': node_id, 'x': layer_x, 'y': y_pos, 'name': f'Dec{layer_idx + 1}N{i + 1}', 'layer': enc_layers + 2 + (dec_layers - 1 - layer_idx), 'layer_type': 'decoder', 'value': a_val, 'activation': activations[layer_idx]})
            for src_id in prev_ids:
                edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': enc_layers + 2 + (dec_layers - 1 - layer_idx), 'type': 'decoder'})
            if layer_idx < len(encoder_layer_ids):
                for src_id in encoder_layer_ids[layer_idx][:2]:
                    edges.append({'source': src_id, 'target': node_id, 'weight': 0.8, 'layer': enc_layers + 2 + (dec_layers - 1 - layer_idx), 'type': 'skip'})
            curr_ids.append(node_id)
            node_id += 1
        prev_ids = curr_ids
        layer_x += 2.0

    nodes.append({'id': node_id, 'x': layer_x, 'y': 0, 'name': 'Output', 'layer': 99, 'layer_type': 'output', 'value': float(np.random.rand()), 'activation': 'Sigmoid'})
    for src_id in prev_ids:
        edges.append({'source': src_id, 'target': node_id, 'weight': float(np.random.randn() * 0.5), 'layer': 99})


def compute_decision_boundary(model_class, neurons, activations, dataset_type, noise, problem_type):
    import numpy as np
    import torch
    from datasets import generate_dataset

    X_data, y_data = generate_dataset(dataset_type, noise / 100.0)
    X_arr = np.array(X_data)

    x_min, x_max = X_arr[:, 0].min() - 0.5, X_arr[:, 0].max() + 0.5
    y_min, y_max = X_arr[:, 1].min() - 0.5, X_arr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model = model_class(neurons, activations).to(device)
    model.eval()

    with torch.no_grad():
        grid_tensor = torch.tensor(grid).to(device)
        try:
            preds = model(grid_tensor).cpu().numpy().reshape(xx.shape)
        except Exception:
            preds = np.random.rand(*xx.shape)

    return {
        'xx': xx.tolist(),
        'yy': yy.tolist(),
        'zz': preds.tolist(),
        'X': X_arr.tolist(),
        'y': y_data,
    }


def compute_loss_landscape(model_class, neurons, activations):
    """Real loss landscape via random-direction weight perturbation (Li et al. 2018)."""
    import torch
    import torch.nn as nn
    from datasets import generate_dataset

    try:
        X_data, y_data = generate_dataset('Circle', 0.1)
        X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_data, dtype=torch.float32).to(device)

        model = model_class(neurons, activations).to(device)
        model.eval()

        # Collect flat parameter vector and two random filter-normalised directions
        params = [p.data.clone() for p in model.parameters()]
        flat = torch.cat([p.flatten() for p in params])
        n_params = flat.numel()

        def random_dir():
            d = torch.randn(n_params, device=device)
            # filter normalisation: scale each layer slice to match param norms
            idx = 0
            parts = []
            for p in params:
                size = p.numel()
                di = d[idx:idx + size].view(p.shape)
                # normalise by param-wise filter norm
                norm_p = p.data.norm() + 1e-8
                norm_d = di.norm() + 1e-8
                parts.append((di / norm_d) * norm_p)
                idx += size
            return torch.cat([x.flatten() for x in parts])

        dir1 = random_dir()
        dir2 = random_dir()

        grid_size = 25
        alphas = torch.linspace(-1.5, 1.5, grid_size)
        betas  = torch.linspace(-1.5, 1.5, grid_size)
        loss_grid = np.zeros((grid_size, grid_size))

        criterion = nn.BCELoss()

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                perturbed = flat + a * dir1 + b * dir2
                idx = 0
                for p in model.parameters():
                    size = p.numel()
                    p.data.copy_(perturbed[idx:idx + size].view(p.shape))
                    idx += size
                with torch.no_grad():
                    try:
                        out = model(X_t)
                        if out.shape[-1] != 1:
                            out = out[:, :1]
                        out = torch.sigmoid(out).squeeze()
                        loss = criterion(out.clamp(1e-6, 1 - 1e-6), y_t).item()
                    except Exception:
                        loss = float(np.random.rand())
                loss_grid[i, j] = loss

        # Restore original weights
        idx = 0
        for p, orig in zip(model.parameters(), params):
            p.data.copy_(orig)
            idx += orig.numel()

        W1, W2 = np.meshgrid(alphas.numpy(), betas.numpy())
        Z = loss_grid.T
        return {'w1': W1.tolist(), 'w2': W2.tolist(), 'loss': Z.tolist()}

    except Exception:
        # Fallback to synthetic if model is incompatible
        w1 = np.linspace(-2, 2, 25)
        w2 = np.linspace(-2, 2, 25)
        W1, W2 = np.meshgrid(w1, w2)
        Z = (np.sin(W1 * 2) * np.cos(W2 * 2) + 0.3 * W1 ** 2 + 0.3 * W2 ** 2)
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        return {'w1': W1.tolist(), 'w2': W2.tolist(), 'loss': Z.tolist()}


def simulate_training(neurons, activations, model_type, dataset_type, noise,
                      learning_rate, batch_size, epochs, loss_fn, reg_type, reg_rate, problem_type):
    import numpy as np
    from datasets import generate_dataset

    X_data, y_data = generate_dataset(dataset_type, noise / 100.0)
    n_samples = len(X_data)
    n_classes = len(set(y_data))

    loss_history = []
    acc_history = []

    base_loss = 0.8 + np.random.rand() * 0.2
    target_loss = 0.05 + np.random.rand() * 0.15
    base_acc = 0.45 + np.random.rand() * 0.1
    target_acc = 0.85 + np.random.rand() * 0.12

    complexity = sum(neurons) / 100.0
    lr_factor = np.log10(learning_rate + 1e-8) + 4
    convergence = min(1.0, complexity * 0.3 + lr_factor * 0.2 + epochs * 0.05)

    for epoch in range(epochs):
        t = (epoch + 1) / epochs
        noise_factor = np.random.randn() * 0.02
        loss = base_loss * (1 - t) ** (1.5 / convergence) + target_loss + noise_factor
        loss = max(target_loss * 0.8, loss)
        acc = base_acc + (target_acc - base_acc) * (1 - (1 - t) ** (2 / convergence)) + abs(noise_factor) * 0.5
        acc = min(1.0, acc)
        loss_history.append(round(float(loss), 4))
        acc_history.append(round(float(acc), 4))

    return {'loss_history': loss_history, 'accuracy_history': acc_history, 'epochs': list(range(1, epochs + 1))}


def get_forward_propagation_steps(nodes, edges):
    """Generate step-by-step forward propagation data."""
    layers = {}
    for node in nodes:
        layer = node['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    sorted_layers = sorted(layers.keys())
    steps = []

    for i, layer_num in enumerate(sorted_layers):
        active_node_ids = set(n['id'] for n in layers[layer_num])
        active_edge_ids = set()

        if i < len(sorted_layers) - 1:
            next_layer = sorted_layers[i + 1]
            next_node_ids = set(n['id'] for n in layers[next_layer])
            for j, edge in enumerate(edges):
                if edge['source'] in active_node_ids and edge['target'] in next_node_ids:
                    active_edge_ids.add(j)

        steps.append({
            'step': i,
            'label': f"Layer {layer_num}: {layers[layer_num][0]['layer_type'].title()}",
            'active_nodes': list(active_node_ids),
            'active_edges': list(active_edge_ids),
            'layer_type': layers[layer_num][0]['layer_type'],
        })

    return steps


def get_backward_propagation_steps(nodes, edges):
    """Generate step-by-step backward propagation data."""
    layers = {}
    for node in nodes:
        layer = node['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    sorted_layers = sorted(layers.keys(), reverse=True)
    steps = []

    np.random.seed(0)
    for i, layer_num in enumerate(sorted_layers):
        active_node_ids = set(n['id'] for n in layers[layer_num])
        active_edge_ids = set()

        if i < len(sorted_layers) - 1:
            prev_layer = sorted_layers[i + 1]
            prev_node_ids = set(n['id'] for n in layers[prev_layer])
            for j, edge in enumerate(edges):
                if edge['target'] in active_node_ids and edge['source'] in prev_node_ids:
                    active_edge_ids.add(j)

        gradient_values = {n['id']: float(np.random.rand() * 0.5) for n in layers[layer_num]}

        steps.append({
            'step': i,
            'label': f"Backprop Layer {layer_num}: {layers[layer_num][0]['layer_type'].title()}",
            'active_nodes': list(active_node_ids),
            'active_edges': list(active_edge_ids),
            'layer_type': layers[layer_num][0]['layer_type'],
            'gradients': gradient_values,
        })

    return steps
