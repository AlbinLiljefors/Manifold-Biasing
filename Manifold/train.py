import torch
import torch.nn as nn

from noise import add_class_noise


def train_model(model, train_loader, epochs=10, lr=1e-3, recon_lambda=0.0,
                device='cpu', verbose=True, param_groups=None,
                noise_params=None):
    model.to(device)
    model.train()

    if param_groups is not None:
        groups = []
        matched_params = set()
        for pg in param_groups:
            group_params = []
            for name, p in model.named_parameters():
                if p.requires_grad and pg['filter'](name):
                    group_params.append(p)
                    matched_params.add(name)
            group_cfg = {'params': group_params, 'lr': pg['lr']}
            if 'weight_decay' in pg:
                group_cfg['weight_decay'] = pg['weight_decay']
            groups.append(group_cfg)
        default_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and n not in matched_params]
        if default_params:
            groups.append({'params': default_params, 'lr': lr})
        optimizer = torch.optim.Adam(groups)
    else:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if noise_params is not None:
                x = add_class_noise(x, y, *noise_params)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if recon_lambda > 0 and hasattr(model, 'compute_recon_loss'):
                loss = loss + recon_lambda * model.compute_recon_loss(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f}, "
                  f"acc={100.0*correct/total:.1f}%")

    return model


def evaluate_model(model, test_loader, device='cpu', noise_sigma=0.0,
                   noise_params=None):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if noise_params is not None:
                x = add_class_noise(x, y, *noise_params)
            elif noise_sigma > 0:
                x = x + noise_sigma * torch.randn_like(x)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return 100.0 * correct / total
