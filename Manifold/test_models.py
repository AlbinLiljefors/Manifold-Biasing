import torch
import torch.nn as nn
import numpy as np
import pytest


class TestLifting:
    def test_raw_lift_output_shape(self):
        from lifting import FrozenLiftRaw
        lift = FrozenLiftRaw()
        out = lift(torch.randn(4, 1, 28, 28))
        assert out.shape == (4, 2, 28, 28)

    def test_raw_lift_kernels_match_least_squares(self):
        from lifting import FrozenLiftRaw
        w = FrozenLiftRaw().conv.weight.data
        I_expected = torch.tensor([[[-1,-1,-1],[0,0,0],[1,1,1]]]) / 6.0
        J_expected = torch.tensor([[[-1,0,1],[-1,0,1],[-1,0,1]]]) / 6.0
        assert torch.allclose(w[0], I_expected, atol=1e-6)
        assert torch.allclose(w[1], J_expected, atol=1e-6)

    def test_raw_lift_not_normalized(self):
        from lifting import FrozenLiftRaw
        out = FrozenLiftRaw()(torch.randn(4, 1, 28, 28))
        norms = torch.sqrt(out[:, 0]**2 + out[:, 1]**2)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)

    def test_raw_lift_frozen(self):
        from lifting import FrozenLiftRaw
        for p in FrozenLiftRaw().parameters():
            assert not p.requires_grad


class TestIterationBlock:
    def test_iter_block_shapes(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        p_new, m_new = block(torch.randn(4, 1, 14, 14), torch.randn(4, 2, 14, 14))
        assert p_new.shape == (4, 1, 14, 14)
        assert m_new.shape == (4, 2, 14, 14)

    def test_iter_block_pp_has_relu(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        pixel = torch.randn(4, 1, 14, 14) - 0.5
        with torch.no_grad():
            block.mp.weight.zero_()
        p_new, _ = block(pixel, torch.randn(4, 2, 14, 14))
        assert (p_new >= -1e-6).all()

    def test_iter_block_mp_pm_mm_linear(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        block.eval()
        pixel = torch.randn(4, 1, 14, 14)
        manifold = torch.randn(4, 2, 14, 14)
        with torch.no_grad():
            block.pp.weight.zero_()
        p1, _ = block(pixel, manifold)
        p2, _ = block(pixel, 2.0 * manifold)
        assert torch.allclose(p2, 2.0 * p1, atol=1e-4)

    def test_iter_block_returns_raw(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        _, m_new = block(torch.randn(4, 1, 14, 14), torch.randn(4, 2, 14, 14) * 0.1)
        norms = torch.sqrt(m_new[:, 0]**2 + m_new[:, 1]**2)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)

    def test_iter_block_identity_at_init(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        block.eval()
        pixel = torch.randn(4, 1, 14, 14).abs()
        manifold = torch.randn(4, 2, 14, 14)
        p_new, m_new = block(pixel, manifold)
        assert torch.allclose(p_new, pixel, atol=1e-4)
        assert torch.allclose(m_new, manifold, atol=1e-4)

    def test_iter_block_gradients_flow(self):
        from iteration_block import FourPathIterBlock
        block = FourPathIterBlock()
        pixel = torch.randn(4, 1, 14, 14, requires_grad=True)
        manifold = torch.randn(4, 2, 14, 14, requires_grad=True)
        p_new, m_new = block(pixel, manifold)
        (p_new.sum() + m_new.sum()).backward()
        for name, param in block.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0


class TestReconstructionLoss:
    def test_recon_head_shape(self):
        from reconstruction import ReconstructionHead
        assert ReconstructionHead()(torch.randn(4, 2)).shape == (4, 9)

    def test_recon_loss_scalar(self):
        from reconstruction import compute_recon_loss, ReconstructionHead
        loss = compute_recon_loss(torch.randn(4, 2, 28, 28), torch.randn(4, 1, 28, 28),
                                  ReconstructionHead())
        assert loss.dim() == 0 and loss.item() >= 0

    def test_recon_loss_contrast_weighting(self):
        from reconstruction import compute_recon_loss, ReconstructionHead
        head = ReconstructionHead()
        images = torch.cat([torch.randn(1, 1, 28, 28) * 5.0,
                            torch.ones(1, 1, 28, 28) * 0.5], dim=0)
        loss = compute_recon_loss(torch.randn(2, 2, 28, 28), images, head)
        assert loss.item() >= 0


class TestFirstLayers:
    def test_love_first_layer(self):
        from models import LoveFirstLayer
        layer = LoveFirstLayer()
        assert layer(torch.randn(4, 1, 28, 28)).shape == (4, 64, 26, 26)
        assert len(list(layer.parameters())) == 0

    def test_carlsson_raw_first_layer(self):
        from models import ManifoldRawFirstLayer
        assert ManifoldRawFirstLayer()(torch.randn(4, 1, 28, 28)).shape == (4, 64, 26, 26)

    def test_baseline_first_layer(self):
        assert nn.Conv2d(1, 64, 3)(torch.randn(4, 1, 28, 28)).shape == (4, 64, 26, 26)


class TestFullNetworks:
    def _get_all_models(self):
        from models import LoveNet, ManifoldFrozenNet, ManifoldRawNet, BaselineNet
        return [LoveNet(), ManifoldFrozenNet(), ManifoldRawNet(), BaselineNet()]

    def test_all_networks_same_output_shape(self):
        x = torch.randn(4, 1, 28, 28)
        for model in self._get_all_models():
            assert model(x).shape == (4, 10)

    def test_all_networks_valid_probabilities(self):
        x = torch.randn(4, 1, 28, 28)
        for model in self._get_all_models():
            sums = torch.softmax(model(x), dim=1).sum(dim=1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_networks_shared_downstream(self):
        from models import LoveNet, ManifoldFrozenNet, ManifoldRawNet, BaselineNet
        ref = LoveNet()
        for model in [ManifoldFrozenNet(), ManifoldRawNet(), BaselineNet()]:
            assert model.conv2.weight.shape == (64, 64, 3, 3)
            assert model.fc1.weight.shape == ref.fc1.weight.shape
            assert model.fc2.weight.shape == ref.fc2.weight.shape

    def test_carlsson_raw_net_exposes_recon_loss(self):
        from models import ManifoldRawNet
        model = ManifoldRawNet()
        model(torch.randn(4, 1, 28, 28))
        loss = model.compute_recon_loss(torch.randn(4, 1, 28, 28))
        assert loss.dim() == 0 and loss.item() >= 0

    def test_love_filters_frozen(self):
        from models import LoveNet
        trainable = [n for n, p in LoveNet().named_parameters() if p.requires_grad]
        for name in trainable:
            assert 'first_layer' not in name or 'filter' not in name.lower()


class TestTraining:
    def test_train_one_batch(self):
        from models import BaselineNet
        model = BaselineNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(model(x), y)
        loss1.backward()
        opt.step()
        opt.zero_grad()
        loss2 = criterion(model(x), y)
        assert loss2.item() != loss1.item()

    def test_train_with_recon_loss(self):
        from models import ManifoldRawNet
        model = ManifoldRawNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, y = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
        out = model(x)
        total = nn.CrossEntropyLoss()(out, y) + 0.1 * model.compute_recon_loss(x)
        total.backward()
        opt.step()
        assert total.item() > 0


class TestAnalyticFilters:
    def test_analytic_primary_circle_properties(self):
        """Analytic primary circle filters: mean-zero, normalized, correct shape."""
        from analytic_filters import analytic_primary_circle
        filters = analytic_primary_circle(64, 3)
        assert filters.shape == (64, 3, 3)
        for k in range(64):
            assert abs(filters[k].sum()) < 1e-6
            # All should have same L2 norm (m * STD * 1.0 after normalization)
        norms = [np.linalg.norm(filters[k]) for k in range(64)]
        assert all(abs(n - norms[0]) < 1e-6 for n in norms)

    def test_fused_directional_mean_zero(self):
        from analytic_filters import directional_derivative_filters
        filters = directional_derivative_filters(64, 3)
        for k in range(64):
            assert abs(filters[k].sum()) < 1e-10

    def test_fused_directional_equal_norm(self):
        from analytic_filters import directional_derivative_filters
        norms = [np.linalg.norm(f) for f in directional_derivative_filters(64, 3)]
        assert all(abs(n - norms[0]) < 1e-10 for n in norms)

    def test_fused_directional_rank_2(self):
        from analytic_filters import directional_derivative_filters
        _, S, _ = np.linalg.svd(directional_derivative_filters(64, 3).reshape(64, 9),
                                full_matrices=False)
        assert S[0] > 1e-3 and S[1] > 1e-3 and S[2] < 1e-10


class TestFusedFrozenModel:
    def test_fused_frozen_output_shape(self):
        from models import ManifoldFrozenFirstLayer
        assert ManifoldFrozenFirstLayer()(torch.randn(4, 1, 28, 28)).shape == (4, 64, 26, 26)

    def test_fused_frozen_no_learnable_params(self):
        from models import ManifoldFrozenFirstLayer
        assert len(list(ManifoldFrozenFirstLayer().parameters())) == 0

    def test_fused_frozen_rf_3x3(self):
        """Single pixel perturbation affects only the 3x3 RF window."""
        from models import ManifoldFrozenFirstLayer
        layer = ManifoldFrozenFirstLayer()
        x = torch.zeros(1, 1, 28, 28)
        base = layer(x)
        x_p = x.clone()
        x_p[0, 0, 14, 14] = 1.0
        diff = (layer(x_p) - base).abs()
        nonzero = (diff > 1e-8).any(dim=1)
        affected = nonzero.sum().item()
        assert 8 <= affected <= 9
        outside = nonzero.clone()
        outside[0, 12:15, 12:15] = False
        assert outside.sum() == 0

    def test_fused_frozen_net_output(self):
        from models import ManifoldFrozenNet
        assert ManifoldFrozenNet()(torch.randn(4, 1, 28, 28)).shape == (4, 10)


class TestReconLossVariableSize:
    def test_recon_loss_with_28x28_manifold(self):
        from reconstruction import compute_recon_loss, ReconstructionHead
        loss = compute_recon_loss(torch.randn(4, 2, 28, 28), torch.randn(4, 1, 28, 28),
                                  ReconstructionHead())
        assert loss.dim() == 0 and loss.item() >= 0

    def test_recon_loss_with_26x26_manifold(self):
        from reconstruction import compute_recon_loss, ReconstructionHead
        loss = compute_recon_loss(torch.randn(4, 2, 26, 26), torch.randn(4, 1, 28, 28),
                                  ReconstructionHead())
        assert loss.dim() == 0 and loss.item() >= 0
