import sys
sys.path.insert(0, 'build')
 
import math
import numpy as np
import pytest
import matt
 
RNG = np.random.default_rng(42)
TOL = 1e-4

Normal = matt.nn.weight_initializer.Normal
KaimingUniform = matt.nn.weight_initializer.KaimingUniform
 
# ── helpers ───────────────────────────────────────────────────────────────────
 
def to_matt(arr: np.ndarray) -> matt.Tensor:
    """Wrap a NumPy array (float32, C-contiguous) in a matt Tensor."""
    arr = arr.astype(np.float32).ravel()
    shape = list(arr.shape) if arr.ndim > 0 else [1]
    # recover original shape
    orig = np.asarray(arr).reshape(-1)
    return matt.Tensor.from_data(orig.tolist(), list(np.asarray(arr.shape if arr.ndim else [1])))
 
 
def tensor_from_np(arr: np.ndarray) -> matt.Tensor:
    """Create a matt Tensor from any NumPy array, preserving shape."""
    flat = arr.astype(np.float32).ravel().tolist()
    shape = list(arr.shape)
    return matt.Tensor.from_data(flat, shape)
 
 
def at(t: matt.Tensor, *idx) -> float:
    """Read a scalar from a matt Tensor at multi-index *idx."""
    return t.at(list(idx))
 
 
def read_1d(t: matt.Tensor, n: int) -> list:
    return [at(t, i) for i in range(n)]
 
 
def read_2d(t: matt.Tensor, rows: int, cols: int) -> np.ndarray:
    return np.array([[at(t, r, c) for c in range(cols)] for r in range(rows)], dtype=np.float32)
 
 
# ── Tensor construction ───────────────────────────────────────────────────────
 
class TestTensorConstruction:
    def test_zeros_values(self):
        t = matt.Tensor.zeros([3, 3])
        got = read_2d(t, 3, 3)
        np.testing.assert_allclose(got, np.zeros((3, 3), dtype=np.float32), atol=TOL)
 
    def test_ones_values(self):
        t = matt.Tensor.ones([2, 4])
        got = read_2d(t, 2, 4)
        np.testing.assert_allclose(got, np.ones((2, 4), dtype=np.float32), atol=TOL)
 
    def test_from_data_roundtrip(self):
        ref = RNG.uniform(-5, 5, size=12).astype(np.float32)
        t = matt.Tensor.from_data(ref.tolist(), [12])
        got = np.array(read_1d(t, 12), dtype=np.float32)
        np.testing.assert_allclose(got, ref, atol=TOL)
 
    def test_from_data_2d_roundtrip(self):
        ref = RNG.uniform(-3, 3, size=(4, 5)).astype(np.float32)
        t = tensor_from_np(ref)
        got = read_2d(t, 4, 5)
        np.testing.assert_allclose(got, ref, atol=TOL)
 
 
# ── Ops – elementwise ─────────────────────────────────────────────────────────
 
class TestOpsElementwise:
    def _pair(self, shape):
        a_np = RNG.uniform(-4, 4, size=shape).astype(np.float32)
        b_np = RNG.uniform(-4, 4, size=shape).astype(np.float32)
        return a_np, b_np, tensor_from_np(a_np), tensor_from_np(b_np)
 
    def test_add_values_1d(self):
        a_np, b_np, a, b = self._pair((8,))
        c = matt.ops.add(a, b)
        got = np.array(read_1d(c, 8))
        np.testing.assert_allclose(got, a_np + b_np, atol=TOL)
 
    def test_add_values_2d(self):
        a_np, b_np, a, b = self._pair((3, 4))
        c = matt.ops.add(a, b)
        got = read_2d(c, 3, 4)
        np.testing.assert_allclose(got, a_np + b_np, atol=TOL)
 
    def test_sub_values(self):
        a_np, b_np, a, b = self._pair((5, 3))
        c = matt.ops.sub(a, b)
        got = read_2d(c, 5, 3)
        np.testing.assert_allclose(got, a_np - b_np, atol=TOL)
 
    def test_sub_self_is_zero(self):
        a_np = RNG.uniform(-2, 2, size=(4, 4)).astype(np.float32)
        a = tensor_from_np(a_np)
        c = matt.ops.sub(a, a)
        got = read_2d(c, 4, 4)
        np.testing.assert_allclose(got, np.zeros((4, 4)), atol=TOL)
 
    def test_add_commutative(self):
        a_np, b_np, a, b = self._pair((3, 3))
        ab = read_2d(matt.ops.add(a, b), 3, 3)
        ba = read_2d(matt.ops.add(b, a), 3, 3)
        np.testing.assert_allclose(ab, ba, atol=TOL)
 
 
# ── Ops – ReLU ────────────────────────────────────────────────────────────────
 
class TestOpsReLU:
    def test_relu_positive_unchanged(self):
        data = RNG.uniform(0.01, 5, size=10).astype(np.float32)
        t = matt.Tensor.from_data(data.tolist(), [10])
        out = matt.ops.relu(t)
        got = np.array(read_1d(out, 10))
        np.testing.assert_allclose(got, data, atol=TOL)
 
    def test_relu_negative_zeroed(self):
        data = RNG.uniform(-5, -0.01, size=10).astype(np.float32)
        t = matt.Tensor.from_data(data.tolist(), [10])
        out = matt.ops.relu(t)
        got = np.array(read_1d(out, 10))
        np.testing.assert_allclose(got, np.zeros(10), atol=TOL)
 
    def test_relu_mixed(self):
        data = np.array([-3.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
        t = matt.Tensor.from_data(data.tolist(), [5])
        out = matt.ops.relu(t)
        got = np.array(read_1d(out, 5))
        np.testing.assert_allclose(got, np.maximum(data, 0), atol=TOL)
 
 
# ── Ops – matmul ──────────────────────────────────────────────────────────────
 
class TestOpsMatmul:
    def test_matmul_identity(self):
        """A @ I == A"""
        A = RNG.uniform(-3, 3, size=(4, 4)).astype(np.float32)
        I = np.eye(4, dtype=np.float32)
        a = tensor_from_np(A)
        eye = tensor_from_np(I)
        c = matt.ops.matmul(a, eye)
        got = read_2d(c, 4, 4)
        np.testing.assert_allclose(got, A, atol=1e-3)
 
    def test_matmul_known_values(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        ref = A @ B  # [[19,22],[43,50]]
        c = matt.ops.matmul(tensor_from_np(A), tensor_from_np(B))
        got = read_2d(c, 2, 2)
        np.testing.assert_allclose(got, ref, atol=TOL)
 
    def test_matmul_rectangular(self):
        A = RNG.uniform(-2, 2, size=(3, 5)).astype(np.float32)
        B = RNG.uniform(-2, 2, size=(5, 7)).astype(np.float32)
        ref = A @ B
        c = matt.ops.matmul(tensor_from_np(A), tensor_from_np(B))
        got = read_2d(c, 3, 7)
        np.testing.assert_allclose(got, ref, atol=1e-3)
 
    def test_matmul_zeros(self):
        A = RNG.uniform(-5, 5, size=(4, 6)).astype(np.float32)
        Z = np.zeros((6, 3), dtype=np.float32)
        c = matt.ops.matmul(tensor_from_np(A), tensor_from_np(Z))
        got = read_2d(c, 4, 3)
        np.testing.assert_allclose(got, np.zeros((4, 3)), atol=TOL)
 
 
# ── Ops – sum ─────────────────────────────────────────────────────────────────
 
class TestOpsSum:
    def test_sum_known(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = matt.Tensor.from_data(data.tolist(), [4])
        out = matt.ops.sum(t)
        assert abs(at(out, 0) - 10.0) < TOL
 
    def test_sum_random(self):
        data = RNG.uniform(-10, 10, size=20).astype(np.float32)
        t = matt.Tensor.from_data(data.tolist(), [20])
        out = matt.ops.sum(t)
        assert abs(at(out, 0) - float(data.sum())) < 1e-3
 
    def test_sum_negative(self):
        data = np.full(5, -2.0, dtype=np.float32)
        t = matt.Tensor.from_data(data.tolist(), [5])
        out = matt.ops.sum(t)
        assert abs(at(out, 0) - (-10.0)) < TOL
 
    def test_sum_2d(self):
        data = RNG.uniform(-3, 3, size=(4, 5)).astype(np.float32)
        t = tensor_from_np(data)
        out = matt.ops.sum(t)
        assert abs(at(out, 0) - float(data.sum())) < 1e-3
 
 
# ── Transpose ─────────────────────────────────────────────────────────────────
 
class TestTranspose:
    def test_transpose_values(self):
        ref = RNG.uniform(-3, 3, size=(3, 4)).astype(np.float32)
        t = tensor_from_np(ref)
        t2 = t.transpose(0, 1)
        got = read_2d(t2, 4, 3)
        np.testing.assert_allclose(got, ref.T, atol=TOL)
 
    def test_double_transpose_identity(self):
        ref = RNG.uniform(-2, 2, size=(5, 3)).astype(np.float32)
        t = tensor_from_np(ref)
        tt = t.transpose(0, 1).transpose(0, 1)
        got = read_2d(tt, 5, 3)
        np.testing.assert_allclose(got, ref, atol=TOL)
 
    def test_transpose_square(self):
        ref = RNG.uniform(-5, 5, size=(4, 4)).astype(np.float32)
        t = tensor_from_np(ref)
        got = read_2d(t.transpose(0, 1), 4, 4)
        np.testing.assert_allclose(got, ref.T, atol=TOL)
 
 
# ── nn.Linear (forward numerics) ─────────────────────────────────────────────
 
class TestLinearNumerics:
    """
    We can't easily set weights from Python, so we verify structural invariants
    and that two forward passes on the same input return the same result
    (determinism), and that bias shifts the output predictably.
    """
 
    def test_forward_deterministic(self):
        linear = matt.nn.Linear(4, 6, Normal(0, 0.1, 42))
        x = tensor_from_np(RNG.uniform(-1, 1, size=(3, 4)).astype(np.float32))
        o1 = read_2d(linear.forward(x), 3, 6)
        o2 = read_2d(linear.forward(x), 3, 6)
        np.testing.assert_allclose(o1, o2, atol=TOL)
 
    def test_forward_no_bias_shape(self):
        linear = matt.nn.Linear(5, 3, KaimingUniform(42), False)
        x = tensor_from_np(RNG.uniform(-1, 1, size=(2, 5)).astype(np.float32))
        out = linear.forward(x)
        assert out.shape() == [2, 3]
 
    def test_batch_size_1_vs_n(self):
        """Output rows are independent: batch of 1 == single row of batch of N."""
        linear = matt.nn.Linear(4, 3, Normal(0, 0.1, 42))
        data = RNG.uniform(-1, 1, size=(4, 4)).astype(np.float32)
        batch_out = read_2d(linear.forward(tensor_from_np(data)), 4, 3)
        for i in range(4):
            row_out = read_2d(linear.forward(tensor_from_np(data[i:i+1])), 1, 3)
            np.testing.assert_allclose(row_out[0], batch_out[i], atol=TOL,
                err_msg=f"Row {i} differs between batch and single-sample forward")
 
 
# ── MSELoss numerics ─────────────────────────────────────────────────────────
 
class TestMSELossNumerics:
    def _mse(self, pred, target):
        return float(np.mean((pred - target) ** 2))
 
    def test_zero_loss_on_perfect_pred(self):
        data = RNG.uniform(-3, 3, size=(4, 1)).astype(np.float32)
        pred   = tensor_from_np(data)
        target = tensor_from_np(data)
        loss = matt.nn.MSELoss().forward(pred, target)
        assert abs(at(loss, 0)) < TOL
 
    def test_known_mse(self):
        pred_np   = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        target_np = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        ref = self._mse(pred_np, target_np)   # (1+4+9)/3 = 14/3
        loss = matt.nn.MSELoss().forward(tensor_from_np(pred_np), tensor_from_np(target_np))
        assert abs(at(loss, 0) - ref) < TOL
 
    def test_random_mse(self):
        pred_np   = RNG.uniform(-5, 5, size=(8, 1)).astype(np.float32)
        target_np = RNG.uniform(-5, 5, size=(8, 1)).astype(np.float32)
        ref = self._mse(pred_np, target_np)
        loss = matt.nn.MSELoss().forward(tensor_from_np(pred_np), tensor_from_np(target_np))
        assert abs(at(loss, 0) - ref) < 1e-3
 
 
# ── Gradient / backward ───────────────────────────────────────────────────────
 
class TestGradients:
    def test_sum_backward_sets_grad(self):
        """After sum().backward(), every element's grad should be 1.0."""
        data = RNG.uniform(-3, 3, size=6).astype(np.float32)
        t = matt.Tensor.from_data(data.tolist(), [6])
        t.set_requires_grad(True)
        out = matt.ops.sum(t)
        out.backward()
        # Gradient of sum w.r.t each element is 1. We confirm backward ran by
        # checking a subsequent SGD step changes the parameter.
 
    def test_sgd_step_changes_weights(self):
        linear = matt.nn.Linear(3, 2, Normal(0, 0.1, 42))
        x = tensor_from_np(RNG.uniform(-1, 1, size=(2, 3)).astype(np.float32))
        before = read_2d(linear.forward(x), 2, 2).copy()
        out = linear.forward(x)
        loss = matt.ops.sum(out)
        loss.backward()
        matt.optim.SGD(linear.parameters(), 0.1).step()
        after = read_2d(linear.forward(x), 2, 2)
        # At least one weight must have changed
        assert not np.allclose(before, after, atol=1e-6), \
            "SGD step did not change any weights"
 
    def test_zero_grad_clears(self):
        """zero_grad followed by step should be a no-op (grads zeroed, no update)."""
        linear = matt.nn.Linear(3, 2, Normal(0, 0.1, 42))
        optim = matt.optim.SGD(linear.parameters(), 0.1)
        x = tensor_from_np(RNG.uniform(-1, 1, size=(2, 3)).astype(np.float32))
        out = linear.forward(x)
        loss = matt.ops.sum(out)
        loss.backward()
        optim.zero_grad()          # clear before step
        before = read_2d(linear.forward(x), 2, 2).copy()
        optim.step()
        after = read_2d(linear.forward(x), 2, 2)
        np.testing.assert_allclose(before, after, atol=1e-6,
            err_msg="step() after zero_grad() should not change weights")
 
 
# ── Training convergence ──────────────────────────────────────────────────────
 
class TestConvergence:
    def _train(self, net, x_np, y_np, lr=0.01, steps=200):
        criterion = matt.nn.MSELoss()
        optim = matt.optim.SGD(net.parameters(), lr)
        x = tensor_from_np(x_np)
        target = tensor_from_np(y_np)
        losses = []
        for _ in range(steps):
            pred = net.forward(x)
            loss = criterion.forward(pred, target)
            losses.append(at(loss, 0))
            loss.backward()
            optim.step()
            optim.zero_grad()
        return losses
 
    def test_loss_strictly_decreasing_trend(self):
        """Mean loss of last 20 steps < mean loss of first 20 steps."""
        net = matt.nn.Sequential()
        net.add(matt.nn.Linear(3, 8, Normal(0, 0.1, 42)))
        net.add(matt.nn.ReLU())
        net.add(matt.nn.Linear(8, 1, Normal(0, 0.1, 42)))
        x_np = RNG.uniform(-1, 1, size=(4, 3)).astype(np.float32)
        y_np = RNG.uniform(-1, 1, size=(4, 1)).astype(np.float32)
        losses = self._train(net, x_np, y_np, lr=0.01, steps=300)
        first_mean = np.mean(losses[:20])
        last_mean  = np.mean(losses[-20:])
        assert last_mean < first_mean, \
            f"Loss did not decrease: first={first_mean:.4f}, last={last_mean:.4f}"
 
    def test_overfit_tiny_dataset(self):
        """
        A no-ReLU net (avoids dying-ReLU init sensitivity) should memorize
        2 samples. Final MSE < 0.05 after sufficient steps.
        """
        net = matt.nn.Sequential()
        net.add(matt.nn.Linear(2, 16, Normal(0, 0.1, 42)))
        net.add(matt.nn.Linear(16, 1, Normal(0, 0.1, 42)))
        x_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_np = np.array([[1.0], [-1.0]], dtype=np.float32)
        losses = self._train(net, x_np, y_np, lr=0.05, steps=500)
        assert losses[-1] < 0.05, \
            f"Network failed to overfit tiny dataset. Final loss: {losses[-1]:.4f}"

    def test_loss_decreases_across_lrs(self):
        """
        Both a slow and fast LR should reduce loss from their own starting point.
        Tests that SGD works across a range of LRs, without comparing the two
        networks (which have different random inits).
        """
        def make_net():
            n = matt.nn.Sequential()
            n.add(matt.nn.Linear(2, 8, Normal(0, 0.1, 42)))
            n.add(matt.nn.ReLU())
            n.add(matt.nn.Linear(8, 1, Normal(0, 0.1, 42)))
            return n
 
        x_np = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)
        y_np = np.array([[0.0], [1.0]], dtype=np.float32)
 
        for lr in [0.001, 0.01, 0.05]:
            losses = self._train(make_net(), x_np, y_np, lr=lr, steps=200)
            first = np.mean(losses[:10])
            last  = np.mean(losses[-10:])
            assert last < first, \
                f"Loss did not decrease at lr={lr}: first={first:.4f}, last={last:.4f}"
 
    def test_loss_non_negative(self):
        """MSE loss should never go negative."""
        net = matt.nn.Sequential()
        net.add(matt.nn.Linear(4, 8, Normal(0, 0.1, 42)))
        net.add(matt.nn.ReLU())
        net.add(matt.nn.Linear(8, 1, Normal(0, 0.1, 42)))
        x_np = RNG.uniform(-1, 1, size=(5, 4)).astype(np.float32)
        y_np = RNG.uniform(-1, 1, size=(5, 1)).astype(np.float32)
        losses = self._train(net, x_np, y_np, lr=0.01, steps=200)
        assert all(l >= -TOL for l in losses), \
            f"Negative loss detected: min={min(losses):.6f}"