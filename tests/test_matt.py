import sys
sys.path.insert(0, 'build')
import matt
import pytest

Normal = matt.nn.weight_initializer.Normal
KaimingUniform = matt.nn.weight_initializer.KaimingUniform

# ── Tensor ────────────────────────────────────────────────────────────────────

def test_tensor_zeros_shape():
    t = matt.Tensor.zeros([2, 3])
    assert t.shape() == [2, 3]
    assert t.numel() == 6
    assert t.ndim() == 2

def test_tensor_ones_shape():
    t = matt.Tensor.ones([4])
    assert t.shape() == [4]
    assert t.numel() == 4

def test_tensor_from_data():
    t = matt.Tensor.from_data([1.0, 2.0, 3.0], [3])
    assert t.shape() == [3]
    assert t.numel() == 3

def test_tensor_requires_grad():
    t = matt.Tensor.zeros([2, 2])
    assert not t.requires_grad()
    t.set_requires_grad(True)
    assert t.requires_grad()

def test_tensor_transpose():
    t = matt.Tensor.zeros([2, 3])
    t2 = t.transpose(0, 1)
    assert t2.shape() == [3, 2]

# ── Ops ───────────────────────────────────────────────────────────────────────

def test_ops_add():
    a = matt.Tensor.ones([2, 2])
    b = matt.Tensor.ones([2, 2])
    c = matt.ops.add(a, b)
    assert c.shape() == [2, 2]

def test_ops_sub():
    a = matt.Tensor.ones([2, 2])
    b = matt.Tensor.ones([2, 2])
    c = matt.ops.sub(a, b)
    assert c.shape() == [2, 2]

def test_ops_matmul():
    a = matt.Tensor.ones([2, 3])
    b = matt.Tensor.ones([3, 4])
    c = matt.ops.matmul(a, b)
    assert c.shape() == [2, 4]

def test_ops_relu():
    t = matt.Tensor.ones([3])
    out = matt.ops.relu(t)
    assert out.shape() == [3]

def test_ops_sum():
    t = matt.Tensor.ones([3])
    out = matt.ops.sum(t)
    assert out.numel() == 1

# ── nn.Linear ─────────────────────────────────────────────────────────────────

def test_linear_parameter_count():
    linear = matt.nn.Linear(4, 8, Normal(0, 0.1, 42))
    assert len(linear.parameters()) == 2

def test_linear_no_bias_parameter_count():
    linear = matt.nn.Linear(4, 8, KaimingUniform(42), False)
    assert len(linear.parameters()) == 1

def test_linear_forward_shape():
    linear = matt.nn.Linear(4, 8, Normal(0, 0.1, 42))
    x = matt.Tensor.ones([2, 4])
    out = linear.forward(x)
    assert out.shape() == [2, 8]

def test_linear_training_mode():
    linear = matt.nn.Linear(4, 8, Normal(0, 0.1, 42))
    assert linear.is_training()
    linear.train(False)
    assert not linear.is_training()
    linear.train()
    assert linear.is_training()

# ── nn.Sequential ─────────────────────────────────────────────────────────────

def test_sequential_forward_shape():
    net = matt.nn.Sequential()
    net.add(matt.nn.Linear(4, 8, Normal(0, 0.1, 42)))
    net.add(matt.nn.ReLU())
    net.add(matt.nn.Linear(8, 2, Normal(0, 0.1, 42)))
    x = matt.Tensor.ones([3, 4])
    out = net.forward(x)
    assert out.shape() == [3, 2]

def test_sequential_parameter_count():
    net = matt.nn.Sequential()
    net.add(matt.nn.Linear(4, 8, Normal(0, 0.1, 42)))
    net.add(matt.nn.ReLU())
    net.add(matt.nn.Linear(8, 2, Normal(0, 0.1, 42)))
    # 2 params per Linear x 2 = 4
    assert len(net.parameters()) == 4

def test_sequential_training_mode_propagates():
    net = matt.nn.Sequential()
    net.add(matt.nn.Linear(2, 4, Normal(0, 0.1, 42)))
    net.add(matt.nn.ReLU())
    assert net.is_training()
    net.train(False)
    assert not net.is_training()

# ── nn.MSELoss ────────────────────────────────────────────────────────────────

def test_mseloss_output_shape():
    criterion = matt.nn.MSELoss()
    pred   = matt.Tensor.ones([2, 1])
    target = matt.Tensor.zeros([2, 1])
    loss   = criterion.forward(pred, target)
    assert loss.numel() == 1

# ── optim.SGD ─────────────────────────────────────────────────────────────────

def test_sgd_step_runs():
    linear = matt.nn.Linear(2, 1, Normal(0, 0.1, 42))
    optim  = matt.optim.SGD(linear.parameters(), 0.01)
    x      = matt.Tensor.ones([1, 2])
    out    = linear.forward(x)
    loss   = matt.ops.sum(out)
    loss.backward()
    optim.step()
    optim.zero_grad()

# ── End-to-end ────────────────────────────────────────────────────────────────

def test_e2e_loss_decreases():
    net = matt.nn.Sequential()
    net.add(matt.nn.Linear(2, 4, Normal(0, 0.1, 42)))
    net.add(matt.nn.ReLU())
    net.add(matt.nn.Linear(4, 1, Normal(0, 0.1, 42)))

    optim     = matt.optim.SGD(net.parameters(), 0.01)
    criterion = matt.nn.MSELoss()
    x         = matt.Tensor.from_data([1.0, 2.0], [1, 2])
    target    = matt.Tensor.from_data([0.5], [1, 1])

    first_loss = None
    last_loss  = None

    for i in range(100):
        pred = net.forward(x)
        loss = criterion.forward(pred, target)
        if i == 0:
            first_loss = loss.at([0])
        last_loss = loss.at([0])
        loss.backward()
        optim.step()
        optim.zero_grad()

    assert last_loss < first_loss, [first_loss, last_loss]
    assert last_loss >= 0.0, last_loss
    assert first_loss >= 0.0, first_loss