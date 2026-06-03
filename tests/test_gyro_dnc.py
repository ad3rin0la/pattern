"""Phase-5G Gyro-DNC checks (tope.quantum.gyro_dnc).

The eight self-checks shipped with the module, plus gradient flow, as pytest
cases.  Loaded by file path to avoid the (environmental) torch_scatter import
that the top-level ``tope`` package pulls in.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GD_PATH = _REPO_ROOT / "tope" / "quantum" / "gyro_dnc.py"


def _load():
    if "gyro_dnc" in sys.modules:
        return sys.modules["gyro_dnc"]
    spec = importlib.util.spec_from_file_location("gyro_dnc", str(_GD_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gyro_dnc"] = mod
    spec.loader.exec_module(mod)
    return mod


gd = _load()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# (1) gyration isometry: preserves Euclidean norm, fixes 0 (in-ball probes)
def test_gyration_isometry_and_fixes_origin():
    ball = gd._Ball(c=1.0)
    a, b = torch.randn(1, 6) * 0.2, torch.randn(1, 6) * 0.2
    v = ball.proj(torch.randn(4, 6) * 0.15)
    gv = ball.gyration(a.expand_as(v), b.expand_as(v), v)
    assert torch.allclose(gv.norm(dim=-1), v.norm(dim=-1), atol=1e-4)
    assert ball.gyration(a, b, torch.zeros(1, 6)).abs().max() < 1e-5


# (2) gyration linearity (in-ball small vectors)
def test_gyration_linearity():
    ball = gd._Ball(c=1.0)
    a, b = torch.randn(1, 6) * 0.2, torch.randn(1, 6) * 0.2
    al, be = 0.7, -1.3
    w = torch.randn(4, 6) * 0.05
    v = torch.randn(4, 6) * 0.05
    lhs = ball.gyration(a.expand_as(v), b.expand_as(v), al * v + be * w)
    rhs = (al * ball.gyration(a.expand_as(v), b.expand_as(v), v)
           + be * ball.gyration(a.expand_as(w), b.expand_as(w), w))
    assert torch.allclose(lhs, rhs, atol=1e-4)


# (3) Einstein midpoint validity: midpoint of equal points = that point
def test_gyrocentroid_idempotent_and_in_ball():
    ball = gd._Ball(c=1.0)
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    pt = ball.proj(torch.randn(1, 6) * 0.4).expand(8, 6).contiguous()
    mem.M = pt.clone()
    wts = torch.softmax(torch.randn(8), -1)
    mid = mem._einstein_midpoint(mem.M, wts)
    assert torch.allclose(mid, pt[0], atol=1e-4)
    assert mid.norm() < 1.0


# (4) write convergence: repeated pure writes (e=0) drive a slot to content
def test_write_convergence():
    ball = gd._Ball(c=1.0)
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    target = ball.proj(torch.randn(6) * 0.5)
    w_one_hot = torch.zeros(8)
    w_one_hot[3] = 0.6
    for _ in range(40):
        mem.write(target, w_one_hot, erase_gate=0.0)
    assert ball.dist(mem.M[3].unsqueeze(0), target.unsqueeze(0)).item() < 1e-2


# (5) content addressing concentrates on the matching slot
def test_content_addressing():
    ball = gd._Ball(c=1.0)
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    target = ball.proj(torch.randn(6) * 0.5)
    w_one_hot = torch.zeros(8)
    w_one_hot[3] = 0.6
    for _ in range(40):
        mem.write(target, w_one_hot, erase_gate=0.0)
    wr = mem.read_weights(target, beta=20.0)
    assert wr.argmax().item() == 3


# (6) allocation prefers low-radius slots
def test_radial_allocation_avoids_committed():
    ball = gd._Ball(c=1.0)
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    mem.reset_memory()
    mem.M[5] = ball.proj(torch.randn(6) * 0.9)  # commit slot 5 (high radius)
    alloc = mem.allocate(sharpness=4.0)
    assert alloc[5] == alloc.min()


# (7) frustration discrimination + base-point invariance
def test_frustration_discrimination_and_basepoint_invariance():
    ball = gd._Ball(c=1.0)
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    # gyro-collinear loop (points along one axis through origin) -> ~0
    line = torch.zeros(8, 6)
    line[:, 0] = torch.linspace(-0.6, 0.6, 8)
    mem.M = ball.proj(line)
    f_line = mem.frust_index([0, 3, 7])
    # generic triangle -> > 0.  Slots placed at a controlled interior radius
    # (random directions, fixed norm) so the frame-probe stays in the linear
    # regime ‖Hol−I‖_F conjugation-invariance requires; ``randn*0.5`` would push
    # some slots to the boundary, the degenerate regime the docstring flags.
    v = torch.randn(8, 6)
    v = v / v.norm(dim=-1, keepdim=True) * 0.5
    mem.M = ball.proj(v)
    f_tri = mem.frust_index([0, 3, 7])
    f_tri_rot = mem.frust_index([3, 7, 0])  # cyclic relabel
    assert f_line.item() < 1e-3
    assert f_tri.item() > 1e-2
    assert abs(f_tri.item() - f_tri_rot.item()) < 1e-4


# (8) Gauss-Bonnet: single-gyration angle = |K| * area, quadratic in scale
def test_gauss_bonnet_area_law_scaling():
    ball = gd._Ball(c=1.0)

    def _gyr_angle(av, bv):
        W = av.numel()
        Iw = torch.eye(W)
        tt = 1e-3
        fr = ball.gyration(av.expand(W, W), bv.expand(W, W), tt * Iw)
        s = ((fr / tt - Iw).norm() / (2 * math.sqrt(2))).clamp(0.0, 1.0 - 1e-7)
        return float(2 * torch.asin(s))

    ea = torch.tensor([0.2, 0., 0, 0, 0, 0])
    eb = torch.tensor([0., 0.2, 0, 0, 0, 0])
    th1, th2 = _gyr_angle(ea, eb), _gyr_angle(ea / 2, eb / 2)
    ratio = th1 / max(th2, 1e-12)
    assert 3.8 < ratio < 4.2          # halving edges → ~quarter angle (area law)


# (+) gradient flow through read + frustration
def test_gradient_flow_through_read_and_frustration():
    mem = gd.GyroMemory(n_slots=8, mem_dim=6, c=1.0)
    mem.reset_memory()
    mem.M = mem.M.clone().requires_grad_(True)
    loss = mem.read(torch.softmax(torch.randn(8), -1)).sum() + mem.frust_index([0, 2, 5])
    loss.backward()
    assert mem.M.grad is not None and torch.isfinite(mem.M.grad).all()
