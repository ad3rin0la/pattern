import torch
import importlib.util
import sys
from pathlib import Path

_CC_ATTENTION_PATH = Path(__file__).resolve().parents[1] / "tope" / "models" / "cc_attention.py"
_SPEC = importlib.util.spec_from_file_location("cc_attention_under_test", _CC_ATTENTION_PATH)
cc_attention = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = cc_attention
_SPEC.loader.exec_module(cc_attention)

(
    CCAttentionBlock,
    FermionicCCANOConfig,
    FermionicCCAttentionNeuralOperator,
    cell_intersection_size,
    exterior_permutation_sign,
    gyrobarycentric_aggregate,
    wedge_pair,
) = (
    cc_attention.CCAttentionBlock,
    cc_attention.FermionicCCANOConfig,
    cc_attention.FermionicCCAttentionNeuralOperator,
    cc_attention.cell_intersection_size,
    cc_attention.exterior_permutation_sign,
    cc_attention.gyrobarycentric_aggregate,
    cc_attention.wedge_pair,
)


def test_cc_attention_block_forward_with_geometry_package_present():
    torch.manual_seed(0)
    block = CCAttentionBlock(d_s_in=4, d_t_in=5, d_out=6, n_heads=2)
    source_features = torch.randn(3, 4)
    target_features = torch.randn(2, 5)
    incidence = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])

    target_update, source_update = block(source_features, target_features, incidence)

    assert target_update.shape == (2, 6)
    assert source_update.shape == (3, 6)


def test_wedge_pair_is_antisymmetric():
    u = torch.tensor([1.0, 2.0, -1.0])
    v = torch.tensor([0.5, -3.0, 4.0])

    assert torch.allclose(wedge_pair(u, v), -wedge_pair(v, u))
    assert torch.allclose(wedge_pair(u, u), torch.zeros(3, 3))


def test_exterior_permutation_sign_handles_swaps_and_duplicates():
    src = torch.tensor([[0, 2, -1], [1, -1, -1], [0, 1, -1]])
    tgt = torch.tensor([[1, -1], [0, 2], [1, 3]])

    signs = exterior_permutation_sign(src, tgt)

    assert torch.equal(signs, torch.tensor([-1.0, -1.0, 0.0]))


def test_cell_intersection_size_from_membership_mask():
    masks = torch.tensor(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])

    assert torch.equal(cell_intersection_size(masks, edge_index), torch.tensor([1.0, 1.0, 1.0]))


def test_gyrobarycentric_aggregate_stays_inside_ball():
    values = torch.tensor(
        [
            [0.2, 0.0],
            [0.0, 0.3],
            [0.1, 0.1],
        ]
    )
    weights = torch.tensor([0.25, 0.75, 1.0])
    index = torch.tensor([0, 0, 1])

    out = gyrobarycentric_aggregate(values, weights, index, dim_size=2, curvature=1.0)

    assert out.shape == (2, 2)
    assert torch.all(out.norm(dim=-1) < 1.0)


def test_fermionic_ccano_forward_and_attention_normalization():
    torch.manual_seed(0)
    cfg = FermionicCCANOConfig(d_in=10, d_out=12, n_heads=3, n_ranks=4, dropout=0.0)
    layer = FermionicCCAttentionNeuralOperator(cfg)
    features = torch.randn(5, 10)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 4],
            [1, 1, 3, 3, 4, 4],
        ]
    )
    ranks = torch.tensor([0, 1, 1, 2, 2])
    masks = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    ordered_cells = torch.tensor(
        [
            [0, -1],
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
        ]
    )

    result = layer(features, edge_index, ranks, masks, ordered_cells, return_attention=True)

    assert result["output"].shape == (5, 12)
    assert result["dT_dt"].shape == (5, 12)
    assert result["attention"].shape == (6, 3)
    for target in edge_index[1].unique():
        incoming = edge_index[1] == target
        assert torch.allclose(
            result["attention"][incoming].sum(dim=0),
            torch.ones(3),
            atol=1e-6,
        )
    assert "fermionic_sign" in result
