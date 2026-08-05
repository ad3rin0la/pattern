# Gyro-CC Layer — design memo

**Audience.** Whoever decides what ships in `phonon_topology.py` after the HSH commit.
**Status.** Decision pending. §4 (gyration kernel), §5 (tangent-space sheaf Laplacian), and §3
(adjoint-consistent attention) are landed as **parallel paths** that do not touch the
just-merged HSH integration or the dirty in-flight work in `cc_attention.py`,
`cross_attention.py`, `phonon_topology.py`, `ttn_persistent_homology.py`.

## The collision in one paragraph

Last commit (`e339d27`, *Add HSH restriction maps for ToPE sheaf Laplacian — Phase 2*) replaced
the rank-1 Cerrini outer product `R_ij = (Gδ)(Gδ)ᵀ / (δᵀGδ)` with a 36-element HSH expansion in
`Sym(R⁸)`. The new spec (§5 of *Gyro-Combinatorial-Complex Layer for ToPE*) replaces the
restriction map with a fundamentally different object: `F_{x◁y} = K_{xy} · R_{xy}`, where
`K_{xy}` is a VOIP-informed stiffness (SheafENM Hessian block, possibly PSD) and `R_{xy} =
gyr[h_y, −h_x]` is a Möbius gyration acting as parallel transport in the Poincaré ball. The
two designs target the same code path. They are not additive — three coherent stances exist.

## Three options

### Option A — Replace HSH with §5 (`F = K · R`)

- **Pros.** §5 is more principled: it makes parallel transport explicit via `R`, makes the
  measure explicit via `γ^{2d}`, and gives a closed conceptual story (FrustIndex = holonomy of
  `R` around closed loops). The Laplacian `L_r = δ_r^* δ_r` is unconditionally PSD without
  needing `∂² = 0` on the full Hodge tower.
- **Cons.** Discards a freshly-merged module with 15 passing tests. The HSH basis was
  motivated as the *orbital-anisotropy descriptor* — a physics interpretation that doesn't
  immediately map onto the §5 stiffness block. We'd lose the e_g/t_2g coupling story unless
  we re-express it inside `K`.

### Option B — Compose: HSH inside K, §5 supplies R

- **Pros.** Best of both: HSH gives a learnable, physics-grounded form for the stiffness block
  `K_{xy}` (with the 36-element basis providing orbital-anisotropy structure), §5 provides the
  geometrically-correct transport `R_{xy}` that was implicitly identity in the HSH path. The
  restriction map becomes `F_{x◁y} = K_{xy}^{HSH} · R_{xy}^{gyro}` — physics in `K`, geometry
  in `R`. The §5 Laplacian formula `L_r[x,x'] = μ(h_x)^{-1} Σ_y [y:x][y:x'] μ(h_y) F^T F` would
  use this composite `F`.
- **Cons.** Needs a coherence check: HSH was derived for `Sym(R^8)` (self-adjoint maps);
  composing with an orthogonal `R` produces a generally-non-symmetric `F`. That is fine for
  `δ_r^* δ_r` (which only needs `F^T F`), but the HSH "irreducible decomposition" argument
  no longer applies to `F` itself — only to `K`. The interpretation of HSH weights moves from
  *restriction-map basis* to *stiffness-block basis*. Probably correct, but worth a re-derive.
- **My recommendation: this one, conditional on the re-derive checking out.**

### Option C — Parallel, independent heads

- **Pros.** Lowest-risk: keep HSH where it is, add §5 as a sibling spectral head (which is
  what the current parallel-path implementation does). Lets you run both, A/B them on
  benchmarks before committing.
- **Cons.** Two code paths to maintain. Duplicated tests. Likely a transitional state, not
  a long-term resting point.

## My recommendation

**Land §4 unconditionally.** The closed-form gyration is needed by both the §3 attention path
and any future §5 work. It's self-contained, ~120 LoC, and verified to spec's <2e-13 numerical
tolerance.

**Land §3 unconditionally.** The anisotropic-with-`G_ij` attention block is a strict
generalization of the current isotropic `JacobianCorrectedBlock` — it's gated by a new class,
the existing block stays unchanged. No collision.

**On §5: choose Option B in a follow-up.** Before then, the parallel `TangentSheafLaplacian`
class lets the spectral head be benchmarked side-by-side with the HSH `SheafENM`. The re-derive
to confirm HSH-in-K composes coherently with gyro-R should be done before merging.

## Open mathematical questions to settle before §5 ships

1. **`F = K·R` and the §5 Laplacian formula.** The spec gives `L_r[x,x'] = μ(h_x)^{-1} Σ
   [y:x][y:x'] μ(h_y) F_{x◁y}^T F_{x'◁y}`. With `F = K·R`, this becomes `R^T K^T K R'` — note
   the *different* `R` and `R'` because they're the gyrations from different source cells. The
   diagonal case `x = x'` simplifies to `R^T K^T K R` which is PSD; the off-diagonal case has
   no such symmetry. Confirm the resulting block matrix is self-adjoint w.r.t. the γ-weighted
   inner product.

2. **Adjoint operator naming.** Spec writes `δ_r^* = D_r^{-1} δ_r^T D_{r+1}` for the
   *cochain* coboundary and `M^* = D_s^{-1} M^T D_t` for the *attention* push-forward. These
   are the same formula on different bipartite graphs. The §5 `L_r = δ_r^* δ_r` and the §3
   adjoint-consistency condition are literally the same object at different ranks. This is
   worth surfacing in the code via a shared `weighted_adjoint(M, D_src, D_tgt)` helper.

3. **Hodge-tower vs degree-0 Laplacian.** The spec correctly distinguishes the
   unconditional `δ_r^* δ_r` from the `δ_{r-1} δ_{r-1}^* + δ_r^* δ_r` Hodge form (which
   requires flat sheaves, i.e. trivial holonomy). The §5 *degree-0* Laplacian is what's
   implemented. The full Hodge form, and its FrustIndex interpretation, is a separate
   (future) module.

4. **`gyro_memory.py` naming.** The spec puts the gyration kernel and FrustIndex in
   `gyro_memory.py`. I followed the spec's filename but the contents are pure geometry, no
   "memory" — consider renaming to `gyro.py` or `mobius.py` for clarity.

## Files added by this work (parallel paths)

| File | Lines | Purpose |
|---|---|---|
| `tope/topology/gyro_memory.py` | ~280 | §4 closed-form gyration + Möbius primitives + gyrobary + gyrodistance |
| `tope/topology/tangent_sheaf.py` | ~260 | §5 `L_r = δ_r^* δ_r` over `F = K·R` restriction maps, γ^{2d} measure |
| `tope/models/gyro_cc_attention.py` | ~250 | §3 `GyroAdjointAttentionBlock` with anisotropic logit + gyration frame correction + Ferreira Jacobian |
| `tests/test_gyro_cc_layer.py` | ~350 | Property tests across all three |
| `docs/gyro_cc_layer_memo.md` | this file | Decision context |

**Files NOT touched (would be touched under Option A or B):**

- `tope/topology/phonon_topology.py` — has the HSH integration plus pre-existing unstaged TLS/Hirshfeld work
- `tope/models/cc_attention.py` — has pre-existing unstaged work
- `tope/models/cross_attention.py`, `cross_attention_improved.py` — likewise
- `tope/topology/ttn_persistent_homology.py` — likewise

## Action items for whoever picks this up

- [ ] Decide A / B / C on the HSH↔§5 question.
- [ ] If B: do the re-derive of `L_r` with `F = K^{HSH} · R^{gyro}` and confirm self-adjointness.
- [ ] Decide whether to land §3 in-place (replace `JacobianCorrectedBlock`) or as the parallel
      `GyroAdjointAttentionBlock`. If in-place, coordinate with the cc_attention.py WIP.
- [ ] Rename `gyro_memory.py` if the "memory" framing isn't load-bearing.
- [ ] FrustIndex (holonomy of `R` around closed CC loops) is unimplemented — needs a loop
      enumerator on the CC plus the holonomy product. Defer to a separate task.
