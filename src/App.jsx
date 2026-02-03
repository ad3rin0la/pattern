import { useState, useEffect } from "react";

const styles = `
  * { box-sizing: border-box; margin: 0; padding: 0; }

  .root {
    font-family: 'Inter', sans-serif;
    background: #0a0c0f;
    color: #d4d0cb;
    min-height: 100vh;
    overflow-x: hidden;
  }

  .grain-overlay {
    position: fixed; inset: 0; z-index: 100; pointer-events: none;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    opacity: 0.6;
  }

  /* HEADER */
  .header {
    position: relative;
    padding: 80px 40px 60px;
    border-bottom: 1px solid rgba(180,160,120,0.12);
    background: linear-gradient(180deg, #0f1014 0%, #0a0c0f 100%);
  }
  .header-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #c9a84c, #8fb8d4, transparent);
  }
  .header-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: #8fb8d4; margin-bottom: 24px; font-weight: 300;
  }
  .header h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 400; line-height: 1.2;
    color: #e8e2d6; max-width: 680px;
  }
  .header h1 em { font-style: italic; color: #c9a84c; }
  .header-sub {
    margin-top: 18px; font-size: 14px; color: #7a756e;
    max-width: 600px; line-height: 1.6; font-weight: 300;
  }
  .header-meta {
    margin-top: 28px; display: flex; gap: 24px; flex-wrap: wrap;
  }
  .meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
    color: #5a756e; border: 1px solid rgba(143,184,212,0.2);
    padding: 5px 10px; border-radius: 2px;
  }

  /* NAV TABS */
  .nav-strip {
    display: flex; gap: 2px; padding: 0 40px;
    background: #0d0f12; border-bottom: 1px solid rgba(180,160,120,0.08);
    position: sticky; top: 0; z-index: 50;
  }
  .nav-btn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 1px; text-transform: uppercase;
    padding: 14px 18px; border: none; background: transparent;
    color: #6a6560; cursor: pointer; transition: all 0.2s;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
  }
  .nav-btn:hover { color: #a8a298; }
  .nav-btn.active { color: #c9a84c; border-bottom-color: #c9a84c; }

  /* MAIN CONTENT */
  .content { max-width: 860px; margin: 0 auto; padding: 60px 40px 100px; }

  /* SECTION HEADERS */
  .section-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #c9a84c; letter-spacing: 2px;
    margin-bottom: 8px; opacity: 0.7;
  }
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px; font-weight: 400; color: #e8e2d6;
    margin-bottom: 6px; line-height: 1.3;
  }
  .section-title em { font-style: italic; color: #8fb8d4; }
  .section-divider {
    width: 40px; height: 1px; background: linear-gradient(90deg, #c9a84c, transparent);
    margin-bottom: 28px;
  }

  /* PROSE */
  .prose { font-size: 14px; line-height: 1.8; color: #9a9590; font-weight: 300; }
  .prose p { margin-bottom: 18px; }
  .prose strong { color: #d4d0cb; font-weight: 500; }
  .prose code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; background: rgba(143,184,212,0.08);
    padding: 2px 7px; border-radius: 3px; color: #8fb8d4;
  }

  /* CALLOUT BOX */
  .callout {
    border-left: 2px solid #c9a84c;
    background: rgba(201,168,76,0.04);
    padding: 18px 22px; margin: 24px 0; border-radius: 0 4px 4px 0;
  }
  .callout-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
    color: #c9a84c; margin-bottom: 6px;
  }
  .callout p { font-size: 13px; color: #a8a298; line-height: 1.7; margin-bottom: 0; }

  /* BRIDGE CARD */
  .bridge-card {
    background: linear-gradient(135deg, rgba(143,184,212,0.06), rgba(201,168,76,0.04));
    border: 1px solid rgba(143,184,212,0.15);
    border-radius: 6px; padding: 24px 28px; margin: 28px 0;
  }
  .bridge-card .bridge-title {
    font-family: 'Playfair Display', serif; font-size: 16px;
    color: #8fb8d4; margin-bottom: 10px;
  }
  .bridge-card p { font-size: 13px; color: #8a857e; line-height: 1.7; margin-bottom: 8px; }

  /* COMPARISON TABLE */
  .cmp-table { width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 13px; }
  .cmp-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
    color: #c9a84c; text-align: left; padding: 10px 14px;
    border-bottom: 1px solid rgba(201,168,76,0.3);
    background: rgba(201,168,76,0.04);
  }
  .cmp-table td {
    padding: 11px 14px; border-bottom: 1px solid rgba(180,160,120,0.08);
    color: #9a9590; vertical-align: top;
  }
  .cmp-table tr:last-child td { border-bottom: none; }
  .cmp-table .col-ioffe { color: #c9a84c; font-weight: 500; }
  .cmp-table .col-topo { color: #8fb8d4; font-weight: 500; }

  /* PIPELINE VISUAL */
  .pipeline { display: flex; align-items: stretch; gap: 0; margin: 28px 0; flex-wrap: wrap; }
  .pipe-step {
    flex: 1; min-width: 130px; padding: 18px 16px;
    background: rgba(10,12,15,0.8); border: 1px solid rgba(180,160,120,0.1);
    position: relative; text-align: center;
  }
  .pipe-step:not(:last-child)::after {
    content: '→'; position: absolute; right: -12px; top: 50%;
    transform: translateY(-50%); color: #c9a84c; font-size: 14px; z-index: 2;
  }
  .pipe-step .pipe-num {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: #c9a84c; letter-spacing: 1px; margin-bottom: 6px;
  }
  .pipe-step .pipe-label { font-size: 12px; color: #d4d0cb; font-weight: 500; line-height: 1.3; }
  .pipe-step .pipe-desc { font-size: 10px; color: #6a6560; margin-top: 4px; line-height: 1.4; }

  /* CRITERIA LIST */
  .criteria-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 20px 0; }
  .criteria-item {
    background: rgba(10,12,15,0.6); border: 1px solid rgba(180,160,120,0.1);
    border-radius: 4px; padding: 14px 16px; cursor: pointer; transition: all 0.2s;
  }
  .criteria-item:hover { border-color: rgba(143,184,212,0.3); background: rgba(143,184,212,0.04); }
  .criteria-item.active { border-color: #8fb8d4; background: rgba(143,184,212,0.08); }
  .criteria-item .ci-label { font-size: 13px; color: #d4d0cb; font-weight: 500; margin-bottom: 4px; }
  .criteria-item .ci-desc { font-size: 11px; color: #6a6560; line-height: 1.5; }
  .criteria-item .ci-source {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: #c9a84c; margin-top: 6px; letter-spacing: 0.5px;
  }

  /* TOPOLOGY DIAGRAM (SVG) */
  .topo-vis { margin: 28px 0; }

  /* MATH BLOCK */
  .math-block {
    background: rgba(143,184,212,0.05); border: 1px solid rgba(143,184,212,0.12);
    border-radius: 4px; padding: 16px 20px; margin: 18px 0; font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: #8fb8d4; overflow-x: auto; white-space: pre;
  }

  /* CHECKLIST */
  .checklist { list-style: none; margin: 18px 0; }
  .checklist li { padding: 8px 0; border-bottom: 1px solid rgba(180,160,120,0.06); font-size: 13px; color: #9a9590; display: flex; align-items: flex-start; gap: 10px; }
  .checklist li:last-child { border-bottom: none; }
  .check-icon { width: 18px; height: 18px; border-radius: 3px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; }
  .check-done { background: rgba(201,168,76,0.2); color: #c9a84c; }
  .check-todo { background: rgba(143,184,212,0.1); color: #8fb8d4; border: 1px solid rgba(143,184,212,0.25); }
  .checklist .item-title { color: #d4d0cb; font-weight: 500; margin-bottom: 2px; }
  .checklist .item-note { font-size: 11px; color: #5f5a53; }

  /* FOOTER */
  .footer {
    border-top: 1px solid rgba(180,160,120,0.1);
    padding: 40px; text-align: center;
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    letter-spacing: 1.5px; color: #3d3b38; text-transform: uppercase;
  }

  /* PANEL FADE */
  .panel { animation: panelIn 0.35s ease; }
  @keyframes panelIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
`;

const TABS = ["Foundation", "Topology Bridge", "Model Design", "Roadmap"];

// ─── SECTION COMPONENTS ───

function FoundationSection() {
  const [selected, setSelected] = useState(null);
  const parallels = [
    { ioffe: "Potential Functions classifier over physicochemical criteria space", topo: "Persistent Laplacian spectral filtration over atomic point clouds", theme: "Multiscale classification" },
    { ioffe: "Sliding-window criterion ranking (inverse recognition)", topo: "Importance / gradient attribution in TCPNet message passing", theme: "Feature influence ranking" },
    { ioffe: "Additive averaging of orbital parameters for multi-component systems", topo: "SE(3)-equivariant aggregation across hierarchical PCC levels", theme: "Multi-level representation" },
    { ioffe: "Boolean conjunctions in Prognoz-73 (logic-vector classification)", topo: "Combinatorial complex cell interactions (0-/1-/2-cells)", theme: "Discrete structural encoding" },
    { ioffe: "Classifying power αcl vs predictive power αpr separation", topo: "Train / validation / test F-score decomposition (TopEC: 0.72 F)", theme: "Generalization metrics" },
    { ioffe: "Quantitative activity prediction (rate, selectivity %)", topo: "Multi-task regression: log(kcat), Km, kcat/Km (CataPro R²=0.67)", theme: "Kinetic prediction" },
  ];
  return (
    <div className="panel">
      <div className="section-number">01 — FOUNDATION</div>
      <div className="section-title">What Ioffe got right — <em>and what's new</em></div>
      <div className="section-divider" />
      <div className="prose">
        <p>Ioffe, Dobrotvorskii & Belozerskikh published a remarkably prescient framework: use <strong>pattern recognition algorithms</strong> trained on tabulated physicochemical properties to <strong>predict catalyst behaviour</strong> and <strong>infer mechanistic features</strong> (the "inverse recognition problem"). The core insight — that catalytic activity emerges from <strong>combinations</strong> of electronic and geometric properties, not any single descriptor — remains foundational.</p>
        <p>The table below maps Ioffe's original algorithmic choices onto their modern topological equivalents. The conceptual structure is the same; the mathematics has deepened enormously.</p>
      </div>

      <table className="cmp-table">
        <thead>
          <tr>
            <th style={{width:'28%'}}>Theme</th>
            <th style={{width:'36%'}}>Ioffe et al. 1983</th>
            <th style={{width:'36%'}}>Topological DL (2024–25)</th>
          </tr>
        </thead>
        <tbody>
          {parallels.map((r, i) => (
            <tr key={i}>
              <td style={{color:'#d4d0cb', fontWeight:500}}>{r.theme}</td>
              <td className="col-ioffe">{r.ioffe}</td>
              <td className="col-topo">{r.topo}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="callout">
        <div className="callout-label">Key Insight from Ioffe</div>
        <p>The <strong>inverse recognition problem</strong> — ranking which physicochemical properties most strongly influence catalytic selectivity — is structurally identical to modern <strong>feature attribution</strong> in topological deep learning. Ioffe's "sliding recognition" algorithm (omit one criterion, re-classify, measure drop) is a discrete precursor to gradient-based saliency.</p>
      </div>

      <div className="prose">
        <p>Ioffe identified eight influential properties for CO oxidation from an initial set of twenty, and only four of those eight were individually correlated with activity. The remaining four were only "visible" when considered as a <strong>combination</strong>. This is precisely the regime where topological methods excel — they capture <strong>higher-order interactions</strong> that pairwise correlations miss.</p>
        <p>The modern kinetics prediction landscape reinforces this insight. CataPro achieves <strong>R² = 0.67 (kcat)</strong> and <strong>R² = 0.73 (Km)</strong> using sequence + structure features, while DeepEnzyme maintains <strong>R² = 0.42</strong> at &lt;50% sequence identity — exactly the out-of-distribution regime where topological active-site encoding should provide a genuine edge over sequence-only methods.</p>
      </div>
    </div>
  );
}

function TopologyBridgeSection() {
  const [activeLevel, setActiveLevel] = useState(0);
  const levels = [
    { label: "0-Cells (Atoms)", desc: "Individual catalytic-residue atoms become vertices. Ioffe's atomic orbital parameters (VOIP, quantum numbers) map onto per-atom features.", color: "#c9a84c" },
    { label: "1-Cells (Bonds)", desc: "Covalent and coordination bonds become edges. Captures the metal–oxygen bond ionicity Ioffe used, plus H-bond networks in the active site.", color: "#8fb8d4" },
    { label: "2-Cells (Faces)", desc: "Triangular residue clusters form faces. Encodes the 'geometric correspondence' Balandin principle that Ioffe confirmed via inverse recognition.", color: "#a8c9a0" },
    { label: "3-Cells (Volumes)", desc: "Tetrahedral cavities — the pore/channel topology. Maps onto Ioffe's 'ratio of crystallographic hole volume to metal atom volume'.", color: "#d4a08a" },
  ];

  return (
    <div className="panel">
      <div className="section-number">02 — TOPOLOGY BRIDGE</div>
      <div className="section-title">From criterion space to <em>simplicial complexes</em></div>
      <div className="section-divider" />

      <div className="prose">
        <p>Ioffe worked in an <strong>N-dimensional criterion hypercube</strong> — each axis one physicochemical property. Topotein's <strong>Protein Combinatorial Complex (PCC)</strong> does something structurally deeper: it builds a <strong>hierarchical simplicial complex</strong> from the protein itself, then filters it across distance thresholds (a <em>filtration</em>). Persistent Laplacians track how the topology of this complex evolves — which is exactly the "multi-level correlation" Ioffe wanted but couldn't compute.</p>
      </div>

      {/* Interactive cell-level selector */}
      <div style={{ display:'flex', gap: 8, margin: '24px 0 12px', flexWrap:'wrap' }}>
        {levels.map((l, i) => (
          <button key={i} onClick={() => setActiveLevel(i)} style={{
            fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: '7px 14px',
            background: activeLevel === i ? `${l.color}18` : 'rgba(10,12,15,0.6)',
            border: `1px solid ${activeLevel === i ? l.color : 'rgba(180,160,120,0.12)'}`,
            color: activeLevel === i ? l.color : '#6a6560', borderRadius: 3, cursor:'pointer',
            transition: 'all 0.2s', letterSpacing: '0.5px',
          }}>{l.label}</button>
        ))}
      </div>
      <div className="bridge-card" style={{ borderColor: `rgba(${levels[activeLevel].color.slice(1).match(/.{2}/g).map(h=>parseInt(h,16)).join(',')},0.25)` }}>
        <div className="bridge-title" style={{ color: levels[activeLevel].color }}>{levels[activeLevel].label}</div>
        <p>{levels[activeLevel].desc}</p>
      </div>

      {/* Filtration diagram */}
      <div className="prose"><p style={{marginBottom:8}}><strong>How filtration works for enzyme active sites:</strong></p></div>
      <div className="pipeline">
        <div className="pipe-step">
          <div className="pipe-num">ε = 2Å</div>
          <div className="pipe-label">Covalent skeleton</div>
          <div className="pipe-desc">Bonds only</div>
        </div>
        <div className="pipe-step">
          <div className="pipe-num">ε = 4Å</div>
          <div className="pipe-label">H-bond network</div>
          <div className="pipe-desc">Catalytic triad forms</div>
        </div>
        <div className="pipe-step">
          <div className="pipe-num">ε = 6Å</div>
          <div className="pipe-label">Active-site pocket</div>
          <div className="pipe-desc">Substrate cavity</div>
        </div>
        <div className="pipe-step">
          <div className="pipe-num">ε = 8Å</div>
          <div className="pipe-label">Allosteric shell</div>
          <div className="pipe-desc">Distal residues</div>
        </div>
      </div>

      <div className="callout">
        <div className="callout-label">Why Persistent Laplacians ≫ Persistent Homology here</div>
        <p>Persistent homology captures <strong>when</strong> topological features (loops, voids) appear and disappear. Persistent Laplacians additionally encode <strong>how quickly</strong> information propagates across those features via non-harmonic spectra. For enzyme catalysis, the <em>rate</em> of electron / proton transfer across the active-site network is the quantity of interest — that's a spectral, not purely topological, signal.</p>
      </div>

      <div className="prose">
        <p>The <strong>persistent sheaf Laplacian (PSL)</strong>, demonstrated in 2025 on protein B-factor prediction (32% improvement over Gaussian Network Models), is particularly promising: it fuses <strong>geometric and non-geometric</strong> labels at each atom, letting you attach Ioffe-style electronic descriptors (ionisation potential, d-electron count) directly onto the topological scaffold without losing structural information.</p>
      </div>
    </div>
  );
}

function ModelDesignSection() {
  const [expandedCrit, setExpandedCrit] = useState(null);
  const criteria = [
    { id: 0, label: "Persistent Spectral Features", desc: "Eigenvalues of the q-th persistent Laplacian at each filtration radius. The harmonic kernel recovers Betti numbers; non-harmonic spectra encode shape/rate information.", source: "Wei & Wei 2025 (PTL survey)" },
    { id: 1, label: "Sheaf-Augmented Orbital Params", desc: "VOIP, electron affinity, d-electron count attached as sheaf sections on each atom. The PSL discriminates atom types while preserving connectivity topology.", source: "Hayes et al. 2025 (PSL); Ioffe 1983 (VOIP criteria)" },
    { id: 2, label: "SE(3)-Equivariant Cell Messages", desc: "Messages passed across 0→1→2-cells with rotation-translation invariance. Inherits TCPNet's architecture but operates on enzyme active-site complexes.", source: "Topotein (Wang et al. 2025)" },
    { id: 3, label: "Catalytic-Residue Attention Mask", desc: "Learned attention weights biased toward residues in catalytic site annotations (M-CSA / CSA). Mirrors Ioffe's criterion-influence ranking as a soft mask.", source: "TopEC (van der Weg et al. 2025)" },
    { id: 4, label: "Substrate-Product Correspondence", desc: "Ioffe showed selectivity depends on reactant–product property alignment. Encode this as a bipartite graph between substrate topology and product topology, fed into a cross-attention layer.", source: "Ioffe 1983 (Table 6); GraphEC 2024" },
    { id: 5, label: "Kinetics Regression Head", desc: "Dedicated regression heads for log(kcat), log(Km), and log(kcat/Km). Trained jointly with EC classification using BRENDA/SABIO-RK kinetics data (~23k kcat, ~41k Km entries). Benchmarked against CataPro (R²=0.67) and DeepEnzyme (R²=0.42 at <50% identity).", source: "CataPro (Nat Commun 2025); DeepEnzyme (Brief Bioinform 2024)" },
  ];

  return (
    <div className="panel">
      <div className="section-number">03 — MODEL DESIGN</div>
      <div className="section-title">Architectural blueprint for <em>ToPE</em></div>
      <div className="section-divider" />
      <div className="prose">
        <p>Below is a proposed architecture — <strong>Topological Pattern Recognition for Enzymes (ToPE)</strong> — that directly synthesises Ioffe's criterion-space framework with Topotein's combinatorial complex machinery and the persistent Laplacian toolkit. Click each feature block to expand its rationale.</p>
      </div>

      {/* Pipeline */}
      <div className="pipeline">
        <div className="pipe-step" style={{background:'rgba(201,168,76,0.06)', borderColor:'rgba(201,168,76,0.2)'}}>
          <div className="pipe-num">INPUT</div>
          <div className="pipe-label">Enzyme PDB + Substrate</div>
          <div className="pipe-desc">Active-site extraction</div>
        </div>
        <div className="pipe-step" style={{background:'rgba(143,184,212,0.06)', borderColor:'rgba(143,184,212,0.2)'}}>
          <div className="pipe-num">ENCODE</div>
          <div className="pipe-label">Enzyme Comb. Complex</div>
          <div className="pipe-desc">Filtration + PSL</div>
        </div>
        <div className="pipe-step" style={{background:'rgba(168,201,160,0.06)', borderColor:'rgba(168,201,160,0.2)'}}>
          <div className="pipe-num">PROPAGATE</div>
          <div className="pipe-label">TCPNet Layers</div>
          <div className="pipe-desc">SE(3) message pass</div>
        </div>
        <div className="pipe-step" style={{background:'rgba(212,160,138,0.06)', borderColor:'rgba(212,160,138,0.2)'}}>
          <div className="pipe-num">PREDICT</div>
          <div className="pipe-label">EC + kcat/Km + Sel.</div>
          <div className="pipe-desc">Multi-task + attribution</div>
        </div>
      </div>

      <div className="prose"><p style={{marginBottom:12}}><strong>Feature modules — click to expand:</strong></p></div>
      <div className="criteria-grid">
        {criteria.map((c) => (
          <div key={c.id} className={`criteria-item ${expandedCrit === c.id ? 'active' : ''}`} onClick={() => setExpandedCrit(expandedCrit === c.id ? null : c.id)}>
            <div className="ci-label">{c.label}</div>
            {expandedCrit === c.id && (
              <>
                <div className="ci-desc">{c.desc}</div>
                <div className="ci-source">← {c.source}</div>
              </>
            )}
          </div>
        ))}
      </div>

      <div className="prose" style={{marginTop:28}}>
        <p><strong>Inverse attribution (the "inverse ToPE" problem):</strong> After training, freeze the network and run <code>grad-CAM</code> or integrated-gradients over the persistent spectral features. The resulting saliency map across filtration radii and cell dimensions directly tells you <em>which topological scale</em> and <em>which interaction type</em> drives selectivity — a modernised version of Ioffe's sliding-recognition criterion ranking.</p>
      </div>

      <div className="math-block">{`# Pseudo-loss: multi-task classification + kinetics + attribution
L = L_CE(ŷ, y_EC)                         # EC-class cross-entropy
  + λ₁ · L_sel(ŷ_sel, y_selectivity)       # selectivity regression
  + λ₂ · L_kin(ŷ_kcat, y_log_kcat)         # log(kcat) MSE
  + λ₃ · L_kin(ŷ_Km, y_log_Km)             # log(Km) MSE
  + μ  · KL( attr(ε*) ‖ prior_CSA )        # attribution should align
                                            # with known catalytic sites`}</div>
    </div>
  );
}

function RoadmapSection() {
  const phases = [
    { label: "Phase 1", title: "Baseline & Data", done: true, items: [
      { t: "Curate enzyme active-site dataset from M-CSA + PDB (\u22655,000 structures)", d: "Training sample assembly \u2014 mirrors Ioffe\u2019s data-bank step" },
      { t: "Integrate BRENDA/SABIO-RK kinetics (~23k kcat, ~41k Km entries)", d: "Quantitative activity labels for regression heads" },
      { t: "Implement persistent Laplacian filtration pipeline (gudhi / giotto-tda)", d: "Core topological encoding" },
      { t: "Benchmark against TopEC, GraphEC (EC), CataPro/DeepEnzyme (kcat)", d: "Establish performance floor across classification + kinetics" },
    ]},
    { label: "Phase 2", title: "Topological Encoding", done: true, items: [
      { t: "Build Enzyme Combinatorial Complex (Enzyme-PCC)", d: "Adapt PCC from Topotein to catalytic active sites" },
      { t: "Attach sheaf sections (VOIP, d-electrons, electronegativity) via PSL", d: "Fuses Ioffe electronic descriptors into topology" },
      { t: "Train persistent spectral feature extractor", d: "Learn filtration-radius weighting" },
    ]},
    { label: "Phase 3", title: "Full ToPE Model", done: false, items: [
      { t: "Implement TCPNet-style SE(3) message passing over Enzyme-PCC", d: "" },
      { t: "Add substrate\u2013product bipartite cross-attention", d: "Ioffe\u2019s selectivity correspondence" },
      { t: "Train multi-task on EC (F-score) + selectivity + log(kcat/Km) jointly", d: "Unified classification + kinetics regression" },
    ]},
    { label: "Phase 4", title: "Inverse Attribution & Validation", done: false, items: [
      { t: "Derive saliency maps over (filtration radius, cell dimension)", d: "Which topological scale matters" },
      { t: "Validate kinetics on out-of-distribution enzymes (<40% seq identity)", d: "DeepEnzyme holds R\u00B2=0.42 at <50%; ToPE targets R\u00B2>0.50 at <40%" },
      { t: "Compare ranked features to Ioffe\u2019s 8 influential CO oxidation properties", d: "Close the loop" },
      { t: "Experimental validation: predict kcat for 10 unseen enzyme variants", d: "Wet-lab confirmation of topological predictions" },
    ]},
  ];

  return (
    <div className="panel">
      <div className="section-number">04 — ROADMAP</div>
      <div className="section-title">Development <em>phases</em></div>
      <div className="section-divider" />
      {phases.map((phase, pi) => (
        <div key={pi} style={{ marginBottom: 28 }}>
          <div style={{ display:'flex', alignItems:'center', gap: 10, marginBottom: 12 }}>
            <div style={{
              fontFamily:"'JetBrains Mono', monospace", fontSize: 9, letterSpacing: 2,
              color: phase.done ? '#c9a84c' : '#8fb8d4', textTransform:'uppercase',
            }}>{phase.label}</div>
            <div style={{ fontSize: 14, color:'#d4d0cb', fontFamily:"'Playfair Display', serif" }}>{phase.title}</div>
            {phase.done && <span style={{ fontSize: 9, color:'#c9a84c', fontFamily:"'JetBrains Mono', monospace", letterSpacing:1 }}>◆ READY</span>}
          </div>
          <ul className="checklist">
            {phase.items.map((item, ii) => (
              <li key={ii}>
                <div className={`check-icon ${phase.done ? 'check-done' : 'check-todo'}`}>{phase.done ? '✓' : '○'}</div>
                <div>
                  <div className="item-title">{item.t}</div>
                  {item.d && <div className="item-note">{item.d}</div>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}

      <div className="callout" style={{ marginTop: 36 }}>
        <div className="callout-label">Validation Strategy & Performance Targets</div>
        <p>Ioffe demanded ~80–85% prediction accuracy to be "economically effective." For enzyme function, TopEC achieves F = 0.72 across 800+ EC classes. For kinetics, CataPro reaches <strong>R² = 0.67 (kcat)</strong> and <strong>R² = 0.73 (Km)</strong>; DeepEnzyme maintains <strong>R² = 0.42</strong> at &lt;50% sequence identity. ToPE targets: <strong>(1)</strong> EC F-score &gt; 0.75, <strong>(2)</strong> selectivity MAE &lt; 12%, <strong>(3)</strong> kcat R² &gt; 0.50 at &lt;40% identity where topological active-site encoding should provide a genuine edge.</p>
      </div>

      <div className="prose" style={{marginTop:24}}>
        <p><strong>Key references to track:</strong> Topotein (arXiv 2509.03885), TopEC (Nat Commun 2025), Persistent Sheaf Laplacian (J Phys Chem B 2025), DeepEnzyme (Brief Bioinform 2024), CataPro (Nat Commun 2025), GraphKcat (bioRxiv 2025), KcatNet (bioRxiv 2025), CatPred (Nat Commun 2025).</p>
      </div>
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState(0);
  const panels = [<FoundationSection />, <TopologyBridgeSection />, <ModelDesignSection />, <RoadmapSection />];

  return (
    <>
      <style>{styles}</style>
      <div className="root">
        <div className="grain-overlay" />

        {/* Header */}
        <div className="header">
          <div className="header-accent" />
          <div className="header-label">Technical Synthesis · Enzyme Catalysis</div>
          <h1>Topological Pattern Recognition<br />for <em>Enzyme Catalysis</em></h1>
          <div className="header-sub">
            Bridging Ioffe's 1983 criterion-space framework with Topotein, persistent Laplacians, and modern topological deep learning — toward a unified model for predicting and explaining enzymatic selectivity.
          </div>
          <div className="header-meta">
            <span className="meta-tag">Ioffe et al. 1983</span>
            <span className="meta-tag">Topotein · arXiv 2509.03885</span>
            <span className="meta-tag">Persistent Laplacians</span>
            <span className="meta-tag">BRENDA / SABIO-RK</span>
            <span className="meta-tag">Enzyme Catalysis</span>
          </div>
        </div>

        {/* Nav */}
        <div className="nav-strip">
          {TABS.map((t, i) => (
            <button key={i} className={`nav-btn ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>

        {/* Content */}
        <div className="content">
          {panels[tab]}
        </div>

        <div className="footer">Topological Pattern Recognition for Enzymes · Technical Synthesis · 2025</div>
      </div>
    </>
  );
}
