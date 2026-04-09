import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.pipeline import InferencePipeline
from inference.postprocess import generate_heatmap, calculate_severity_score

import matplotlib.pyplot as plt
import seaborn as sns
import io

# ──────────────────────────────────────────────
# Page config & global CSS
# ──────────────────────────────────────────────
st.set_page_config(page_title="Composite Damage Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #0a0d14;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1219;
    border-right: 1px solid #1e2535;
}

/* ── Metric card ── */
.metric-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid #2a3349;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: #6c7a99;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-sub {
    font-size: 0.75rem;
    color: #4ade80;
    margin-top: 2px;
}

/* ── Step card ── */
.step-card {
    background: #111623;
    border: 1px solid #1e2d42;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', monospace;
}
.step-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.step-math {
    font-size: 0.8rem;
    color: #cbd5e1;
    line-height: 1.8;
    white-space: pre-wrap;
}
.step-stat {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid #1e2a3a;
}

/* ── Severity score ── */
.severity-number {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f97316, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section header ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2535;
}

/* ── Iteration badge ── */
.iter-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 6px;
}

/* ── Upload button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def tensor_stats(t: torch.Tensor, label: str) -> str:
    """Return a compact stat string for any tensor."""
    t_f = t.float()
    return (
        f"shape={list(t.shape)}  "
        f"min={t_f.min().item():.4f}  "
        f"max={t_f.max().item():.4f}  "
        f"mean={t_f.mean().item():.4f}  "
        f"std={t_f.std().item():.4f}"
    )

def step_card(title: str, math: str, stats: str, border_color: str = "#3b82f6"):
    """Render a processing step card with math description and tensor stats."""
    st.markdown(f"""
    <div class="step-card" style="border-left-color:{border_color};">
        <div class="step-title">{title}</div>
        <div class="step-math">{math}</div>
        <div class="step-stat">{stats}</div>
    </div>
    """, unsafe_allow_html=True)

def fig_to_img(fig):
    """Convert matplotlib fig to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=100)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return InferencePipeline("configs/config.yaml", "checkpoints/best_model.pth")

pipeline   = load_model()
material_map = {"CFRP": 0, "GFRP": 1, "Hybrid": 2}

# ──────────────────────────────────────────────
# App header
# ──────────────────────────────────────────────
st.markdown("# Multimodal AI — Internal Damage Predictor")
st.markdown("<p style='color:#64748b;margin-top:-10px;'>Vision Transformer × Point-Net × Metadata MLP → Transformer Fusion → C-Scan Decoder</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Dataset"])

# ══════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════
with tab1:

    # ── Left column: inputs ──────────────────
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-header">Data Input</div>', unsafe_allow_html=True)
        img_file = st.file_uploader("Surface Dent Image (JPG / PNG)", type=['jpg', 'jpeg', 'png'])
        pc_file  = st.file_uploader("3D Point Cloud (.ply / .npy)",   type=['ply', 'npy'])

        st.markdown('<div class="section-header">Metadata Parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            thickness    = st.text_input("Thickness (mm)",  value="5.0")
            dent_depth   = st.number_input("Dent Depth (mm)",   min_value=0.0, value=2.0)
            damage_area  = st.number_input("Damage Area (cm²)", min_value=0.0, value=15.0)
        with c2:
            material = st.selectbox("Material Type", ["CFRP", "GFRP", "Hybrid"])
            layup    = st.text_input("Layup Sequence", "[0/90/45/-45]s")

        run_btn = st.button("Predict C-Scan Damage", use_container_width=True)

    # ── Right column: outputs ────────────────
    with col_out:
        st.markdown('<div class="section-header">Analysis Output</div>', unsafe_allow_html=True)

        if run_btn:
            if not img_file:
                st.error("Please upload a surface image first.")
            else:
                try:
                    with st.spinner("Running pipeline…"):

                        # ── Pre-process inputs ───────────────────────────────────────────
                        img = Image.open(img_file)
                        pc_path = None
                        if pc_file is not None:
                            pc_ext = os.path.splitext(pc_file.name)[1]
                            with tempfile.NamedTemporaryFile(delete=False, suffix=pc_ext) as tmp:
                                tmp.write(pc_file.getvalue())
                                pc_path = tmp.name

                        try:
                            t_val = float(thickness)
                        except Exception:
                            t_val = 5.0

                        metadata = {
                            "thickness":              t_val,
                            "dent_depth":             dent_depth,
                            "damage_area":            damage_area,
                            "material_type_encoded":  material_map[material],
                            "layup_sequence_encoded": len(layup) / 10.0
                        }

                        # ── Run pipeline with intermediate captures ──────────────────────
                        # 1. Image pre-processing
                        img_tensor  = pipeline.preprocess_image(img)
                        # 2. Point cloud pre-processing
                        pc_tensor   = pipeline.preprocess_pointcloud(pc_path)
                        # 3. Metadata pre-processing
                        meta_tensor = pipeline.preprocess_metadata(metadata)

                        model = pipeline.model
                        model.eval()

                        with torch.no_grad():
                            # 4. Encode each modality
                            if hasattr(model.img_enc, 'forward_with_features'):
                                img_feat, spatial_features = model.img_enc.forward_with_features(img_tensor)
                            else:
                                img_feat = model.img_enc(img_tensor)
                                spatial_features = None
                                
                            pc_feat   = model.pc_enc(pc_tensor)
                            meta_feat = model.meta_enc(meta_tensor)

                            # 5. Fusion Transformer with attention maps
                            if hasattr(model.fusion, 'forward_with_attention'):
                                fused, layer_outputs, attn_maps = model.fusion.forward_with_attention(img_feat, pc_feat, meta_feat)
                                pos_emb_added = torch.stack([img_feat, pc_feat, meta_feat], dim=1) + model.fusion.pos_embedding
                            else:
                                stacked = torch.stack([img_feat, pc_feat, meta_feat], dim=1)
                                pos_emb_added = stacked + model.fusion.pos_embedding
                                x_iter = pos_emb_added
                                layer_outputs = []
                                attn_maps = []
                                for layer in model.fusion.transformer.layers:
                                    x_iter = layer(x_iter)
                                    layer_outputs.append(x_iter.clone())
                                fused = x_iter.flatten(start_dim=1)

                            # 8. Decoder — capture per-UpBlock output
                            proj_out  = model.decoder.proj(fused)
                            reshaped  = proj_out.view(-1, 256, 4, 4)
                            up1_out   = model.decoder.up1(reshaped)
                            up2_out   = model.decoder.up2(up1_out)
                            up3_out   = model.decoder.up3(up2_out)
                            up4_out   = model.decoder.up4(up3_out)
                            up5_out   = model.decoder.up5(up4_out)
                            up6_out   = model.decoder.up6(up5_out)
                            final_out = model.decoder.final_conv(up6_out)
                            output    = torch.sigmoid(final_out)                           # [1,1,256,256]

                        # ── Display results ──────────────────────────────────────────────
                        cscan_np    = output[0].squeeze(0).cpu().numpy()                   # [256,256]
                        cscan_uint8 = (cscan_np * 255).astype(np.uint8)
                        cscan_pil   = Image.fromarray(cscan_uint8, mode='L')
                        heatmap     = generate_heatmap(output[0])
                        score       = calculate_severity_score(output[0])

                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.image(cscan_pil, caption="Predicted C-Scan", use_container_width=True)
                        with rc2:
                            st.image(heatmap, caption="Severity Heatmap", use_container_width=True)

                        st.markdown("### Predicted Damage Severity")
                        st.markdown(f"<div class='severity-number'>{score}<span style='font-size:1.2rem;color:#94a3b8'>/100</span></div>", unsafe_allow_html=True)

                        if pc_path:
                            os.remove(pc_path)

                except Exception as _exc:
                    st.error(f"Pipeline error: {type(_exc).__name__}: {_exc}")
                    st.exception(_exc)

    # ────────────────────────────────────────────────────────────────────────────
    # MATH / PROCESSING TRACE — full width, only shown after a prediction
    # ────────────────────────────────────────────────────────────────────────────
    if run_btn and img_file:
        st.markdown("---")
        st.markdown("## Pipeline Processing Trace")
        st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Every stage of the forward pass — tensor shapes, statistics, and the math behind each transformation.</p>", unsafe_allow_html=True)

        # ── STAGE 0: Raw Inputs ──────────────────────────────────────────────────
        with st.expander("Stage 0 — Raw Inputs Visualization", expanded=True):
            st.markdown("**(Uploaded Surface Dent & 3D Topography)**")
            ci1, ci2 = st.columns(2)
            with ci1:
                st.image(img, caption="Raw Image Input", use_container_width=True)
            with ci2:
                if pc_path is not None:
                    try:
                        # pc_tensor comes in as [1, num_points, 3] from pre-processing
                        pts = pc_tensor[0].cpu().numpy()
                        fig_pc = plt.figure(figsize=(4,4))
                        ax_pc = fig_pc.add_subplot(111, projection='3d')
                        ax_pc.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c=pts[:,2], cmap='viridis')
                        ax_pc.set_title("Sampled Point Cloud (XYZ Topology)", color='white')
                        ax_pc.axis('off')
                        fig_pc.patch.set_facecolor('#111623')
                        st.image(fig_to_img(fig_pc), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render point cloud graph: {e}")

        # ── STAGE 1: Inputs ──────────────────────────────────────────────────────
        with st.expander("Stage 1 — Input Pre-Processing", expanded=True):
            step_card(
                "Image Pre-Processing  ·  ViT-B/16 pipeline",
                "img ∈ ℝ^{H×W×3}\n"
                "→ Resize to 224×224\n"
                "→ ToTensor()  →  x ∈ [0,1]^{3×224×224}\n"
                "→ Normalize(μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225])\n"
                "→ Unsqueeze(0)  →  img_tensor ∈ ℝ^{1×3×224×224}",
                tensor_stats(img_tensor, "img_tensor"),
                "#6366f1"
            )
            step_card(
                "Point Cloud Pre-Processing  ·  PointNet-style",
                "raw PLY/NPY  →  N points × 3 coords\n"
                "→ Random sample / repeat to num_points\n"
                "→ pc_tensor ∈ ℝ^{1×num_points×3}",
                tensor_stats(pc_tensor, "pc_tensor"),
                "#8b5cf6"
            )

            meta_raw = list(metadata.values())
            step_card(
                "Metadata Encoding",
                f"raw_features = {meta_raw}\n"
                f"→ dent_depth={dent_depth}, damage_area={damage_area}, "
                f"thickness={t_val}\n"
                f"→ material_encoded={material_map[material]}, "
                f"layup_encoded={len(layup)/10.0:.2f}\n"
                f"→ meta_tensor ∈ ℝ^{{1×5}}",
                tensor_stats(meta_tensor, "meta_tensor"),
                "#a855f7"
            )

        # ── STAGE 2: Encoders ─────────────────────────────────────────────────────
        with st.expander("Stage 2 — Modality Encoders", expanded=True):
            step_card(
                "Image Encoder  ·  ViT-B/16 + Linear projection",
                "img_tensor [1,3,224,224]\n"
                "→ ViT-B/16 patch embedding (16×16 patches = 196 tokens)\n"
                "→ 12 Transformer encoder layers  (d_model=768, heads=12)\n"
                "→ [CLS] token  →  Linear(768 → 512)\n"
                "→ img_feat ∈ ℝ^{1×512}",
                tensor_stats(img_feat, "img_feat"),
                "#3b82f6"
            )
            
            if spatial_features is not None:
                st.markdown("**(Extracted ViT Spatial Core Feature Map)**")
                # Mean over channels for visualization
                spatial_mean = spatial_features[0].mean(dim=0).cpu().numpy()
                fig, ax = plt.subplots(figsize=(3,3))
                sns.heatmap(spatial_mean, cmap='viridis', cbar=False, ax=ax, xticklabels=False, yticklabels=False)
                ax.set_title("Reshaped Patch Embeddings [14x14]", color='white')
                fig.patch.set_facecolor('#111623')
                st.image(fig_to_img(fig), use_container_width=False)
            step_card(
                "Point Cloud Encoder  ·  PointNet-style Conv1D",
                "pc_tensor [1,N,3]  →  transpose  →  [1,3,N]\n"
                "→ Conv1d(3→64)   + BN + ReLU\n"
                "→ Conv1d(64→128) + BN + ReLU\n"
                "→ Conv1d(128→256)+ BN + ReLU\n"
                "→ Conv1d(256→1024)+BN + ReLU\n"
                "→ GlobalMaxPool  →  [1,1024]\n"
                "→ FC(1024→512) + BN + ReLU + Dropout(0.3)\n"
                "→ FC(512→512)\n"
                "→ pc_feat ∈ ℝ^{1×512}",
                tensor_stats(pc_feat, "pc_feat"),
                "#06b6d4"
            )
            step_card(
                "Metadata Encoder  ·  3-layer MLP",
                "meta_tensor [1,5]\n"
                "→ Linear(5→64)   + ReLU\n"
                "→ Linear(64→256) + ReLU\n"
                "→ Linear(256→512)+ BN + Dropout(0.3)\n"
                "→ meta_feat ∈ ℝ^{1×512}",
                tensor_stats(meta_feat, "meta_feat"),
                "#10b981"
            )

            st.markdown("**(Extracted 1D Modality Latents)**")
            fig_lat, ax_lat = plt.subplots(1, 3, figsize=(10, 2.5))
            sns.heatmap(img_feat[0].view(16,32).cpu().numpy(), ax=ax_lat[0], cbar=False, cmap='mako', xticklabels=False, yticklabels=False)
            ax_lat[0].set_title("Image Feature [1, 512]\n(Reshaped 16x32)", color='white', fontsize=9)
            
            sns.heatmap(pc_feat[0].view(16,32).cpu().numpy(), ax=ax_lat[1], cbar=False, cmap='mako', xticklabels=False, yticklabels=False)
            ax_lat[1].set_title("Point Cloud Feature [1, 512]\n(Reshaped 16x32)", color='white', fontsize=9)
            
            sns.heatmap(meta_feat[0].view(16,32).cpu().numpy(), ax=ax_lat[2], cbar=False, cmap='mako', xticklabels=False, yticklabels=False)
            ax_lat[2].set_title("Metadata Feature [1, 512]\n(Reshaped 16x32)", color='white', fontsize=9)
            
            fig_lat.patch.set_facecolor('#111623')
            plt.tight_layout()
            st.image(fig_to_img(fig_lat), use_container_width=True)

        # ── STAGE 3: Fusion Transformer ───────────────────────────────────────────
        with st.expander("Stage 3 — Multimodal Transformer Fusion", expanded=True):
            step_card(
                "Token Stacking + Positional Embeddings",
                "Stack([img_feat, pc_feat, meta_feat], dim=1)  →  x ∈ ℝ^{1×3×512}\n"
                "pos_embedding ∈ ℝ^{1×3×512}  (learnable)\n"
                "x ← x + pos_embedding",
                tensor_stats(pos_emb_added, "x_after_pos_embed"),
                "#f59e0b"
            )

            for i, layer_out in enumerate(layer_outputs):
                step_card(
                    f"Transformer Layer {i+1} / {len(layer_outputs)}",
                    f"Self-Attention (heads=8, d_k=64)\n"
                    f"  Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V\n"
                    f"  Q=K=V projected from x_{i}  →  attended_x\n"
                    f"Add & LayerNorm  →  x' = LN(x_{i} + attended_x)\n"
                    f"FeedForward: Linear(512→2048) + GELU + Linear(2048→512)\n"
                    f"Add & LayerNorm  →  x_{i+1} = LN(x' + FF(x'))\n"
                    f"Output shape after layer {i+1}: {list(layer_out.shape)}",
                    tensor_stats(layer_out, f"layer_{i+1}_output"),
                    "#f97316"
                )
                if len(attn_maps) > i and attn_maps[i] is not None:
                    # avg over heads if it's [B, num_heads, Seq, Seq], or check shape
                    attn_m = attn_maps[i][0]
                    if attn_m.dim() == 3:
                        attn_m = attn_m.mean(dim=0)
                    attn_np = attn_m.cpu().numpy()
                    
                    fig2, ax2 = plt.subplots(figsize=(4,3))
                    sns.heatmap(attn_np, annot=True, cmap='coolwarm', ax=ax2, 
                                xticklabels=['Img', 'PC', 'Meta'], yticklabels=['Img', 'PC', 'Meta'])
                    ax2.set_title(f"Layer {i+1} Cross-Attention", color='white')
                    fig2.patch.set_facecolor('#111623')
                    ax2.tick_params(colors='white')
                    st.image(fig_to_img(fig2), use_container_width=False)

            step_card(
                "Flatten Fusion Tokens",
                "x_final ∈ ℝ^{1×3×512}\n"
                "→ flatten(start_dim=1)  →  fused ∈ ℝ^{1×1536}",
                tensor_stats(fused, "fused"),
                "#ef4444"
            )

        # ── STAGE 4: Decoder ──────────────────────────────────────────────────────
        with st.expander("Stage 4 — C-Scan Decoder (Progressive Upsampling)", expanded=True):
            step_card(
                "Linear Projection",
                "fused ∈ ℝ^{1×1536}\n"
                "→ Linear(1536 → 256×4×4 = 4096)\n"
                "→ Reshape  →  ℝ^{1×256×4×4}",
                tensor_stats(reshaped, "proj_reshaped"),
                "#8b5cf6"
            )

            decoder_steps = [
                ("UpBlock 1  →  4×4 to 8×8",
                 "Upsample(×2, bilinear) → Conv2d(256→128, k=3) + BN + ReLU",
                 up1_out, "#6366f1"),
                ("UpBlock 2  →  8×8 to 16×16",
                 "Upsample(×2, bilinear) → Conv2d(128→64,  k=3) + BN + ReLU",
                 up2_out, "#7c3aed"),
                ("UpBlock 3  →  16×16 to 32×32",
                 "Upsample(×2, bilinear) → Conv2d(64→32,   k=3) + BN + ReLU",
                 up3_out, "#8b5cf6"),
                ("UpBlock 4  →  32×32 to 64×64",
                 "Upsample(×2, bilinear) → Conv2d(32→16,   k=3) + BN + ReLU",
                 up4_out, "#a855f7"),
                ("UpBlock 5  →  64×64 to 128×128",
                 "Upsample(×2, bilinear) → Conv2d(16→8,    k=3) + BN + ReLU",
                 up5_out, "#c026d3"),
                ("UpBlock 6  →  128×128 to 256×256",
                 "Upsample(×2, bilinear) → Conv2d(8→8,     k=3) + BN + ReLU",
                 up6_out, "#db2777"),
            ]

            fig_grid, axes_grid = plt.subplots(2, 3, figsize=(10, 6))
            axes_grid = axes_grid.flatten()
            for idx, (title, math, tensor, color) in enumerate(decoder_steps):
                step_card(title, math, tensor_stats(tensor, title), color)
                # Plot progression
                up_mean = tensor[0].mean(dim=0).cpu().numpy()
                axes_grid[idx].imshow(up_mean, cmap='magma')
                axes_grid[idx].set_title(f"Up{idx+1}: {up_mean.shape[0]}x{up_mean.shape[1]}", color='white', fontsize=10)
                axes_grid[idx].axis('off')
            
            fig_grid.patch.set_facecolor('#111623')
            plt.tight_layout()
            st.markdown("**(Progressive Representation Upsampling)**")
            st.image(fig_to_img(fig_grid), use_container_width=True)

            step_card(
                "Final Conv + Sigmoid  →  C-Scan Output",
                "x ∈ ℝ^{1×8×256×256}\n"
                "→ Conv2d(8→1, k=3, padding=1)  →  logits ∈ ℝ^{1×1×256×256}\n"
                "→ Sigmoid(logits)  →  output ∈ [0,1]^{1×1×256×256}\n"
                "→ squeeze  →  cscan ∈ [0,1]^{256×256}",
                tensor_stats(output[0], "final_output"),
                "#e11d48"
            )

        # ── STAGE 5: Post-Processing ──────────────────────────────────────────────
        with st.expander("Stage 5 — Post-Processing & Scoring", expanded=True):
            top_k = max(1, int(0.05 * output[0].numel()))
            topk_vals = torch.topk(output[0].view(-1), top_k)[0]

            step_card(
                "Heatmap Generation",
                "cscan ∈ [0,1]^{256×256}\n"
                "→ × 255  →  uint8 grayscale\n"
                "→ cv2.applyColorMap(COLORMAP_JET)\n"
                "→ BGR → RGB  →  heatmap ∈ ℝ^{256×256×3}",
                f"colormap=JET  output_dtype=uint8  shape=[256,256,3]",
                "#14b8a6"
            )
            step_card(
                "Severity Score",
                f"cscan_flat ∈ ℝ^{{{output[0].numel()}}}\n"
                f"k = max(1, int(0.05 × {output[0].numel()})) = {top_k}\n"
                f"top_k_vals = TopK(cscan_flat, k={top_k})\n"
                f"score = mean(top_k_vals) × 100 = {topk_vals.mean().item()*100:.2f}\n"
                f"clamped_score = clip(score, 0, 100) = {score}",
                f"top-5% mean intensity = {topk_vals.mean().item():.4f}  →  score = {score}/100",
                "#f59e0b"
            )

# ══════════════════════════════════════════════
# TAB 2 — Batch Dataset
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Connect OneDrive Dataset</div>', unsafe_allow_html=True)
    st.markdown("Provide the local path to your OneDrive folder containing dataset images, point clouds, and a CSV manifest.")

    onedrive_path = st.text_input("OneDrive Folder Path", placeholder="C:/Users/yourname/OneDrive/Dataset")
    dataset_csv   = st.text_input("CSV Manifest File Name", placeholder="dataset.csv")

    st.markdown('<div class="section-header">Global Metadata</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        dataset_thickness = st.text_input("Base Thickness (mm)", value="5.0", key="d_thick")
    with dc2:
        dataset_material = st.selectbox("Material Type", ["CFRP", "GFRP", "Hybrid"], key="d_mat")

    load_btn = st.button("Load & Preview Dataset", use_container_width=True)

    if load_btn:
        if not os.path.exists(onedrive_path):
            st.error(f"Cannot find path: {onedrive_path}")
        else:
            csv_path = os.path.join(onedrive_path, dataset_csv)
            if not os.path.exists(csv_path):
                st.error(f"Cannot find manifest: {csv_path}")
            else:
                try:
                    df = pd.read_csv(csv_path)
                    st.success(f"Loaded {len(df)} samples from dataset manifest.")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.info("Batch inference can be triggered by iterating rows and calling `pipeline.predict()` for each sample.")
                except Exception as e:
                    st.error("Error reading CSV file.")
                    st.exception(e)
