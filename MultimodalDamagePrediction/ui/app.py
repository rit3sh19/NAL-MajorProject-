import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.pipeline import InferencePipeline
from inference.postprocess import generate_heatmap, calculate_severity_score

st.set_page_config(page_title="Composite Damage Predictor", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Multimodal AI: Internal Damage Predictor")

@st.cache_resource
def load_model():
    return InferencePipeline("configs/config.yaml", "checkpoints/best_model.pth")

pipeline = load_model()
material_map = {"CFRP": 0, "GFRP": 1, "Hybrid": 2}

tab1, tab2 = st.tabs(["Single Prediction", "Dataset Batch Processing (From OneDrive)"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Data Input")
        img_file = st.file_uploader("Upload Surface Dent Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
        pc_file = st.file_uploader("Upload 3D Point Cloud Data (.npy)", type=['npy'])
        
        st.markdown("### Metadata Parameters")
        c1, c2 = st.columns(2)
        with c1:
            thickness = st.text_input("Thickness (mm)", value="5.0")
            dent_depth = st.number_input("Dent Depth (mm)", min_value=0.0, value=2.0)
            damage_area = st.number_input("Damage Area", min_value=0.0, value=15.0)
        with c2:
            material = st.selectbox("Material Type", ["CFRP", "GFRP", "Hybrid"])
            layup = st.text_input("Layup Sequence", "[0/90/45/-45]s")
            
        run_btn = st.button("Predict C-Scan Damage")

    with col2:
        st.subheader("Analysis Output")
        if run_btn:
            if not img_file:
                st.error("Please upload an image.")
            else:
                with st.spinner("Processing Data..."):
                    img = Image.open(img_file)
                    pc_path = None
                    if pc_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
                            tmp.write(pc_file.getvalue())
                            pc_path = tmp.name
                    
                    try:
                        t_val = float(thickness)
                    except:
                        t_val = 5.0
                        
                    metadata = {
                        "thickness": t_val,
                        "dent_depth": dent_depth,
                        "damage_area": damage_area,
                        "material_type_encoded": material_map[material],
                        "layup_sequence_encoded": len(layup) / 10.0
                    }
                    
                    cscan_tensor = pipeline.predict(img, pc_path, metadata)
                    cscan_display = cscan_tensor.squeeze(0).cpu().numpy()
                    cscan_display = (cscan_display * 255).astype(np.uint8)
                    cscan_pil = Image.fromarray(cscan_display)
                    heatmap = generate_heatmap(cscan_tensor)
                    score = calculate_severity_score(cscan_tensor)
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.image(cscan_pil, caption="Predicted C-Scan", use_container_width=True)
                    with rc2:
                        st.image(heatmap, caption="Severity Heatmap", use_container_width=True)
                        
                    st.markdown(f"### Predicted Damage Severity")
                    st.markdown(f"<div class='metric-value'>{score}/100</div>", unsafe_allow_html=True)
                    if pc_path:
                        os.remove(pc_path)

with tab2:
    st.subheader("Connect Dataset from OneDrive")
    st.markdown("Provide the local absolute path to your OneDrive folder containing your dataset images, point clouds, and a CSV mapping file.")
    
    onedrive_path = st.text_input("OneDrive Folder Path (e.g. C:/Users/rksur/OneDrive/Dataset)", placeholder="C:/Users/rksur/OneDrive/...")
    dataset_csv = st.text_input("Dataset Manifest File Name", placeholder="dataset.csv [Must contain columns: Image_Name, PC_Name, Dent_Depth, Damage_Area, Layup_Sequence]")
    
    st.markdown("### Global Metadata Properties")
    dc1, dc2 = st.columns(2)
    with dc1:
        dataset_thickness = st.text_input("Dataset Base Thickness (mm)", value="5.0", key="d_thick")
    with dc2:
        dataset_material = st.selectbox("Dataset Material Type", ["CFRP", "GFRP", "Hybrid"], key="d_mat")
    
    load_btn = st.button("Load & Preview OneDrive Dataset")
    
    if load_btn:
        if not os.path.exists(onedrive_path):
            st.error(f"Cannot find the specified path: {onedrive_path}")
        else:
            csv_path = os.path.join(onedrive_path, dataset_csv)
            if not os.path.exists(csv_path):
                st.error(f"Cannot find manifest file at: {csv_path}")
            else:
                try:
                    df = pd.read_csv(csv_path)
                    st.success(f"Successfully loaded dataset mapping from OneDrive! Found {len(df)} samples.")
                    st.dataframe(df.head())
                    st.info("Batch inference functionality can be integrated to process these records using the InferencePipeline.")
                except Exception as e:
                    st.error("Error reading CSV file.")
                    st.exception(e)
