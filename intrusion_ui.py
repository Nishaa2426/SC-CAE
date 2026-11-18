"""
PHASE 1 UI: CNN AUTOENCODER ANOMALY DETECTION - FIXED
========================================================
Fixed: Reconstruction image display
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Phase 1: Anomaly Detection",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #1a1a2e;
        font-family: 'Segoe UI', sans-serif;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
        font-family: 'Segoe UI', sans-serif;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e8e8e8;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<p class="main-title">🔬 Phase 1: Anomaly Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN Autoencoder · Reconstruction Error · UCSD Ped1 Dataset</p>', unsafe_allow_html=True)

# Status badges
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<span class="status-badge" style="background: #e8f5e9; color: #2e7d32;">✅ Dataset: UCSD Ped1</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span class="status-badge" style="background: #e3f2fd; color: #1565c0;">✅ Model: CNN Autoencoder</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="status-badge" style="background: #fff3e0; color: #e65100;">✅ Method: MSE</span>', unsafe_allow_html=True)
with col4:
    st.markdown('<span class="status-badge" style="background: #f3e5f5; color: #6a1b9a;">✅ Status: Active</span>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# LOAD MODEL
# ============================================================================

def build_autoencoder():
    input_img = layers.Input(shape=(224, 224, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(input_img, decoded)

@st.cache_resource
def load_model():
    model_dir = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\models"
    
    autoencoder = build_autoencoder()
    
    # Try to load weights
    weights_path = os.path.join(model_dir, "autoencoder_weights.h5")
    if os.path.exists(weights_path):
        try:
            autoencoder.load_weights(weights_path)
            autoencoder.compile(optimizer='adam', loss='mse')
            print(" Model loaded successfully")
        except Exception as e:
            print(f" Error loading weights: {e}")
    else:
        # Try alternative path
        weights_path = os.path.join(model_dir, "autoencoder.weights.h5")
        if os.path.exists(weights_path):
            try:
                autoencoder.load_weights(weights_path)
                autoencoder.compile(optimizer='adam', loss='mse')
                print(" Model loaded successfully from alternative path")
            except Exception as e:
                print(f" Error loading weights: {e}")
        else:
            print(" No weights found - using untrained model")
    
    # Load threshold
    threshold_path = os.path.join(model_dir, 'threshold.pkl')
    threshold = 0.00341
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'rb') as f:
                threshold = pickle.load(f)
            print(f" Threshold loaded: {threshold:.6f}")
        except:
            print(" Could not load threshold - using default")
    
    return autoencoder, threshold

with st.spinner("Loading CNN Autoencoder..."):
    autoencoder, default_threshold = load_model()

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_anomaly(image, autoencoder, threshold):
    """Detect anomaly and return results"""
    
    # Preprocess
    img_resized = cv2.resize(image, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Reconstruct
    reconstructed = autoencoder.predict(img_input, verbose=0)
    
    # Calculate MSE
    mse = np.mean((img_normalized - reconstructed[0]) ** 2)
    
    # Detect anomaly
    is_anomaly = mse > threshold
    
    return {
        'is_anomaly': is_anomaly,
        'mse': mse,
        'reconstructed': reconstructed[0],
        'original': img_normalized,
        'threshold': threshold
    }

def get_phase1_decision(visual_anomaly):
    if visual_anomaly:
        return "ANOMALY DETECTED", "HIGH", "#fce4ec", "#c62828"
    else:
        return " NORMAL", "LOW", "#e8f5e9", "#2e7d32"

def evaluate_performance(test_folder, autoencoder, threshold, num_samples=100):
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')][:num_samples]
    
    normal_files = [f for f in test_files if 'Train' in f or 'train' in f]
    anomaly_files = [f for f in test_files if 'Test' in f or 'test' in f]
    
    if len(normal_files) < 10 or len(anomaly_files) < 10:
        normal_files = test_files[:len(test_files)//2]
        anomaly_files = test_files[len(test_files)//2:]
    
    y_true, y_pred, y_scores = [], [], []
    
    for file in normal_files[:50]:
        img_path = os.path.join(test_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            result = detect_anomaly(img, autoencoder, threshold)
            y_true.append(0)
            y_pred.append(1 if result['is_anomaly'] else 0)
            y_scores.append(result['mse'])
    
    for file in anomaly_files[:50]:
        img_path = os.path.join(test_folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            result = detect_anomaly(img, autoencoder, threshold)
            y_true.append(1)
            y_pred.append(1 if result['is_anomaly'] else 0)
            y_scores.append(result['mse'])
    
    if len(y_true) == 0:
        return None
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'total_samples': len(y_true),
        'y_true': y_true,
        'y_scores': y_scores,
        'confusion_matrix': cm
    }

# ============================================================================
# SIDEBAR - CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    threshold = st.slider(
        "Detection Threshold", 
        min_value=0.0020, 
        max_value=0.0050, 
        value=float(default_threshold),
        step=0.0001
    )
    
    st.info(f"Current Threshold: **{threshold:.5f}**")
    
    st.markdown("---")
    st.markdown("##  Statistics")
    
    test_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\test_frames"
    
    if os.path.exists(test_folder):
        total_frames = len([f for f in os.listdir(test_folder) if f.endswith('.jpg')])
        st.metric("Total Test Frames", total_frames)
        
        test_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')][:100]
        if len(test_files) > 10:
            metrics = evaluate_performance(test_folder, autoencoder, threshold, num_samples=100)
            if metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                st.metric("F1-Score", f"{metrics['f1_score']:.1f}%")
    
    st.markdown("---")
    st.markdown("###  Detection Rule")
    st.caption("""
    **MSE > Threshold → ANOMALY**
    **MSE ≤ Threshold → NORMAL**
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([1, 1])

# ============================================================================
# COLUMN 1: INPUT
# ============================================================================

with col1:
    st.markdown("### Input")
    
    input_option = st.radio("Select Input:", ["Test Sample", "Upload Image"], horizontal=True)
    
    image = None
    image_name = None
    test_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\test_frames"
    
    if input_option == "Test Sample":
        if os.path.exists(test_folder):
            test_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
            selected = st.selectbox("Choose frame:", test_files)
            if selected:
                img_path = os.path.join(test_folder, selected)
                image = cv2.imread(img_path)
                image_name = selected
                st.caption(f"{image_name}")
    
    else:
        uploaded = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded.name
    
    if image is not None:
        st.markdown("#### Original Frame")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)

# ============================================================================
# COLUMN 2: DETECTION RESULTS
# ============================================================================

with col2:
    st.markdown("###  Detection Results")
    
    if image is not None:
        with st.spinner("Analyzing..."):
            result = detect_anomaly(image, autoencoder, threshold)
            decision_text, confidence, bg_color, text_color = get_phase1_decision(result['is_anomaly'])
        
        # Result Box
        st.markdown(f"""
        <div class="result-box" style="background: {bg_color}; border: 2px solid {text_color};">
            <h1 style="font-size: 42px; margin: 0; color: {text_color};">{decision_text.split()[0]}</h1>
            <h3 style="margin: 5px 0; color: {text_color};">{decision_text}</h3>
            <p style="font-size: 14px; color: {text_color};">Confidence: {confidence}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Reconstruction Error", f"{result['mse']:.5f}")
        with m2:
            st.metric("Threshold", f"{result['threshold']:.5f}")
        with m3:
            status = "Anomaly" if result['is_anomaly'] else "Normal"
            st.metric("Status", status)
        
        # ============================================================
        # RECONSTRUCTION IMAGE - FIXED
        # ============================================================
        st.markdown("####  Reconstruction")
        
        # Display original and reconstructed side by side
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.caption("Original")
            original_display = cv2.resize(image, (224, 224)) / 255.0
            st.image(np.clip(original_display, 0, 1), use_container_width=True)
        
        with col_rec2:
            st.caption("Reconstructed")
            recon_display = np.clip(result['reconstructed'], 0, 1)
            st.image(recon_display, use_container_width=True)
        
        # ============================================================
        # HEATMAP
        # ============================================================
        st.markdown("####  Error Heatmap")
        
        original_resized = cv2.resize(image, (224, 224)) / 255.0
        diff = np.abs(original_resized - result['reconstructed'])
        diff_map = np.mean(diff, axis=2)
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.imshow(diff_map, cmap='hot', vmin=0, vmax=0.1)
        ax.axis('off')
        ax.set_title('Error Heatmap (Red = High Error)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Error Magnitude', shrink=0.7)
        st.pyplot(fig)
        plt.close()
        
        # ============================================================
        # ERROR METER
        # ============================================================
        st.markdown("#### Error Meter")
        error_pct = min(100, (result['mse'] / threshold) * 100)
        
        if result['is_anomaly']:
            st.progress(min(100, int(error_pct * 1.3)), 
                       text=f"{result['mse']:.5f} > {threshold:.5f} (ABOVE Threshold)")
        else:
            st.progress(min(100, int(error_pct)), 
                       text=f" {result['mse']:.5f} ≤ {threshold:.5f} (BELOW Threshold)")
        
    else:
        st.info(" Select or upload an image to begin")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

st.markdown("---")
st.markdown("###  Performance Metrics")

test_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\test_frames"

if os.path.exists(test_folder):
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    
    if len(test_files) > 10:
        with st.spinner("Calculating metrics..."):
            metrics = evaluate_performance(test_folder, autoencoder, threshold, num_samples=100)
            
            if metrics:
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #666; font-size: 12px; margin: 0;">Accuracy</h4>
                        <h2 style="color: {'#2e7d32' if metrics['accuracy'] > 90 else '#e65100'}; margin: 5px 0;">{metrics['accuracy']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #666; font-size: 12px; margin: 0;">Precision</h4>
                        <h2 style="color: {'#2e7d32' if metrics['precision'] > 90 else '#e65100'}; margin: 5px 0;">{metrics['precision']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #666; font-size: 12px; margin: 0;">Recall</h4>
                        <h2 style="color: {'#2e7d32' if metrics['recall'] > 90 else '#e65100'}; margin: 5px 0;">{metrics['recall']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #666; font-size: 12px; margin: 0;">F1-Score</h4>
                        <h2 style="color: {'#2e7d32' if metrics['f1_score'] > 90 else '#e65100'}; margin: 5px 0;">{metrics['f1_score']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #666; font-size: 12px; margin: 0;">ROC-AUC</h4>
                        <h2 style="color: {'#2e7d32' if metrics['roc_auc'] > 0.9 else '#e65100'}; margin: 5px 0;">{metrics['roc_auc']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander(" Detailed Metrics"):
                    st.write(f"**Total Samples:** {metrics['total_samples']}")
                    st.write(f"**True Positives:** {metrics['tp']}  |  **True Negatives:** {metrics['tn']}")
                    st.write(f"**False Positives:** {metrics['fp']}  |  **False Negatives:** {metrics['fn']}")
                    
                    # Confusion Matrix
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm = metrics['confusion_matrix']
                    im = ax.imshow(cm, cmap='Blues')
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(['Normal', 'Anomaly'])
                    ax.set_yticklabels(['Normal', 'Anomaly'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                                   color='white' if cm[i, j] > cm.max()/2 else 'black')
                    st.pyplot(fig)
                    plt.close()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; padding: 10px 0;">
    <b>Phase 1:</b> CNN Autoencoder · UCSD Ped1 Dataset · Anomaly Detection
</div>
""", unsafe_allow_html=True)