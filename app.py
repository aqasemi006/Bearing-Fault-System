import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os

# --- Page Config ---
st.set_page_config(page_title="SUT Advanced Diagnosis", page_icon="🔬", layout="wide")

# --- Sidebar ---
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    st.markdown("### 🎓 Academic Project")
    st.info("System optimized for Transfer Learning & Cross-Domain Analysis.")
    st.divider()
    st.markdown("**Researcher:** Amir Mohammad Ghasemi Nezhad")
    st.markdown("**Supervisors:** Dr. Abbasloo & Dr. Ghasemi")

# --- Header ---
st.title("Intelligent Bearing Fault Diagnosis System")
st.subheader("Advanced Transfer Learning Approach (CWRU Standard)")
st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bearing_model.h5")

try:
    model = load_model()
    uploaded_file = st.file_uploader("Upload Vibration File (.mat)", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('Pre-processing Signal & Domain Adaptation...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            
            # پیدا کردن هوشمند داده‌ها
            for key in mat_data.keys():
                if not key.startswith('__'):
                    raw_data = mat_data[key]
                    # حل مشکل اعداد مختلط و DType که در ارور بود
                    if np.iscomplexobj(raw_data):
                        vibration_data = np.real(raw_data).flatten()
                    else:
                        vibration_data = np.array(raw_data, dtype=float).flatten()
                    st.info(f"Signal Data extracted from: {key}")
                    break

            if vibration_data is not None and len(vibration_data) > 4096:
                # پردازش سیگنال (STFT)
                segment = vibration_data[:4096]
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                # نرمال‌سازی دقیق
                spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-6)
                
                input_data = np.expand_dims(spec_db, axis=(0, -1))
                
                # پیش‌بینی با مدل
                prediction = model.predict(input_data)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx] * 100
                
                classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                classes_fa = ["سالم", "خرابی رینگ داخلی", "خرابی رینگ خارجی", "خرابی ساچمه"]

                st.success("Analysis Completed Successfully.")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Diagnosis Result", classes[class_idx])
                    st.metric("Reliability Score", f"{confidence:.2f}%")
                    if class_idx != 0:
                        st.error(f"⚠️ Warning: {classes_fa[class_idx]} Detected")
                
                with col2:
                    img_name = classes[class_idx].lower().replace(" ", "_") + ".png"
                    if os.path.exists(img_name):
                        st.image(img_name, width=300)
            else:
                st.error("Error: Signal length is too short or data is corrupted.")
except Exception as e:
    st.error(f"Processing Error: {str(e)}")
