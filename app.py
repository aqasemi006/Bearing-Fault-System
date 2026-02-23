import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os

# --- تنظیمات صفحه ---
st.set_page_config(page_title="SUT Advanced Diagnosis", page_icon="🔬", layout="wide")

# --- منوی سمت چپ ---
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    st.markdown("### 🎓 Academic Project")
    st.info("System optimized for Transfer Learning approach as per SUT guidelines.")
    st.divider()
    st.markdown("**Researcher:** A.M. Ghasemi Nezhad")
    st.markdown("**Supervisors:** Dr. Abbasloo & Dr. Ghasemi")

# --- بدنه اصلی ---
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
        with st.spinner('Extracting Features and Analyzing...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            
            # --- بخش هوشمند برای پیدا کردن داده در فایل‌های مختلف ---
            for key in mat_data.keys():
                if 'time' in key.lower() and 'DE' in key.upper():
                    vibration_data = mat_data[key].flatten()
                    st.info(f"Variable detected: {key}")
                    break
            
            if vibration_data is None: # اگر پیدا نشد، اولین آرایه عددی را بردار
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        vibration_data = mat_data[key].flatten()
                        break

            if vibration_data is not None:
                # آماده‌سازی داده (باید با ورودی مدل جدید هماهنگ باشد)
                segment = vibration_data[:4096]
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
                
                # اگر مدل Transfer Learning باشد معمولاً ورودی 3 کاناله (RGB) می‌خواهد
                # فعلا طبق مدل قبلی شما:
                input_data = np.expand_dims(spec_db, axis=(0, -1))
                
                prediction = model.predict(input_data)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx] * 100
                
                classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                
                st.success("Analysis Completed.")
                col1, col2 = st.columns(2)
                col1.metric("Diagnosis Result", classes[class_idx])
                col1.metric("Reliability Score", f"{confidence:.2f}%")
                
                if os.path.exists(f"{classes[class_idx].lower().replace(' ', '_')}.png"):
                    col2.image(f"{classes[class_idx].lower().replace(' ', '_')}.png", width=300)
            else:
                st.error("Format Error: Could not find vibration signal in this .mat file.")
except Exception as e:
    st.error(f"System Error: {e}")
