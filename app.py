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
    elif os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    st.markdown("### 🎓 Academic Project")
    st.info("Transfer Learning & Domain Adaptation System.")
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
        with st.spinner('Scanning Data Structures...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            
            # --- بخش هوشمند استخراج دیتا از ساختارهای پیچیده (Structs) ---
            for key in mat_data.keys():
                if not key.startswith('__'):
                    raw = mat_data[key]
                    
                    # اگر دیتا ساختار پیچیده (VoidDType/Object) داشت
                    if raw.dtype.names is not None:
                        for name in raw.dtype.names:
                            candidate = raw[name].flatten()
                            # چک کردن اینکه آیا این فیلد حاوی اعداد است یا متن
                            if candidate.dtype.kind in 'if': # i=integer, f=float
                                vibration_data = candidate.astype(float)
                                break
                    # اگر دیتا آرایه مستقیم بود
                    elif raw.dtype.kind in 'ifc': # i, f, c=complex
                        vibration_data = np.real(raw).flatten().astype(float)
                    
                    if vibration_data is not None:
                        st.success(f"Signal found in key: '{key}'")
                        break

            if vibration_data is not None and len(vibration_data) > 4096:
                # تحلیل سیگنال
                segment = vibration_data[:4096]
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-6)
                
                input_data = np.expand_dims(spec_db, axis=(0, -1))
                prediction = model.predict(input_data)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx] * 100
                
                classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                classes_fa = ["سالم", "خرابی رینگ داخلی", "خرابی رینگ خارجی", "خرابی ساچمه"]

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Diagnosis Result", classes[class_idx])
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                    if class_idx != 0:
                        st.error(f"⚠️ {classes_fa[class_idx]}")
                with col2:
                    img_name = classes[class_idx].lower().replace(" ", "_") + ".png"
                    if os.path.exists(img_name):
                        st.image(img_name, width=280)
            else:
                st.error("Could not find a valid numerical vibration signal in the uploaded file.")

except Exception as e:
    st.error(f"Technical Error: {str(e)}")
