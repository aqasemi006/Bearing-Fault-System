import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os
import cv2

# --- Page Configuration ---
st.set_page_config(page_title="SUT Bearing Diagnosis", page_icon="⚙️", layout="wide")

# --- Sidebar ---
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    elif os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    st.markdown("### 🎓 Academic Project")
    st.info("Cross-Domain Fault Diagnosis using Advanced Transfer Learning")
    st.divider()
    st.markdown("**Researcher:** Amir Mohammad Ghasemi Nezhad")
    st.markdown("**Supervisors:** Dr. Abbasloo & Dr. Ghasemi")

st.title("Intelligent Bearing Condition Monitoring System")
st.subheader("Domain Adaptation Framework for Multi-Source Datasets")
st.divider()

@st.cache_resource
def load_bearing_model():
    if os.path.exists("bearing_model.h5"):
        return tf.keras.models.load_model("bearing_model.h5")
    return None

try:
    model = load_bearing_model()
    uploaded_file = st.file_uploader("Upload Vibration File (.mat)", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('Parsing Advanced Matrix Structures...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            
            # --- الگوریتم فوق‌هوشمند اسکن عمقی ماتریس ---
            for key in mat_data.keys():
                if not key.startswith('__'):
                    raw = mat_data[key]
                    
                    # بررسی لایه اول: اگر مستقیم آرایه عددی باشد
                    try:
                        flat_data = np.real(raw).flatten()
                        if flat_data.dtype.kind in 'ifc' and len(flat_data) > 1024:
                            vibration_data = flat_data.astype(float)
                            st.write(f"📊 Signal found directly in: `{key}` (Length: {len(vibration_data)})")
                            break
                    except: pass
                    
                    # بررسی لایه دوم: اگر متغیر ساختاریافته (Struct) باشد
                    if hasattr(raw, 'dtype') and raw.dtype.names is not None:
                        for name in raw.dtype.names:
                            try:
                                # استخراج لایه‌های تو در تو
                                sub_element = raw[name]
                                while sub_element.ndim > 0 and sub_element.dtype.kind == 'O':
                                    sub_element = sub_element[0]
                                if sub_element.size > 0:
                                    sub_element = sub_element[0, 0]
                                
                                flat_sub = np.real(sub_element).flatten()
                                if flat_sub.dtype.kind in 'ifc' and len(flat_sub) > 1024:
                                    vibration_data = flat_sub.astype(float)
                                    st.write(f"📂 Signal extracted from nested Struct: `{key} -> {name}`")
                                    break
                            except: continue
                if vibration_data is not None: break

            # --- بخش پردازش سیگنال و مدل هوش مصنوعی ---
            if vibration_data is not None:
                segment = vibration_data[:4096]
                
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                spec_norm = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-6)
                
                img_resized = cv2.resize(spec_norm, (224, 224))
                img_3channel = np.stack([img_resized] * 3, axis=-1)
                input_tensor = np.expand_dims(img_3channel, axis=0)
                
                if model:
                    prediction = model.predict(input_tensor)
                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx] * 100
                    
                    classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                    
                    st.success("Domain Adaptation Analysis Complete.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Generalization Diagnosis", classes[class_idx])
                        st.metric("Domain Confidence Score", f"{confidence:.2f}%")
                    with col2:
                        img_file = classes[class_idx].lower().replace(" ", "_") + ".png"
                        if os.path.exists(img_file): st.image(img_file, width=280)
                else:
                    st.warning("Model core (`bearing_model.h5`) is missing. Please upload the Transfer Learning weight file.")
            else:
                st.error("Structure Error: Unable to locate any 1D numerical vibration vector inside this .mat file.")
                # چاپ کلیدها برای راهنمایی کاربر در محیط وب
                st.info(f"Available keys in this file: {list(mat_data.keys())}")
except Exception as e:
    st.error(f"Execution Error: {str(e)}")
