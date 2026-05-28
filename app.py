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
            
            # --- الگوریتم هوشمند شکافتن ماتریس‌های ساختاریافته (Struct) ---
            for key in mat_data.keys():
                if not key.startswith('__'):
                    raw = mat_data[key]
                    
                    # حالت اول: اگر دیتا ساختار داده‌ای پیچیده (شامل فیلد) دارد
                    if raw.dtype.names is not None:
                        for name in raw.dtype.names:
                            # به دنبال فیلدهای عددی مثل X یا Y یا Data بگرد
                            sub_data = raw[name][0, 0] if raw.ndim > 1 else raw[name]
                            try:
                                temp_arr = np.real(sub_data).flatten().astype(float)
                                if len(temp_arr) > 1024:
                                    vibration_data = temp_arr
                                    st.write(f"📂 Extracted from Struct: `{key} -> {name}` (Length: {len(vibration_data)})")
                                    break
                            except: continue
                    # حالت دوم: اگر دیتا یک آرایه عددی مستقیم (مختلط یا ساده) است
                    else:
                        try:
                            temp_arr = np.real(raw).flatten().astype(float)
                            if len(temp_arr) > 1024:
                                vibration_data = temp_arr
                                st.write(f"📊 Extracted from Array: `{key}` (Length: {len(vibration_data)})")
                                break
                        except: continue
                if vibration_data is not None: break

            # --- بخش پردازش و هوش مصنوعی ---
            if vibration_data is not None:
                # برش استاندارد
                segment = vibration_data[:4096] if len(vibration_data) >= 4096 else vibration_data
                if len(segment) < 1024:
                    st.error("Signal segment is too short for meaningful frequency analysis.")
                else:
                    # تبدیل زمان-فرکانس (Spectrogram)
                    f, t, Sxx = signal.spectrogram(segment, fs=12000)
                    spec_db = 10 * np.log10(Sxx + 1e-10)
                    spec_norm = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-6)
                    
                    # انطباق ابعادی با متد یادگیری انتقالی (224x224 RGB)
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
                        st.warning("Model core (`bearing_model.h5`) is missing. Please train and upload the Transfer Learning weight file.")
            else:
                st.error("Structure Error: Unable to locate any 1D numerical vibration vector inside this .mat file.")
except Exception as e:
    st.error(f"Execution Error: {str(e)}")
