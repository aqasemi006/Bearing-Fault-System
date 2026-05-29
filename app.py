import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os
import cv2
import gc

# --- ۱. مدیریت سخت‌گیرانه سشن برای جلوگیری از انباشت رم ---
if "current_file" not in st.session_state:
    st.session_state["current_file"] = None

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

    # مکانیزم ریست خودکار سشن در صورت تغییر فایل
    if uploaded_file is not None:
        if st.session_state["current_file"] != uploaded_file.name:
            st.session_state["current_file"] = uploaded_file.name
            # تخلیه اجباری رم قبل از باز کردن فایل جدید
            gc.collect()

        with st.spinner('Parsing Advanced Matrix Structures safely...'):
            # خواندن بایت‌ها به صورت مستقیم برای جلوگیری از کپی شدن در رم
            file_bytes = uploaded_file.read()
            mat_data = scipy.io.loadmat(scipy.io.matlab.mio.BytesIO(file_bytes))
            del file_bytes # حذف فوری بایت‌های خام
            
            vibration_data = None
            target_key = None
            
            for key in mat_data.keys():
                if not key.startswith('__') and key not in ['header', 'version', 'globals']:
                    target_key = key
                    break
            
            if target_key is not None:
                raw = mat_data[target_key]
                
                # استخراج ساختار تودرتو پادربورن
                if hasattr(raw, 'dtype') and raw.dtype.names is not None:
                    for name in raw.dtype.names:
                        try:
                            sub_element = raw[name][0, 0]
                            if hasattr(sub_element, 'dtype') and sub_element.dtype.names is not None:
                                for sub_name in sub_element.dtype.names:
                                    deep_data = sub_element[sub_name][0, 0]
                                    flat_arr = np.real(deep_data).flatten()
                                    if len(flat_arr) > 4096 and flat_arr.dtype.kind in 'ifc':
                                        vibration_data = flat_arr.astype(np.float32)
                                        break
                            else:
                                flat_arr = np.real(sub_element).flatten()
                                if len(flat_arr) > 4096 and flat_arr.dtype.kind in 'ifc':
                                    vibration_data = flat_arr.astype(np.float32)
                                    break
                        except: continue
                        if vibration_data is not None: break
                # استخراج آرایه مستقیم CWRU
                else:
                    try:
                        flat_arr = np.real(raw).flatten()
                        if len(flat_arr) > 4096 and flat_arr.dtype.kind in 'ifc':
                            vibration_data = flat_arr.astype(np.float32)
                    except: pass

            # نابود کردن کل دیکشنری ماتریس بعد از استخراج و فرار از کرش
            del mat_data
            gc.collect()

            # --- پردازش سیگنال و مدل هوش مصنوعی ---
            if vibration_data is not None:
                st.info(f"📊 Signal loaded successfully. Vector Length: {len(vibration_data)}")
                segment = vibration_data[:4096]
                
                del vibration_data
                gc.collect()
                
                # فرکانس فرضی ۱۲ کیلوهرتز استاندارد
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
                    st.warning("Model core (`bearing_model.h5`) not found.")
            else:
                st.error("Structure Error: No valid vibration data found.")
                
    else:
        # اگر کاربر فایلی را دیسکانکت یا ضربدر زد، کل مموری سشن پاک شود
        st.session_state["current_file"] = None
        gc.collect()

except Exception as e:
    st.error(f"Execution Error: {str(e)}")
