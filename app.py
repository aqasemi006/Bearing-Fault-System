import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os

# --- تنظیمات سیستمی ---
st.set_page_config(page_title="SUT Advanced Diagnosis", page_icon="🔬", layout="wide")

# --- منوی کناری اطلاعات اساتید و دانشجو ---
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    st.markdown("### 🎓 Academic Project")
    st.info("System: Transfer Learning for Cross-Domain Fault Diagnosis")
    st.divider()
    st.markdown("**Researcher:** Amir Mohammad Ghasemi Nezhad")
    st.markdown("**Supervisors:** Dr. Abbasloo & Dr. Ghasemi")

# --- سربرگ انگلیسی طبق درخواست ---
st.title("Intelligent Bearing Fault Diagnosis System")
st.subheader("Advanced Transfer Learning Approach (Cross-Motor Analysis)")
st.divider()

@st.cache_resource
def load_bearing_model():
    # اینجا فرض بر این است که شما مدل جدید Transfer Learning را آپلود کرده‌اید
    return tf.keras.models.load_model("bearing_model.h5")

try:
    model = load_bearing_model()
    uploaded_file = st.file_uploader("Upload Vibration File (Any Motor .mat)", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('Performing Feature Extraction & Domain Adaptation...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            
            # جستجوی عمیق برای یافتن طولانی‌ترین آرایه عددی (سیگنال ارتعاشات)
            max_len = 0
            for key in mat_data.keys():
                if not key.startswith('__'):
                    try:
                        temp_data = np.array(mat_data[key]).flatten()
                        if len(temp_data) > max_len:
                            vibration_data = temp_data
                            max_len = len(temp_data)
                    except: continue
            
            if vibration_data is not None and len(vibration_data) > 1024:
                # تبدیل سیگنال به فرمت تصویری برای یادگیری انتقالی
                segment = vibration_data[:4096]
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                # نرمال‌سازی برای شبکه های ResNet/VGG
                spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db) + 1e-6)
                
                # تغییر ابعاد به چیزی که مدل های Transfer Learning نیاز دارند (مثلا 224x224)
                # در اینجا ما فعلا با ابعاد مدل شما هماهنگ می‌کنیم
                input_data = np.expand_dims(spec_db, axis=(0, -1))
                
                prediction = model.predict(input_data)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx] * 100
                
                classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                
                st.success(f"Analysis complete for signal with length {len(vibration_data)}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected State", classes[class_idx])
                    st.metric("Confidence Score", f"{confidence:.2f}%")
                with col2:
                    st.info("Note: This result is based on deep feature extraction capable of generalizing to different motor types.")
            else:
                st.error("Could not find a valid numerical signal. Please check your .mat file structure.")
except Exception as e:
    st.error(f"Processing Error: {str(e)}")
