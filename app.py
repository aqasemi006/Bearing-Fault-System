import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os

# --- تنظیمات صفحه ---
st.set_page_config(page_title="Bearing Fault Diagnosis", page_icon="⚙️")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛠 سامانه هوشمند پایش وضعیت بیرینگ")
st.write("طراحی شده توسط: امیرمحمد قاسمی نژاد")
st.divider()

# --- بارگذاری مدل ---
@st.cache_resource
def load_bearing_model():
    if os.path.exists("bearing_model.h5"):
        return tf.keras.models.load_model("bearing_model.h5")
    return None

model = load_bearing_model()

if model is None:
    st.error("❌ فایل مدل (bearing_model.h5) پیدا نشد! ابتدا مدل را آموزش دهید.")
else:
    # --- بخش آپلود فایل ---
    uploaded_file = st.file_uploader("فایل ارتعاشات (.mat) را انتخاب کنید", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('در حال تحلیل سیگنال و عیب‌یابی...'):
            try:
                # ۱. خواندن داده از فایل متلب
                mat_data = scipy.io.loadmat(uploaded_file)
                vibration_data = None
                for key in mat_data.keys():
                    if 'DE_time' in key or 'FE_time' in key:
                        vibration_data = mat_data[key].flatten()
                        break
                
                if vibration_data is not None:
                    # ۲. پیش‌پردازش (اسپکتروگرام)
                    segment = vibration_data[:4096]
                    f, t, Sxx = signal.spectrogram(segment, fs=12000)
                    spec_db = 10 * np.log10(Sxx + 1e-10)
                    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
                    
                    # ۳. آماده‌سازی برای مدل (1, 129, 18, 1)
                    input_data = np.expand_dims(spec_db, axis=(0, -1))
                    
                    # ۴. پیش‌بینی
                    prediction = model.predict(input_data)
                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx] * 100
                    
                    classes = ["Healthy (سالم)", "Inner Race Fault (خرابی رینگ داخلی)", 
                               "Outer Race Fault (خرابی رینگ خارجی)", "Ball Fault (خرابی ساچمه)"]
                    
                    # --- نمایش نتایج ---
                    st.success("✅ تحلیل با موفقیت انجام شد.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("وضعیت شناسایی شده", classes[class_idx])
                    with col2:
                        st.metric("درصد اطمینان", f"{confidence:.2f}%")

                    # نمایش هشدار بر اساس نتیجه
                    if class_idx == 0:
                        st.balloons()
                        st.info("💡 بیرینگ در شرایط نرمال کار می‌کند.")
                    else:
                        st.warning(f"⚠️ هشدار: عیب در بخش {classes[class_idx]} شناسایی شد. نیاز به بررسی فنی!")

                else:
                    st.error("❌ متغیر ارتعاش در فایل پیدا نشد.")
            except Exception as e:
                st.error(f"❌ خطا در پردازش فایل: {e}")