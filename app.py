import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os
import gc  # اضافه شده برای مدیریت بهتر حافظه در گوشی

# --- تنظیمات صفحه ---
st.set_page_config(page_title="SUT Bearing Diagnosis", page_icon="⚙️", layout="wide")

# --- استایل‌دهی ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stAlert { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- منوی سمت چپ (Sidebar) ---
with st.sidebar:
    # عکس لوگوی دانشگاه (می‌توانی به جای این لینک، بنویسی "logo.jpg" و عکس را در گیت‌هاب آپلود کنی)
    st.image("logo.jpg", use_container_width=True)
    
    st.markdown("### 🎓 About Project")
    st.markdown("Designed at Sirjan University of Technology")
    st.divider()
    
    st.markdown("Designed by:")
    st.markdown("Amir Mohammad Ghasemi Nezhad")
    st.markdown("📧 *aqasemi006@gmail.com*")
    st.text("") # فاصله خالی
    
    st.markdown("Supervisors:")
    st.markdown("Dr. Aslan Abbasloo")
    st.markdown("📧 *aslan.abbasloo642@gmail.com*")
    st.text("")
    
    st.markdown("Dr. Morteza Ghasemi")
    st.markdown("📧 *morteza_ghasemi2010@yahoo.com*")
    st.divider()
    
    st.markdown("University Address:")
    st.markdown("Sirjan University of Technology, Sirjan, Kerman, Iran")

# --- بدنه اصلی سایت ---
st.warning("⚠️ نکته مهم: جهت پایداری کامل و جلوگیری از خطای آپلود فایل‌های حجیم، استفاده از مرورگر کامپیوتر (PC) یا فعال‌سازی حالت 'Desktop Site' در موبایل پیشنهاد می‌شود.")

st.title("🛠 سامانه هوشمند پایش وضعیت و عیب‌یابی بیرینگ")
st.write("پروژه پایان‌نامه کارشناسی ارشد - مبتنی بر شبکه‌های عصبی عمیق")
st.divider()

# --- بارگذاری مدل ---
@st.cache_resource
def load_bearing_model():
    if os.path.exists("bearing_model.h5"):
        return tf.keras.models.load_model("bearing_model.h5")
    return None

model = load_bearing_model()

if model is None:
    st.error("❌ فایل مدل (bearing_model.h5) پیدا نشد! لطفا بررسی کنید که فایل در گیت‌هاب آپلود شده باشد.")
else:
    uploaded_file = st.file_uploader("فایل ارتعاشات (.mat) را انتخاب کنید", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('در حال پردازش سیگنال و تحلیل هوشمند...'):
            try:
                # خواندن داده
                mat_data = scipy.io.loadmat(uploaded_file)
                vibration_data = None
                for key in mat_data.keys():
                    if 'DE_time' in key or 'FE_time' in key:
                        vibration_data = mat_data[key].flatten()
                        break
                
                # خالی کردن حافظه رم از متغیرهای اضافی (برای پایداری در گوشی)
                del mat_data
                gc.collect()
                
                if vibration_data is not None:
                    # پردازش سیگنال
                    segment = vibration_data[:4096]
                    f, t, Sxx = signal.spectrogram(segment, fs=12000)
                    spec_db = 10 * np.log10(Sxx + 1e-10)
                    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
                    input_data = np.expand_dims(spec_db, axis=(0, -1))
                    
                    # پیش‌بینی
                    prediction = model.predict(input_data)
                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx] * 100
                    
                    classes = ["Healthy (سالم)", "Inner Race Fault (خرابی رینگ داخلی)", 
                               "Outer Race Fault (خرابی رینگ خارجی)", "Ball Fault (خرابی ساچمه)"]
                    
                    # نمایش نتایج
                    st.success("✅ تحلیل با موفقیت انجام شد.")
                    c1, c2 = st.columns(2)
                    c1.metric("وضعیت شناسایی شده", classes[class_idx])
                    c2.metric("درصد اطمینان", f"{confidence:.2f}%")
                    
                    if class_idx == 0:
                        st.balloons()
                        st.info("💡 بیرینگ در شرایط نرمال کار می‌کند.")
                    else:
                        st.error(f"⚠️ هشدار: عیب در بخش {classes[class_idx]} شناسایی شد. نیاز به بررسی فنی!")
                        
                    # پاکسازی نهایی حافظه
                    del vibration_data, segment, input_data
                    gc.collect()

                else:
                    st.error("❌ متغیر ارتعاش (DE_time یا FE_time) در فایل پیدا نشد.")
            except Exception as e:
                st.error(f"❌ خطا در پردازش فایل. لطفا فایل دیگری را امتحان کنید. متن خطا: {e}")


