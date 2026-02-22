import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os

# --- Page Config ---
st.set_page_config(page_title="SUT Bearing Diagnosis", page_icon="⚙️", layout="wide")

# --- Sidebar (English Info) ---
with st.sidebar:
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    elif os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
        
    st.markdown("### 🎓 About Project")
    st.markdown("**Designed at Sirjan University of Technology**")
    st.divider()
    st.markdown("**Designed by:** \n Amir Mohammad Ghasemi Nezhad  \n 📧 *aqasemi006@gmail.com*")
    st.markdown("**Supervisors:** \n Dr. Aslan Abbasloo  \n 📧 *aslan.abbasloo642@gmail.com* \n\n Dr. Morteza Ghasemi  \n 📧 *morteza_ghasemi2010@yahoo.com*")
    st.divider()
    st.markdown("**University Address:** \n Sirjan University of Technology, Sirjan, Kerman, Iran")

# --- Header Section (English) ---
st.title("Intelligent Bearing Condition Monitoring and Fault Diagnosis System")
st.subheader("Based on Deep Neural Networks")
st.divider()

# --- Warning (Persian) ---
st.warning("⚠️ جهت عملکرد سریع و پایداری در آپلود، ترجیحاً از مرورگر کامپیوتر یا حالت Desktop Site موبایل استفاده کنید.")

# --- Load Model ---
@st.cache_resource
def load_bearing_model():
    return tf.keras.models.load_model("bearing_model.h5")

try:
    model = load_bearing_model()
    
    uploaded_file = st.file_uploader("Select Vibration File (.mat)", type=["mat"])

    if uploaded_file is not None:
        with st.spinner('Analyzing...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None
            for key in mat_data.keys():
                if 'DE_time' in key or 'FE_time' in key:
                    vibration_data = mat_data[key].flatten()
                    break
            
            if vibration_data is not None:
                segment = vibration_data[:4096]
                f, t, Sxx = signal.spectrogram(segment, fs=12000)
                spec_db = 10 * np.log10(Sxx + 1e-10)
                spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
                input_data = np.expand_dims(spec_db, axis=(0, -1))
                
                prediction = model.predict(input_data)
                class_idx = np.argmax(prediction)
                confidence = prediction[0][class_idx] * 100
                
                classes = ["Healthy", "Inner Race Fault", "Outer Race Fault", "Ball Fault"]
                classes_fa = ["سالم", "خرابی رینگ داخلی", "خرابی رینگ خارجی", "خرابی ساچمه"]
                img_files = ["healthy.png", "inner_race.png", "outer_race.png", "ball.png"]

                st.success("Analysis Completed Successfully.")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Detected Status", classes[class_idx])
                    st.metric("Confidence", f"{confidence:.2f}%")
                    if class_idx == 0:
                        st.balloons()
                    else:
                        st.error(f"Alert: {classes_fa[class_idx]} شناسایی شد")
                
                with col2:
                    # نمایش عکس مربوط به خرابی
                    if os.path.exists(img_files[class_idx]):
                        st.image(img_files[class_idx], caption=classes[class_idx], width=300)
                    else:
                        st.info("No status image found in repository.")
            else:
                st.error("No vibration data found in the file.")
except Exception as e:
    st.error(f"Error: {e}")
