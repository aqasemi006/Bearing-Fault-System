import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io
from scipy import signal
import os
import cv2

# --- Page Configuration ---
st.set_page_config(
    page_title="SUT Bearing Diagnosis",
    page_icon="⚙️",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    # نمایش لوگوی دانشگاه
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    elif os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

    st.markdown("### 🎓 Academic Project")
    st.info("System: Transfer Learning for Cross-Domain Fault Diagnosis")
    st.divider()
    st.markdown("Designed at Sirjan University of Technology")
    st.markdown("Researcher: Amir Mohammad Ghasemi Nezhad")
    st.markdown("📧 *aqasemi006@gmail.com*")
    st.divider()
    st.markdown("Supervisors:")
    st.markdown("Dr. Aslan Abbasloo  \n 📧 *aslan.abbasloo642@gmail.com*")
    st.markdown("Dr. Morteza Ghasemi  \n 📧 *morteza_ghasemi2010@yahoo.com*")
    st.divider()
    st.markdown("Address: Sirjan University of Technology, Kerman, Iran")

# --- Header Section ---
st.title("Intelligent Bearing Condition Monitoring and Fault Diagnosis System")
st.subheader("Deep Transfer Learning for Induction Motor Health Monitoring")
st.divider()

# --- Load Model ---
@st.cache_resource
def load_bearing_model():
    if os.path.exists("bearing_model.h5"):
        return tf.keras.models.load_model("bearing_model.h5")
    return None

try:
    model = load_bearing_model()

    uploaded_file = st.file_uploader(
        "Upload Vibration File (.mat) from any Motor", type=["mat"]
    )

    if uploaded_file is not None:
        with st.spinner('Analyzing with Transfer Learning Model...'):
            mat_data = scipy.io.loadmat(uploaded_file)
            vibration_data = None

            # جستجوی طولانی‌ترین آرایه عددی
            max_len = 0
            for key in mat_data.keys():
                if not key.startswith('__'):
                    try:
                        raw = mat_data[key]
                        temp_data = np.real(raw).flatten().astype(float)
                        if len(temp_data) > max_len:
                            vibration_data = temp_data
                            max_len = len(temp_data)
                    except:
                        continue

            if vibration_data is not None and len(vibration_data) > 1024:
                # --- Preprocessing ---
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

                    st.success("Analysis Completed Successfully.")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Detected Status", classes[class_idx])
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        if class_idx == 0:
                            st.balloons()
                        else:
                            st.error(f"Alert: {classes[class_idx]} Detected!")

                    with col2:
                        img_file = classes[class_idx].lower().replace(" ", "_") + ".png"
                        if os.path.exists(img_file):
                            st.image(img_file, caption=classes[class_idx], width=300)
                        else:
                            st.info("Status image not found in repository.")
                else:
                    st.error("Model file (bearing_model.h5) not found. Please upload it to GitHub.")
            else:
                st.error("No valid vibration data found in the file.")

except Exception as e:
    st.error(f"System Error: {str(e)}")
