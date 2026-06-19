import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# ۱. تنظیمات ظاهری صفحه وب‌اپلیکیشن (مخصوص نسخه آنلاین)
st.set_page_config(
    page_title="سیستم هوشمند عیب‌یابی بلبرینگ", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تعریف کلاس‌ها و ایموجی‌ها به ترتیب آموزش مدل
SIGNAL_LENGTH = 2048
class_mapping = {
    0: ('Ball_Fault (عیب ساچمه/بال)', '🔴'),
    1: ('Healthy (وضعیت سالم)', '🟢'),
    2: ('Inner_Race (عیب رینگ داخلی)', '🔵'),
    3: ('Outer_Race (عیب رینگ خارجی)', '🟠')
}

st.title("🎯 سامانه آنلاین پایش وضعیت و عیب‌یابی بلبرینگ")
st.write("این وب‌اپلیکیشن مجهز به شبکه عصبی عمیق 1D-CNN است که سیگنال‌های ارتعاشی حسگرها را از روی فایل‌های متلب (.mat) تحلیل می‌کند.")
st.markdown("---")

# ۲. بارگذاری بهینه مدل هوش مصنوعی با قابلیت کش در سرور ابری
@st.cache_resource
def load_bearing_model():
    # پیدا کردن خودکار مسیر فایل مدل در مخزن گیت‌هاب شما
    model_path = 'bearing_model.keras'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری ساختار مدل روی سرور: {e}")
            return None
    return None

model = load_bearing_model()

if model is None:
    st.warning("⚠️ فایل مدل 'bearing_model.keras' در مخزن گیت‌هاب شما پیدا نشند یا به دلیل حجم بالا کامل آپلود نشده است. لطفاً مطمئن شوید فایل مدل در کنار همین اسکریپت در گیت‌هاب قرار دارد.")
else:
    st.success("✅ مدل هوشمند با موفقیت روی سرور ابری بارگذاری و آماده کار شد.")

    # ۳. باکس آپلود فایل متلب توسط کاربر در سایت
    uploaded_file = st.file_uploader("لطفاً فایل متلب (.mat) سیگنال ارتعاشی بلبرینگ را آپلود کنید:", type=["mat"])

    if uploaded_file is not None:
        try:
            # ۴. خواندن فایل متلب مستقیم از حافظه بافر مرورگر (بدون نیاز به ذخیره روی هارد سرور)
            mat_contents = sio.loadmat(uploaded_file)
            
            # استخراج خودکار متغیر سیگنال عددی درون فایل متلب
            data_key = [k for k in mat_contents.keys() if not k.startswith('__')][0]
            raw_signal = mat_contents[data_key].flatten()

            if len(raw_signal) < SIGNAL_LENGTH:
                st.error(f"❌ طول سیگنال فایل آپلود شده کمتر از {SIGNAL_LENGTH} است و شبکه نمی‌تواند آن را پردازش کند.")
            else:
                st.info(f"📊 فایل با موفقیت در سرور پردازش شد. تعداد کل نقاط داده: {len(raw_signal)}")

                # ۵. رسم نمودار سیگنال در دامنه زمان
                st.subheader("📈 نمودار سیگنال ارتعاشی در دامنه زمان (تایم دامین)")
                segment = raw_signal[0:SIGNAL_LENGTH]
                
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(segment, color='#1f77b4', linewidth=1)
                ax.set_xlabel("Time Samples")
                ax.set_ylabel("Amplitude")
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

                # ۶. نرمال‌سازی سیگنال در لحظه (دقیقاً متناسب با زمان آموزش)
                segment_normalized = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
                input_data = np.expand_dims(segment_normalized, axis=0)
                input_data = np.expand_dims(input_data, axis=-1)

                # ۷. اجرای عملیات استنتاج و پیش‌بینی توسط هوش مصنوعی
                with st.spinner('در حال تحلیل الگوریتم‌های فرکانسی توسط شبکه ۱ بعدی...'):
                    predictions = model.predict(input_data)
                    predicted_class_idx = np.argmax(predictions[0])
                    class_name, emoji = class_mapping[predicted_class_idx]
                    confidence = predictions[0][predicted_class_idx] * 100

                st.markdown("---")
                st.subheader("🎯 نتیجه تحلیل آنلاین هوش مصنوعی")
                # نمایش خروجی نهایی متناسب با نوع عیب
                if predicted_class_idx == 1:
                    st.balloons() # افکت انیمیشنی برای وضعیت بدون عیب
                    st.success(f"### {emoji} وضعیت بلبرینگ: {class_name} | درصد اطمینان مدل: {confidence:.2f}%")
                else:
                    st.error(f"### {emoji} وضعیت بلبرینگ: {class_name} | درصد اطمینان مدل: {confidence:.2f}%")

                # ۸. رسم نمودار توزیع احتمال عیوب به صورت درصد
                st.subheader("📊 توزیع تفکیکی احتمال عیوب")
                labels = [class_mapping[i][0].split(' ')[0] for i in range(4)]
                probabilities = [predictions[0][i] * 100 for i in range(4)]

                fig2, ax2 = plt.subplots(figsize=(8, 3))
                bars = ax2.barh(labels, probabilities, color=['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
                ax2.set_xlim(0, 100)
                ax2.set_xlabel('Probability (%)')
                
                # قرار دادن درصدها روی لبه میله‌های نمودار
                for bar in bars:
                    width = bar.get_width()
                    ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                             va='center', ha='left', fontsize=9)
                
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ خطا در ساختار داخلی فایل متلب آپلود شده: {e}")
