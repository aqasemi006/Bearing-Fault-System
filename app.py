import os
import io
import numpy as np
import scipy.io as sio
import tensorflow as tf
import streamlit as st
import matplotlib
# جلوگیری از خطای مانیتور و لود گرافیکی در سرورهای بدون رابط بصری (Headless)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ۱. تنظیمات ظاهری و راست‌چین کردن صفحه برای زبان فارسی
st.set_page_config(
    page_title="سامانه هوشمند پایش وضعیت بیرینگ",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
    .reportview-container { direction: rtl; text-align: right; }
    h1, h2, h3, h4, p, div { text-align: right; font-family: 'Tahoma', sans-serif; }
    .stAlert { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚙️ سامانه هوشمند و تحت وب پایش وضعیت و عیب‌یابی ماشین‌آلات دوار")
st.subheader("تحلیل فرکانسی و پایش عیوب بیرینگ بر پایه یادگیری انتقالی عمیق ۲ بعدی (CWRU & Paderborn)")
st.write("---")

# ۲. تنظیمات ابعادی پروژه هماهنگ با مدل ۲بعدی MobileNetV2
SIGNAL_LENGTH = 2304
IMG_ROWS, IMG_COLS = 48, 48
class_names = ['Ball_Fault', 'Healthy', 'Inner_Race', 'Outer_Race']

# ۳. بارگذاری بهینه مدل با مکانیزم Cache (برای جلوگیری از لود مکرر و کندی سرور)
@st.cache_resource
def load_bearing_model():
    model_path = 'bearing_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_bearing_model()

if model is None:
    st.error("❌ خطا: فایل مدل عمیق 'bearing_model.keras' در مخزن گیت‌هاب یافت نشد! لطفا فایل مدل (12 مگابایتی) را در کنار این کد آپلود کنید.")
    st.stop()
else:
    st.success("✅ مدل شبکه عصبی عمیق ۲ بعدی با موفقیت در سرور ابری بارگذاری شد و آماده پردازش است.")

# ۴. بخش دریافت فایل متلب (.mat) از کاربر
st.write("### 📬 بارگذاری سیگنال ارتعاشی بیرینگ")
uploaded_file = st.file_uploader("لطفاً فایل متلب بیرینگ (.mat) را انتخاب یا اینجا رها کنید:", type=["mat"])

if uploaded_file is not None:
    st.info("📊 در حال اسکن عمیق و لایه‌برداری از ساختار فایل متلب در سرور...")
    
    try:
        # خواندن مستقیم بایت‌های فایل از حافظه موقت (بدون نیاز به ذخیره روی هارد سرور)
        file_bytes = io.BytesIO(uploaded_file.read())
        mat_contents = sio.loadmat(file_bytes)
        raw_signal = None
        
        keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        
        # الگوریتم جامع هوشمند برای تفکیک خودکار دیتاست‌ها
        for key in keys:
            data = mat_contents[key]
            
            # الف) ساختار چند لایه پادربورن
            if data.dtype.names is not None:
                if 'Y' in data.dtype.names:
                    y_data = data['Y'][0, 0]
                    if y_data.dtype.names is not None:
                        for sub_name in y_data.dtype.names:
                            sub_array = y_data[sub_name]
                            if isinstance(sub_array, np.ndarray) and sub_array.size > 5000:
                                raw_signal = sub_array.flatten()
                                break
                    else:
                        if isinstance(y_data, np.ndarray) and y_data.size > 5000:
                            raw_signal = y_data.flatten()
                
                if raw_signal is None:
                    for name in data.dtype.names:
                        sub_field = data[name].flatten()
                        for item in sub_field:
                            if isinstance(item, np.ndarray) and item.size > 5000:
                                raw_signal = item.flatten()
                                break
                        if raw_signal is not None: break

            # ب) ساختار ساده و مستقیم CWRU
            elif isinstance(data, np.ndarray) and data.size > 5000:
                raw_signal = data.flatten()
                break

        # ج) استراتژی بازگشتی عمیق (پشتیبان نهایی برای پادربورن‌های پیچیده)
        if raw_signal is None:
            for key in keys:
                def extract_deep(obj):
                    if isinstance(obj, np.ndarray):
                        if obj.size > 5000 and obj.dtype.kind in 'iuf':
                            return obj.flatten()
                        if obj.dtype.names is not None:
                            for n in obj.dtype.names:
                                res = extract_deep(obj[n])
                                if res is not None: return res
                        for i in range(min(len(obj), 5)):
                            res = extract_deep(obj[i])
                            if res is not None: return res
                    return None
                raw_signal = extract_deep(mat_contents[key])
                if raw_signal is not None: break

        if raw_signal is None:
            st.error("❌ خطا: سیستم نتوانست سیگنال عددی معتبری (بالای ۵۰۰۰ نقطه) از درون این فایل متلب استخراج کند.")
            st.stop()

        total_points = len(raw_signal)
        st.success(f"🎯 استخراج سیگنال با موفقیت انجام شد! تعداد نقاط داده: {total_points} نقطه.")

        # ۵. بخش‌بندی سیگنال (Sliding Window) و استنتاج مدل ۲بعدی
        num_segments = total_points // SIGNAL_LENGTH
        all_predictions = []
        
        for i in range(num_segments):
            segment = raw_signal[i*SIGNAL_LENGTH : (i+1)*SIGNAL_LENGTH]
            
            # نرمال‌سازی سیگنال
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            if std_val == 0: std_val = 1e-6
            normalized_segment = (segment - mean_val) / std_val
            
            # تبدیل به ماتریکس ۲بعدی و اعمال ۳ کانال رنگی برای MobileNetV2
            matrix_2d = normalized_segment.reshape(IMG_ROWS, IMG_COLS)
            img_rgb = np.stack([matrix_2d, matrix_2d, matrix_2d], axis=-1)
            input_tensor = np.expand_dims(img_rgb, axis=0)
            
            pred = model.predict(input_tensor, verbose=0)
            all_predictions.append(pred[0])
            
        # میانگین‌گیری نهایی برای پایداری خروجی
        mean_predictions = np.mean(all_predictions, axis=0)
        predicted_class_idx = np.argmax(mean_predictions)
        predicted_class_name = class_names[predicted_class_idx]
        confidence = mean_predictions[predicted_class_idx] * 100

        # ۶. نمایش ویژوال و شکیل نتایج نهایی ارزیابی
        st.write("---")
        st.write("## 🎯 نتایج ارزیابی و پایش وضعیت عیب")
        
        col1, col2 = st.columns(2)
        with col1:
            if predicted_class_name == 'Healthy':
                st.metric(label="وضعیت سلامت بیرینگ", value="سالم (HEALTHY)", delta="بدون عیب")
            else:
                st.metric(label="عیب شناسایی‌شده", value=predicted_class_name.upper(), delta="- خطر آسیب مکانیکی", delta_color="inverse")
                
        with col2:
            st.metric(label="میزان قطعیت و پایداری مدل", value=f"{confidence:.2f} %")

        # ۷. رسم و نمایش نمودارهای مهندسی در صفحه وب
        st.write("### 📊 تحلیل گرافیکی سیگنال ارتعاشی")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
        
        # نمودار اول: شکل موج زمانی سیگنال
        ax1.plot(raw_signal[:SIGNAL_LENGTH * 2], color='#d62728' if predicted_class_name != 'Healthy' else '#2ca02c', linewidth=0.8)
        ax1.set_title(f"Time Domain Signal (First 2 Segments) - {uploaded_file.name}", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Data Points")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # نمودار دوم: توزیع درصد احتمالات عیوب
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
        bars = ax2.bar(class_names, mean_predictions * 100, color=colors)
        ax2.set_title("Fault Probability Distribution (%)", fontsize=10, fontweight='bold')
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 120)
        ax2.grid(True, linestyle=':', alpha=0.5, axis='y')
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # بستن شکل برای جلوگیری از پر شدن حافظه رم سرور ابری

    except Exception as e:
        st.error(f"❌ خطا در پردازش یا لایه‌برداری فایل متلب: {e}")
