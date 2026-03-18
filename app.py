import streamlit as st
import numpy as np
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Bio-Classification Pro",
    page_icon="🧬",
    layout="centered"
)

# Custom CSS สำหรับปรับแต่งฟอนต์และปุ่ม
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .stButton>button {
        width: 100%; border-radius: 25px; height: 3.5em;
        background-color: #5D9C59; color: white; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== 2. โหลดโมเดล (แก้ไขเส้นทางไฟล์ให้ตรงกับ GitHub) =====
@st.cache_resource
def load_assets():
    try:
        # แก้ไขจาก 'model_artifacts/...' เป็นชื่อไฟล์ตรงๆ ตามที่เห็นใน GitHub
        model = joblib.load('protein_rmsd_model.pkl')
        with open("model_metadata.json", "r") as f:
            meta = json.load(f)
        return model, meta
    except Exception as e:
        # คืนค่า Error กลับมาดูว่าติดปัญหาอะไร (ถ้ามี)
        return None, str(e)

model, metadata = load_assets()

# ===== 3. ส่วน UI =====
st.title("🧬 ระบบจำแนกกลุ่มโปรตีน")
st.write("วิเคราะห์กลุ่มโครงสร้าง (Threshold: RMSD = 5)")

if model is None:
    st.error(f"❌ ไม่พบไฟล์โมเดลใน Repository: {metadata}")
    st.info("ตรวจสอบว่าไฟล์ protein_rmsd_model.pkl และ model_metadata.json อยู่ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    f3 = st.number_input("F3 (Non polar area)", value=0.0, format="%.4f")
    f4 = st.number_input("F4 (Polar residue)", value=0.0, format="%.4f")
with col2:
    f2 = st.number_input("F2 (Non polar exposure)", value=0.0, format="%.4f")
    f9 = st.number_input("F9 (Molecular weight depth)", value=0.0, format="%.4f")

if st.button("🔍 เริ่มการจำแนกกลุ่ม"):
    if all(v == 0 for v in [f3, f4, f2, f9]):
        st.warning("กรุณากรอกข้อมูลก่อนวิเคราะห์")
    else:
        with st.spinner('กำลังประมวลผล...'):
            features = np.array([[f3, f4, f2, f9]])
            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0] 
            
            st.divider()
            
            # --- แสดงผลลัพธ์หลัก ---
            if prediction == 0:
                st.success(f"### ผลลัพธ์: กลุ่ม 0 (RMSD < 5)")
                st.info(f"ความมั่นใจ: {prob[0]*100:.2f}%")
            else:
                st.error(f"### ผลลัพธ์: กลุ่ม 1 (RMSD >= 5)")
                st.warning(f"ความมั่นใจ: {prob[1]*100:.2f}%")

            # --- ส่วนกราฟแท่ง ---
            st.subheader("📊 กราฟเปรียบเทียบความน่าจะเป็น")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            classes = ['Group 0\n(RMSD < 5)', 'Group 1\n(RMSD >= 5)']
            colors = ['#5D9C59', '#D24545']
            
            bars = ax.bar(classes, prob, color=colors, alpha=0.8)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Probability')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)

            # --- Export CSV ---
            report_data = {
                "Parameter": ["F3", "F4", "F2", "F9", "Prediction", "Prob_0", "Prob_1"],
                "Value": [f3, f4, f2, f9, int(prediction), f"{prob[0]:.4f}", f"{prob[1]:.4f}"]
            }
            csv = pd.DataFrame(report_data).to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 ดาวน์โหลดผลการวิเคราะห์", data=csv, file_name='result.csv', mime='text/csv')