import streamlit as st
import numpy as np
import joblib
import json
import pandas as pd # เพิ่มสำหรับจัดการข้อมูลขาออก

# ===== 1. ตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Bio-Predictor Pro",
    page_icon="🧬",
    layout="centered"
)

# Custom CSS เพิ่มเติมเพื่อความสวยงาม
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3.5em;
        background-color: #5D9C59;
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(93, 156, 89, 0.2);
    }
    
    /* ตกแต่งส่วนหัวของ Expander */
    .st-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== 2. โหลดโมเดล =====
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model_artifacts/protein_rmsd_model.pkl") 
        with open("model_artifacts/model_metadata.json", "r") as f:
            meta = json.load(f)
        return model, meta
    except:
        return None, None

model, metadata = load_assets()
# กำหนดค่ามาตรฐานเริ่มต้นกรณีโหลด metadata ไม่สำเร็จ
median_val = metadata.get("median_rmsd", 0.5) if metadata else 0.5

# ===== 3. หน้าตาแอป (User Interface) =====

st.title("🧬 ระบบวิเคราะห์โปรตีนอัจฉริยะ")
st.write("เครื่องมือประเมินความแม่นยำของโครงสร้างโปรตีน (RMSD Prediction)")

if not model:
    st.error("❌ ไม่พบโมเดลในระบบ: กรุณาตรวจสอบไฟล์ในโฟลเดอร์ model_artifacts")
    st.stop()

# --- ส่วนที่ 1: คำอธิบายตัวแปร (Expander) ---
with st.expander("❓ ข้อมูลตัวแปรแต่ละตัวคืออะไร?"):
    st.markdown("""
    * **F9 (Feature 9):** ค่าความหนาแน่นของอะตอมในบริเวณแกนกลาง (Core Density)
    * **F6 (Feature 6):** พลังงานพันธะไฮโดรเจนสะสม (Hydrogen Bond Energy)
    * **F1 (Feature 1):** พื้นที่ผิวที่สัมผัสกับตัวทำละลาย (Solvent Accessible Surface Area)
    * **F3 (Feature 3):** ค่าความยืดหยุ่นเฉลี่ยของโครงสร้าง (Average B-factor)
    """)

st.divider()

# --- ส่วนที่ 2: การกรอกข้อมูลและการตรวจสอบ (Validation) ---
st.subheader("📍 ข้อมูลทางชีวภาพ")
col1, col2 = st.columns(2)

with col1:
    f9 = st.number_input("ค่าทดสอบที่ 1 (F9)", value=0.0, format="%.4f")
    f6 = st.number_input("ค่าทดสอบที่ 2 (F6)", value=0.0, format="%.4f")

with col2:
    f1 = st.number_input("ค่าทดสอบที่ 3 (F1)", value=0.0, format="%.4f")
    f3 = st.number_input("ค่าทดสอบที่ 4 (F3)", value=0.0, format="%.4f")

# ฟังก์ชันตรวจสอบความถูกต้องทางชีวภาพ
def is_valid_input(vals):
    if all(v == 0 for v in vals):
        return False, "กรุณากรอกข้อมูลจากแล็บ (ค่าต้องไม่เป็น 0 ทั้งหมด)"
    if any(v < 0 for v in vals):
        return False, "พบค่าที่ติดลบ ซึ่งเป็นค่าที่เป็นไปไม่ได้ทางชีวภาพสำหรับตัวแปรเหล่านี้"
    return True, ""

# --- ส่วนที่ 3: การวิเคราะห์และแสดงผลกราฟิก ---
if st.button("🔍 เริ่มการวิเคราะห์เดี๋ยวนี้"):
    valid, msg = is_valid_input([f9, f6, f1, f3])
    
    if not valid:
        st.error(msg)
    else:
        with st.spinner('กำลังคำนวณโครงสร้าง...'):
            features = np.array([[f9, f6, f1, f3]])
            prediction = model.predict(features)[0]
            
            st.write("")
            st.subheader("📊 ผลการวิเคราะห์")

            # คำนวณเปอร์เซ็นต์สำหรับ Progress Bar (ยิ่ง RMSD น้อย ยิ่งดี)
            # สมมติว่า RMSD เกิน 1.0 คือแย่มาก (แถบเต็ม)
            progress_val = min(prediction / 1.0, 1.0) 
            
            # การเลือกสีและข้อความตามระดับความคลาดเคลื่อน
            if prediction < (median_val * 0.8):
                color = "Green"
                status = "ดีมาก (High Precision)"
                st.success(f"### ✅ {status}")
            elif prediction <= median_val:
                color = "Orange"
                status = "ปกติ (Acceptable)"
                st.warning(f"### ⚠️ {status}")
            else:
                color = "Red"
                status = "คลาดเคลื่อนสูง (Low Precision)"
                st.error(f"### 🚨 {status}")

            # แสดงค่า RMSD และ Progress Bar
            st.metric("ค่าความคลาดเคลื่อน (RMSD)", f"{prediction:.4f}")
            st.progress(progress_val)
            st.caption(f"แถบแสดงระดับความคลาดเคลื่อน: เขียว (ดี) → แดง (คลาดเคลื่อนมาก)")

            # --- ส่วนที่ 4: การจัดการข้อมูลขาออก (Export) ---
            st.divider()
            # สร้าง DataFrame สำหรับ Export
            report_data = {
                "Metric": ["F9", "F6", "F1", "F3", "Predicted RMSD", "Status"],
                "Value": [f9, f6, f1, f3, f"{prediction:.4f}", status]
            }
            df = pd.DataFrame(report_data)
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดผลการวิเคราะห์เป็น CSV",
                data=csv,
                file_name=f'protein_report_{prediction:.2f}.csv',
                mime='text/csv',
            )

# ส่วนท้าย
st.divider()
st.caption("© 2026 Protein Analysis Tool | พัฒนาเพื่อความแม่นยำสูงเพื่อเป็นผู้ช่วยในงานวิจัย")