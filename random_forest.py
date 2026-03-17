import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. โหลดข้อมูล
df = pd.read_csv('protein_no_duplicates-2.csv')

# 2. เตรียม Feature (X) และ Target (y)
features = ['F9', 'F6', 'F1', 'F3']
X = df[features]
y_reg = df['RMSD']
median_val = y_reg.median()

# 3. แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# 4. สร้าง Model ที่ถูกจำกัดขนาด (เพื่อให้ไฟล์เล็กกว่า 25MB)
# ปรับ n_estimators และ max_depth เพื่อคุมขนาดไฟล์
reg_model = RandomForestRegressor(
    n_estimators=50,       # ลดจำนวนต้นไม้เหลือ 50 ต้น
    max_depth=10,          # จำกัดความลึกไม่เกิน 10 ชั้น
    min_samples_leaf=10,   # เพิ่มค่านี้เพื่อให้กิ่งก้านน้อยลง ไฟล์จะเล็กลงมาก
    random_state=42,
    n_jobs=-1
)

# 5. สร้าง Pipeline (รวม Scaler และ Model เข้าด้วยกัน)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', reg_model)
])

# 6. เทรนโมเดล
print("กำลังเทรนโมเดล... กรุณารอครู่หนึ่ง")
pipeline.fit(X_train, y_train)

# 7. บันทึกโมเดลและ Metadata
if not os.path.exists('model_artifacts'):
    os.makedirs('model_artifacts')

# บันทึก Pipeline ด้วยการบีบอัดสูงสุด (compress=9)
model_path = 'model_artifacts/protein_rmsd_model.pkl'
joblib.dump(pipeline, model_path, compress=9)

# บันทึก Metadata
metadata = {
    "model_type": "Random Forest Regressor (Optimized Size)",
    "features": features,
    "median_rmsd": float(median_val)
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)

# ตรวจสอบขนาดไฟล์หลังบันทึก
file_size = os.path.getsize(model_path) / (1024 * 1024) # แปลงเป็น MB
print(f"\n--- เสร็จเรียบร้อย ---")
print(f"ไฟล์โมเดลถูกบันทึกที่: {model_path}")
print(f"ขนาดไฟล์โมเดล: {file_size:.2f} MB")

if file_size > 25:
    print("⚠️ คำเตือน: ไฟล์ยังใหญ่เกิน 25MB ลองลด n_estimators หรือ max_depth ลงอีก")
else:
    print("✅ เยี่ยม! ไฟล์มีขนาดเหมาะสมสำหรับอัปโหลดแล้ว")