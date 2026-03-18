import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. โหลดข้อมูล
df = pd.read_csv('protein_no_duplicates-2.csv')

# 2. เตรียม Feature (X) และสร้าง Target (y) สำหรับ Classification
# เกณฑ์: RMSD < 5 ให้เป็น 0, RMSD >= 5 ให้เป็น 1
features = ['F3', 'F4', 'F2', 'F9']
X = df[features]
y_class = np.where(df['RMSD'] < 5, 0, 1)

# 3. แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 4. สร้าง Model Classifier ที่ถูกจำกัดขนาด
clf_model = RandomForestClassifier(
    n_estimators=50,       # จำนวนต้นไม้
    max_depth=10,          # จำกัดความลึก
    min_samples_leaf=10,   
    random_state=42,
    n_jobs=-1
)

# 5. สร้าง Pipeline (รวม Scaler และ Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', clf_model)
])

# 6. เทรนโมเดล
print("กำลังเทรนโมเดล Classification... กรุณารอครู่หนึ่ง")
pipeline.fit(X_train, y_train)

# 7. บันทึกโมเดลและ Metadata
if not os.path.exists('model_artifacts'):
    os.makedirs('model_artifacts')

model_path = 'model_artifacts/protein_rmsd_model.pkl'
joblib.dump(pipeline, model_path, compress=9)

# บันทึก Metadata เพื่อบอก App ว่าเราใช้ Threshold ที่เท่าไหร่
metadata = {
    "model_type": "Random Forest Classifier",
    "features": features,
    "threshold": 5.0,
    "classes": ["Group 0 (RMSD < 5)", "Group 1 (RMSD >= 5)"]
}
with open('model_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f)

# ตรวจสอบขนาดไฟล์
file_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n--- เสร็จเรียบร้อย ---")
print(f"ไฟล์โมเดล (Classifier) ถูกบันทึกที่: {model_path}")
print(f"ขนาดไฟล์โมเดล: {file_size:.2f} MB")