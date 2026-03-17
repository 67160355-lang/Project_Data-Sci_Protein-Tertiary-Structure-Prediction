Protein Tertiary Structure Prediction
การทำนายโครงสร้างตติยภูมิของโปรตีนจากคุณสมบัติทางเคมีกายภาพ
 Introduction & Background
โปรตีนเป็นองค์ประกอบพื้นฐานที่สำคัญของสิ่งมีชีวิต โดยโครงสร้างสามมิติ (Tertiary Structure) คือตัวกำหนดหน้าที่ (Function) การทำงานของโปรตีนนั้นๆ โปรตีนแต่ละชนิดจะมีลักษณะที่แตกต่างกันออกไป แต่สิ่งมีชีวิตทุกชนิดล้วนมีโปรตีน

โดยปกติแล้ว การหาโครงสร้างที่แม่นยำมักทำในห้องปฏิบัติการ (เช่น X-ray Crystallography หรือ NMR) ซึ่งมีข้อจำกัดคือ:

ใช้เวลานาน อาจใช้เวลาหลายเดือนหรือเป็นปีต่อหนึ่งโครงสร้าง

ค่าใช้จ่ายสูง ต้องใช้อุปกรณ์และสารเคมีที่มีราคาสูงซึ่งทำให้บุคคลธรรมดาส่วนใหญ่ในประเทศไทยไม่สามารถเข้าถึงได้

ความซับซ้อน โปรตีนบางชนิดจัดเตรียมรูปผลึกได้ยากทำให้เกิดความยุ่งยากเสียเวลา

การนำ Machine Learning มาใช้จึงเป็นทางเลือกที่มีประสิทธิภาพสูง ช่วยในการทำนายคุณสมบัติโครงสร้างได้อย่างรวดเร็ว แม่นยำ และประหยัดทรัพยากรเป็นอย่างมาก

 Project Objective
เพื่อสร้างแบบจำลอง Machine Learning (Regression Model) ที่สามารถทำนายค่า RMSD (Root Mean Square Deviation) ของโครงสร้างโปรตีน โดยอาศัยคุณสมบัติทางเคมีกายภาพ (Physicochemical Properties) เพื่อลดขั้นตอนและเวลาในการทดลองทางชีววิทยา

 Dataset Description
ข้อมูลประกอบด้วยคุณสมบัติทางกายภาพของโปรตีน (Features) และค่าความเบี่ยงเบนของโครงสร้าง (Target)

Target Variable
RMSD (Root Mean Square Deviation): ค่าที่ใช้ระบุความแตกต่างระหว่างโครงสร้างที่ทำนายกับโครงสร้างจริง ยิ่งค่า RMSD ต่ำ แสดงว่าโครงสร้างมีความใกล้เคียงกับธรรมชาติมากเท่านั้น

Features (F1 - F9)
Feature	ชื่อเต็ม	คำอธิบาย
F1	Total surface area	พื้นที่ผิวทั้งหมดของโปรตีน
F2	Non polar exposure area	พื้นที่ผิวส่วนที่ไม่ชอบน้ำ (Non-polar) ที่สัมผัสกับสิ่งแวดล้อม
F3	Fractional area of non polar exposure	สัดส่วนพื้นที่ผิวไม่ชอบน้ำเมื่อเทียบกับพื้นที่ทั้งหมด
F4	Area of polar residue exposure	พื้นที่ผิวส่วนที่ชอบน้ำ (Polar residue) ที่สัมผัสกับสิ่งแวดล้อม
F5	Fractional area of polar residue exposure	สัดส่วนพื้นที่ผิวชอบน้ำเมื่อเทียบกับพื้นที่ทั้งหมด
F6	Exposed area of residue with non polar side chain	พื้นที่ผิวสัมผัสของเรซิดิวที่มีสายโซ่ข้างไม่ชอบน้ำ
F7	Fractional area of residue with non polar side chain	สัดส่วนพื้นที่ผิวของเรซิดิวที่มีสายโซ่ข้างไม่ชอบน้ำ
F8	Number of non-polar residues	จำนวนเรซิดิวที่ไม่ชอบน้ำทั้งหมดในสายโซ่
F9	Molecular weight	น้ำหนักโมเลกุลโดยรวม (หรือความลึกโดยเฉลี่ยของเรซิดิว)
 Model & Methodology
โปรเจกต์นี้เลือกใช้

Algorithm: RandomForestRegressor

Preprocessing: StandardScaler (การทำ Feature Scaling)

Pipeline: รวมขั้นตอนการ Scale และการ Train เข้าด้วยกันเพื่อป้องกัน Data Leakage

Optimization: จำกัดความลึกของต้นไม้ (max_depth=10) และจำนวนต้นไม้ (n_estimators=50) เพื่อให้โมเดลมีขนาดกะทัดรัดและทำงานได้รวดเร็ว

  How to Use
เตรียมข้อมูล: ตรวจสอบให้มั่นใจว่ามีไฟล์ protein_no_duplicates-2.csv

ติดตั้ง Library: ```bash
pip install pandas numpy scikit-learn joblib

Train Model: รันไฟล์ Python เพื่อสร้างไฟล์โมเดล

Bash
python random_forest.py
Artifacts: คุณจะได้ไฟล์ model_artifacts/protein_rmsd_model.pkl สำหรับนำไปใช้งานต่อ
