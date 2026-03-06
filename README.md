Project Title การทำนายโครงสร้างตติยภูมิของโปรตีนจากคุณสมบัติทางเคมีกายภาพ Protein Tertiary Structure Prediction

Introduction 
Project Background 
 -โปรตีนนั้นเป็นองค์ประกอบส่วนที่สำคัญในสิ่งมีชีวิต นอกจากนี้โครงสร้าง3มิติ(Tertiary Structure) ของมันเป็นตัวกำหนดหน้าที่การทำงาน
 -สาเหตุทำไมจึงต้องใช้MLมาใช้ปกติการหาโครงสร้างในโปรตีนมักจะหาในห้องแล็บ ซึ่งมักจะใช้เวลานานและมีราคาสูงการใช้ Machine Learning 
 นั้นมาทำนายคุณสมบัติึจึงเป็นทางเลือกที่รวดเร็วสะดวกสบายประหยัดค่าใช้จ่าย

Project Objective
 -เพื่อสร้างแบบจำลองMachine Learning ที่สามารถทำนายค่า RMSD (Root Mean Square Deviation) ของโครงสร้างโปรตีนจากคุณสมบัติทางเคมีกายภาพ 
 เพื่อช่วยลดเวลาและทรัพยากรในการทดลองทางชีววิทยา

Dataset Description
 - Target ที่เราจะทำนาย คือค่า RMSD (Root Mean Square Deviation)
 - Features คือค่า F1-F9
อธิบายค่าแต่ละค่าคืออะไร
 -F1 (Total surface area) พื้นที่ผิวทั้งหมดของโปรตีน
 -F2 (Non polar exposure area) พื้นที่ผิวส่วนที่ไม่ชอบน้ำ (Non-polar) ที่สัมผัสกับสิ่งแวดล้อม
 -F3 (Fractional area of non polar exposure) สัดส่วนของพื้นที่ผิวไม่ชอบน้ำเมื่อเทียบกับพื้นที่ทั้งหมด
 -F4 (Area of polar residue exposure): พื้นที่ผิวส่วนที่ชอบน้ำ (Polar residue) ที่สัมผัสกับสิ่งแวดล้อม
 -F5 (Fractional area of polar residue exposure) สัดส่วนของพื้นที่ผิวชอบน้ำเมื่อเทียบกับพื้นที่ทั้งหมด
 -F6 (Exposed area of residue with non polar side chain) พื้นที่ผิวที่สัมผัสของเรซิดิวที่มีสายโซ่ข้างไม่ชอบน้ำ
 -F7 (Fractional area of residue with non polar side chain) สัดส่วนพื้นที่ผิวของเรซิดิวที่มีสายโซ่ข้างไม่ชอบน้ำ
 -F8 (Number of non-polar residues) จำนวนเรซิดิวที่ไม่ชอบน้ำทั้งหมดในสายโซ่
 -F9 (Molecular weight) น้ำหนักโมเลกุลโดยรวม
 (หรือในบางกรณีอาจหมายถึงค่าความลึกโดยเฉลี่ยของเรซิดิว ขึ้นอยู่กับแหล่งที่มาของข้อมูลดิบ แต่ส่วนใหญ่มักใช้เป็นคุณสมบัติที่เกี่ยวข้องกับขนาดโมเลกุล)
