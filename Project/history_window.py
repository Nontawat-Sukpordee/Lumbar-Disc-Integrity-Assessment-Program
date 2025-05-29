import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QPushButton, QHBoxLayout, QLabel, QSizePolicy, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal  
from PyQt6.QtGui import QFont
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class CombinedChartWindow(QMainWindow): #ส่วนสำหรับแสดงตัวกราฟอายุและเพศ
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📊 Gender & Age Distribution")
        self.resize(1366, 768)

        # สร้าง Figure ที่มี 2 subplot ในแนวนอน (1 แถว, 2 คอลัมน์)
        self.figure, (self.ax_gender, self.ax_age) = plt.subplots(1, 2, figsize=(14, 7))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_chart_data()

    def load_chart_data(self): #โหลดข้อมูลจากDatabaseมาสำหรับทำกราฟวงกลม แล้วทำกราฟวกลม
        client = MongoClient('mongodb://localhost:27017/')
        db = client['dicom_database']
        collection = db['dicom_images']

        records = list(collection.find({}, {"Gender": 1, "Age": 1, "_id": 0}))

        # --- กราฟเพศ ---
        gender_counts = {"Male": 0, "Female": 0}
        for record in records:
            gender = record.get("Gender", "Unknown")
            if gender in gender_counts:
                gender_counts[gender] += 1

        self.ax_gender.clear()
        labels_gender = ["Male", "Female"]
        sizes_gender = [gender_counts["Male"], gender_counts["Female"]]
        colors_gender = ["#2196F3", "#E91E63"]
        self.ax_gender.pie(sizes_gender, labels=labels_gender, autopct='%1.1f%%', 
                             colors=colors_gender, startangle=90, wedgeprops={"edgecolor": "black"})
        self.ax_gender.set_title("Patient Gender Distribution")

        # กำหนดกลุ่มอายุ: 0-20, 21-40, 41-60, 61+ และ Unknown
        age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0, "Unknown": 0}
        for record in records:
            age_val = record.get("Age", "Unknown")
            try:
                age = int(age_val)
                if age <= 20:
                    age_groups["0-20"] += 1
                elif age <= 40:
                    age_groups["21-40"] += 1
                elif age <= 60:
                    age_groups["41-60"] += 1
                else:
                    age_groups["61+"] += 1
            except (ValueError, TypeError):
                age_groups["Unknown"] += 1

        self.ax_age.clear()
        # นำเฉพาะกลุ่มที่มีข้อมูล (และแยก Unknown ไว้ท้ายสุด)
        labels_age = [k for k, v in age_groups.items() if v > 0 and k != "Unknown"]
        sizes_age = [v for k, v in age_groups.items() if v > 0 and k != "Unknown"]
        if age_groups["Unknown"] > 0:
            labels_age.append("Unknown")
            sizes_age.append(age_groups["Unknown"])
        colors_age = ["#FFB300", "#8E24AA", "#3949AB", "#00ACC1", "#757575"]

        self.ax_age.pie(sizes_age, labels=labels_age, autopct='%1.1f%%', 
                        colors=colors_age, startangle=90, wedgeprops={"edgecolor": "black"})
        self.ax_age.set_title("Patient Age Distribution")

        self.canvas.draw()

class HistoryWindow(QMainWindow): #ส่วนของหน้าแสดงประวัติ
    selected_patient_signal = pyqtSignal(str)  

    def __init__(self):
        super().__init__()
        self.setWindowTitle("📜 History Records")
        self.resize(1250, 550)  

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f5;
            }
            QTableWidget {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 8px;
                font-size: 14px;
            }
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
                padding: 5px;
            }
        """)

        self.initUI()

    def initUI(self): #กำหนดตัวโครงสร้างUI
        layout = QVBoxLayout()

        # แถบด้านบน มีปุ่ม "📊 Charts" สำหรับแสดงกราฟทั้งสอง
        top_bar = QHBoxLayout()
        title_label = QLabel("📂 MRI Scan History")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        top_bar.addWidget(title_label)

        top_bar.addStretch()

        # ปุ่มเดียวสำหรับเปิดหน้าต่างกราฟที่รวมทั้งเพศและอายุ
        self.charts_button = QPushButton("📊 Charts")
        self.charts_button.setFixedSize(200, 40)
        self.charts_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.charts_button.setStyleSheet(self.button_style("#673AB7", "#5E35B1"))
        self.charts_button.clicked.connect(self.open_combined_chart)
        top_bar.addWidget(self.charts_button)

        layout.addLayout(top_bar)

        # ตารางข้อมูล
        self.table_widget = QTableWidget()
        self.table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(
            ["Age", "Gender", "DateAndTime", "Series Description", "View", "Delete"]
        )

        self.table_widget.setColumnWidth(0, 100)
        self.table_widget.setColumnWidth(1, 100)
        self.table_widget.setColumnWidth(2, 150)
        self.table_widget.setColumnWidth(3, 250)
        self.table_widget.setColumnWidth(4, 180)
        self.table_widget.setColumnWidth(5, 180)

        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setStretchLastSection(False)

        layout.addWidget(self.table_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_history()

    def button_style(self, bg_color, hover_color): #กำหนดลักษณะของปุ่ม
        return f"""
            QPushButton {{
                background-color: {bg_color}; 
                color: white; 
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: #333333;
            }}
        """

    def load_history(self):#โหลดข้อมูลจาก MongoDB และแสดงผลในตาราง
        client = MongoClient('mongodb://localhost:27017/')
        db = client['dicom_database']
        collection = db['dicom_images']
        
        records = list(collection.find({}, {"_id": 0, "PatientID": 1, "Age": 1, "Gender": 1, 
                                              "DateAndTime": 1, "SeriesDescription": 1}))

        if records:
            self.table_widget.setRowCount(len(records))

            for row, record in enumerate(records):
                patient_id = str(record.get("PatientID", "Unknown"))

                age_item = QTableWidgetItem(str(record.get("Age", "Unknown")))
                age_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_widget.setItem(row, 0, age_item)
                self.table_widget.setItem(row, 1, QTableWidgetItem(str(record.get("Gender", "Unknown"))))
                self.table_widget.setItem(row, 2, QTableWidgetItem(str(record.get("DateAndTime", "Unknown"))))
                self.table_widget.setItem(row, 3, QTableWidgetItem(str(record.get("SeriesDescription", "Unknown"))))

                view_button = QPushButton("🔍 View")
                view_button.setFixedSize(160, 30)
                view_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))  
                view_button.setStyleSheet(self.button_style("#4CAF50", "#45a049"))  
                view_button.clicked.connect(lambda _, pid=patient_id: self.view_mri(pid))

                # ปุ่ม Delete พร้อม popup ยืนยัน
                delete_button = QPushButton("🗑 Delete")
                delete_button.setFixedSize(160, 30)
                delete_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                delete_button.setStyleSheet(self.button_style("#FF5722", "#E64A19"))  
                delete_button.clicked.connect(lambda _, r=row, pid=patient_id: self.confirm_delete(r, pid))

                self.add_button_to_table(row, 4, view_button)
                self.add_button_to_table(row, 5, delete_button)

    def add_button_to_table(self, row, col, button):
        """จัดตำแหน่งปุ่มให้อยู่ตรงกลางในช่องประวัติ"""
        layout = QHBoxLayout()
        layout.addWidget(button)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QWidget()
        container.setLayout(layout)
        self.table_widget.setCellWidget(row, col, container)

    def open_combined_chart(self):#เปิดหน้าต่างแสดงกราฟวงกลม
        self.combined_chart_window = CombinedChartWindow()
        self.combined_chart_window.show()

    def view_mri(self, patient_id): #ส่งค่าตำแหน่งในDatabaseไปให้เลือกข้อมูลในDatabaseถูก
        """ส่งสัญญาณไปยังหน้าหลักให้แสดงภาพของ Patient ID ที่เลือก"""
        print(f"🔍 Opening MRI for PatientID: {patient_id}")
        self.selected_patient_signal.emit(patient_id)
        self.close()

    def confirm_delete(self, row, patient_id):# popup เตือนก่อนลบข้อมูล
        """แสดง popup ยืนยันการลบข้อมูลก่อนที่จะลบจริง"""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure to delete this PatientID: {patient_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_record(row, patient_id)

    def delete_record(self, row, patient_id): #ลบข้อมูลจาก MongoDB ตาม Patient ID
        client = MongoClient('mongodb://localhost:27017/')
        db = client['dicom_database']
        collection = db['dicom_images']

        result = collection.delete_one({"PatientID": patient_id})
        if result.deleted_count > 0:
            print(f"✅ ลบข้อมูลสำเร็จ: PatientID {patient_id}")
            self.table_widget.removeRow(row)  
        else:
            print(f"❌ ไม่พบข้อมูลที่ต้องการลบ: PatientID {patient_id}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HistoryWindow()
    window.show()
    sys.exit(app.exec())
