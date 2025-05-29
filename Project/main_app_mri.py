import sys
import pydicom
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from pymongo import MongoClient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QTextEdit, QSplitter, QButtonGroup, QColorDialog
)
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPoint
from datetime import datetime
from segmentation_module import analyze_vertebra_abnormality
from history_window import HistoryWindow

# Custom widget สำหรับวาดลงบนรูป Drawing Areaที่เป็นจุดแสดงภาพSegment
class DrawingLabel(QLabel):
    MODE_NONE = 0
    MODE_FREE = 1
    MODE_LINE = 2
    MODE_ERASER = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = self.MODE_NONE
        self.pen_color = QColor("black")
        # กำหนดค่า default สำหรับ free drawing และ eraser
        self.free_draw_width = 3
        self.eraser_width = 10
        self.pen_width = self.free_draw_width
        self.drawing = False
        self.last_point = None
        self.start_point = None
        self.temp_end_point = None
        self.base_image = QPixmap()
        self.overlay = QPixmap()
        # กำหนดขนาดให้ตรงกับช่องแสดงภาพSegment (420 x 600)
        self.setFixedSize(420, 600)
        self.setStyleSheet("border: 1px solid black;")
    
    def setImage(self, pixmap):
        # ปรับขนาดรูปให้พอดีกับ widget โดยรักษาสัดส่วน
        self.base_image = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        # สร้าง overlay สำหรับวาดที่มีพื้นหลังโปร่งใส โดยมีขนาดเท่ากับ base_image
        self.overlay = QPixmap(self.base_image.size())
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.update()

    def getMergedImage(self):#รวมภาพกับพื้นที่วาดรูปเข้าด้วยกัน ทำให้สามารถวาดลงภาพได้
        if self.base_image.isNull():
            return QPixmap()
        merged = QPixmap(self.base_image.size())
        merged.fill(Qt.GlobalColor.transparent)
        painter = QPainter(merged)
        painter.drawPixmap(0, 0, self.base_image)
        painter.drawPixmap(0, 0, self.overlay)
        painter.end()
        return merged

    def mousePressEvent(self, event): #เช็คการกดคลิกของเม้า
        if self.base_image.isNull() or self.mode == self.MODE_NONE:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            if self.mode in (self.MODE_FREE, self.MODE_ERASER):
                self.last_point = event.pos()
            elif self.mode == self.MODE_LINE:
                self.start_point = event.pos()
                self.temp_end_point = event.pos()

    def mouseMoveEvent(self, event): #ฟังก์ชั่น วาดเส้น,ขีดเส้น,ยางลบ 
        if not self.drawing or self.base_image.isNull():
            return
        if self.mode == self.MODE_FREE:
            painter = QPainter(self.overlay)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
        elif self.mode == self.MODE_ERASER:
            painter = QPainter(self.overlay)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # ใช้ CompositionMode_Clear เพื่อเคลียร์บริเวณที่วาดให้โปร่งใส
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            pen = QPen(QColor(0, 0, 0, 0), self.pen_width, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
        elif self.mode == self.MODE_LINE:
            self.temp_end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event): #เช็คการปล่อยคลิกเม้าของเม้าซ้าย
        if not self.drawing or self.base_image.isNull():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == self.MODE_LINE:
                painter = QPainter(self.overlay)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.start_point, event.pos())
                self.temp_end_point = None
            self.drawing = False
            self.update()

    def paintEvent(self, event): #ตัวเปลี่ยนสีเส้น
        painter = QPainter(self)
        if not self.base_image.isNull():
            painter.drawPixmap(0, 0, self.base_image)
        if not self.overlay.isNull():
            painter.drawPixmap(0, 0, self.overlay)
        # สำหรับโหมดเส้นตรง ให้แสดง preview เส้นด้วยเส้นประ
        if self.mode == self.MODE_LINE and self.drawing and self.start_point and self.temp_end_point:
            pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(self.start_point, self.temp_end_point)

# Main application
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.analysis_displayed = False  # Flag สำหรับตรวจสอบการแสดงผลวิเคราะห์
        self.setWindowTitle("PyQt6 GUI Example")
        self.setGeometry(100, 100, 1366, 768)  # ขนาดหน้าจอ
        self.initUI()


    def initUI(self):
        # ขนาดปุ่ม
        big_btn_width, big_btn_height = 150, 60
        small_btn_width, small_btn_height = 40, 40
        spacing = 5  # ลดความห่างของปุ่มเล็ก
        start_y = 20
        
        # ตำแหน่งแสดงรูปภาพ & ข้อความ
        box_width = 420
        box_height = 600  
        content_start_y = start_y + big_btn_height + 20  # อยู่ใต้ปุ่ม
        img2_x = 460  # ตำแหน่งของ Drawing Area
        text_box_x = 900  # ตำแหน่งช่องแสดงข้อความ

        button_style = """ 
            QPushButton {
                background-color: #aaff7f;
                color: black;
                border-radius: 10px;
                font-size: 18px;
                padding: 5px;
            }
            QPushButton:pressed {
                background-color: #95dd6e;
            }
        """

        # ปุ่มใหญ่ Open DICOM File
        self.upload_botton = QPushButton("Open DICOM File", self)
        self.upload_botton.setGeometry(20, start_y, big_btn_width, big_btn_height)
        self.upload_botton.setStyleSheet(button_style)
        self.upload_botton.clicked.connect(self.open_file)
        
        # ปุ่มใหญ่ Drawing Area
        self.segment_botton = QPushButton("Segmentation", self)
        self.segment_botton.setGeometry(20 + box_width - big_btn_width, start_y, big_btn_width, big_btn_height)
        self.segment_botton.setStyleSheet(button_style)
        self.segment_botton.clicked.connect(self.segmentation_image)

        # ปุ่มใหญ่ Open History
        self.history_botton = QPushButton("Open History", self)
        self.history_botton.setGeometry(text_box_x + box_width - big_btn_width, start_y, big_btn_width, big_btn_height)
        self.history_botton.setStyleSheet(button_style)
        self.history_botton.clicked.connect(self.open_history)
        
        # ปุ่มเล็ก 5 ปุ่ม เลือกโหมดวาด
        self.small_buttons = []
        self.selected_button = None
        small_btn_x = text_box_x  # ให้ขอบซ้ายตรงกับช่องแสดงข้อความ
        small_btn_y = start_y
        
        for i in range(5):  # 5 ปุ่มเล็ก
            btn = QPushButton(self)
            btn.setGeometry(small_btn_x + i * (small_btn_width + spacing), small_btn_y, small_btn_width, small_btn_height)
            btn.setCheckable(True)
            btn.setIcon(QIcon(f"C:\\pythonProject\\FinalProjectForPresent\\icon\\icon{i+1}.png"))
            btn.setIconSize(btn.size())
            btn.setStyleSheet("""
                QPushButton {
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    background-color: #A8E6CF;
                }
                QPushButton:checked {
                    background-color: #388E3C;
                    border: 2px solid #388E3C;
                }
            """)
            btn.clicked.connect(self.toggleButton)
            self.small_buttons.append(btn)

        # ปุ่มเล็ก 3 downloadImage
        self.download_botton = QPushButton(self)
        self.download_botton.setGeometry(small_btn_x + 5 * (small_btn_width + spacing), small_btn_y, small_btn_width, small_btn_height)
        self.download_botton.setStyleSheet("""
                QPushButton {
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    background-color: #A8E6CF;
                }""")
        self.download_botton.setIcon(QIcon(f"C:\\pythonProject\\FinalProjectForPresent\\icon\\icon{6}.png"))
        self.download_botton.setIconSize(self.download_botton.size())
        self.download_botton.clicked.connect(self.download_image)

        # ปรับตำแหน่ง Label ให้ติดกับกล่องแสดงผลและชิดกับขอบซ้าย
        # สำหรับ Input Image
        self.input_label = QLabel("InputImage", self)
        self.input_label.setGeometry(20, content_start_y - 20, 100, 20)  # 100px กว้าง, ติดกับขอบซ้าย
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.input_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # สำหรับ Color bar
        self.colorbar_label = QLabel("ColorBar", self)
        self.colorbar_label.setGeometry(img2_x, content_start_y - 100, 120, 20)  # 120px กว้าง, ชิดขอบซ้าย
        self.colorbar_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.colorbar_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # สำหรับ Segment Image
        self.segment_label = QLabel("SegmentImage", self)
        self.segment_label.setGeometry(img2_x, content_start_y - 20, 120, 20)  # 120px กว้าง, ชิดขอบซ้าย
        self.segment_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.segment_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # สำหรับ Data Output
        self.output_label = QLabel("DataOutput", self)
        self.output_label.setGeometry(text_box_x, content_start_y - 20, 100, 20)  # 100px กว้าง, ชิดขอบซ้าย
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.output_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # แสดงรูปภาพ 1
        self.img_label1 = QLabel(self)
        self.img_label1.setGeometry(20, content_start_y, box_width, box_height)
        self.img_label1.setPixmap(QPixmap("image1.jpg").scaled(box_width, box_height, Qt.AspectRatioMode.KeepAspectRatio))
        self.img_label1.setStyleSheet("border: 1px solid black;")

        # แสดงรูปภาพColorbar
        self.colorbar_img = QLabel(self)
        self.colorbar_img.setGeometry(img2_x, content_start_y - 80, box_width, 60)
        self.colorbar_img.setPixmap(QPixmap("image1.jpg").scaled(box_width, box_height, Qt.AspectRatioMode.KeepAspectRatio))
        self.colorbar_img.setStyleSheet("border: 1px solid black;")

        # แสดงพื้นที่วาด (drawing area) ใน ภาพSegmented
        self.drawing_label = DrawingLabel(self)
        self.drawing_label.setGeometry(img2_x, content_start_y, box_width, box_height)

        # ช่องแสดงข้อความ พร้อมปรับขนาดตัวอักษรให้ใหญ่และอ่านง่ายขึ้น
        self.text_box = QTextEdit(self)
        self.text_box.setGeometry(text_box_x, content_start_y, box_width, box_height)
        self.text_box.setStyleSheet("border: 1px solid black; font-size: 20px; font-family: Arial, sans-serif;")
        self.text_box.setReadOnly(True)

    def toggleButton(self): #เลือกฟังก์ชั่นวาดรูป
        sender = self.sender()
        if self.selected_button and self.selected_button != sender:
            self.selected_button.setChecked(False)
        self.selected_button = sender
        # กำหนดโหมดวาดตาม index ของปุ่มเล็ก
        index = self.small_buttons.index(sender)
        if index == 0:
            self.drawing_label.mode = DrawingLabel.MODE_NONE
        elif index == 1:
            self.drawing_label.mode = DrawingLabel.MODE_FREE
            self.drawing_label.pen_width = self.drawing_label.free_draw_width
        elif index == 2:
            self.drawing_label.mode = DrawingLabel.MODE_LINE
            self.drawing_label.pen_width = self.drawing_label.free_draw_width
        elif index == 3:
            self.change_color()
        elif index == 4:
            self.drawing_label.mode = DrawingLabel.MODE_ERASER
            self.drawing_label.pen_width = self.drawing_label.eraser_width

    def change_color(self): #เปลี่ยนสีปุ่ม
        color = QColorDialog.getColor(initial=self.drawing_label.pen_color, parent=self, title="Select Pen Color")
        if color.isValid():
            self.drawing_label.pen_color = color
            self.drawing_label.mode = DrawingLabel.MODE_FREE
            self.drawing_label.pen_width = self.drawing_label.free_draw_width

    def open_file(self): #Uploadตัวไฟล์MRI
        file_path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm *.ima)")
        latest_document = None
        if file_path:
            self.analysis_displayed = False

            # ล้างรูปที่อาจจะแสดงอยู่ก่อนการเปิดไฟล์
            self.img_label1.clear()  
            self.drawing_label.setImage(QPixmap())  # Clear the Segment Image
            self.colorbar_img.clear()
            dataset = pydicom.dcmread(file_path)
            image = dataset.pixel_array
                
            fig, ax = plt.subplots()
            ax.imshow(image, cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            buffer = BytesIO()
            plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            #ปรับข้อมูลAge และSexก่อนเก็บ
            patientSex = str(dataset.get("PatientSex", "Null"))
            if patientSex.upper() == "M":
                patientSex = "Male"
            elif patientSex.upper() == "F":
                patientSex = "Female"
            else: patientSex = "Not Found Gender"
            
            patientAge = str(dataset.get("PatientAge", "Null"))
            if patientAge != "Null" and patientAge[-1].upper() == 'Y':
                patientAge = str(int(patientAge[:-1]))
            else:
                patientAge = "Not Found Age"

            document = {
                "PatientID": str(dataset.get("SOPInstanceUID", "Null")),
                "Age": patientAge,
                "Gender": patientSex,
                "DateAndTime": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "SeriesDescription": str(dataset.get("SeriesDescription", "Null")),
                "MriBase": image_base64,
            }
                
            client = MongoClient('mongodb://localhost:27017/')
            db = client['dicom_database']
            collection = db['dicom_images']
            existing_doc = collection.find_one({
                "PatientID": document["PatientID"],
            })

            if existing_doc:
                print("ข้อมูลนี้มีอยู่แล้วในฐานข้อมูล!")
            else:
                collection.insert_one(document)
                print("บันทึกข้อมูลเรียบร้อย!")

            latest_document = collection.find_one({}, sort=[("_id", -1)])

        if latest_document:
            metadata_text = "\n".join([f"{key}: {value}" for key, value in latest_document.items() if key not in ["MriBase", "MriSegmented","PatientID","_id","ColorBar","Analysis"]])
            formatted_text = "<p style='line-height: 125%;'>" + "</p><p style='line-height: 125%;'>".join(metadata_text.split('\n')) + "</p>"
            self.text_box.setHtml(formatted_text)

            mri_image_data = base64.b64decode(latest_document["MriBase"])

            pixmap = QPixmap()
            pixmap.loadFromData(mri_image_data)

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.img_label1.width(),
                    self.img_label1.height(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
                img_width, img_height = scaled_pixmap.width(), scaled_pixmap.height()
                label_width, label_height = self.img_label1.width(), self.img_label1.height()
                crop_x = max((img_width - label_width) // 2, 0)
                crop_y = max((img_height - label_height) // 2, 0)
                crop_width = min(label_width, img_width)
                crop_height = min(label_height, img_height)
                cropped_pixmap = scaled_pixmap.copy(crop_x, crop_y, crop_width, crop_height)
                self.img_label1.setPixmap(cropped_pixmap)
            else:
                self.text_box.setText("Error: Could not load image.")

    def segmentation_image(self):
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)  # เปลี่ยนเมาส์เป็นหมุน
        
        client = MongoClient('mongodb://localhost:27017/')
        db = client['dicom_database']
        collection = db['dicom_images']

        latest_document = collection.find_one({}, sort=[("_id", -1)])
        image_base64 = latest_document["MriBase"]
        result = analyze_vertebra_abnormality(image_base64)

        # บันทึก base64 ของภาพ segmentation และผล analysis เข้าไปในเอกสารของฐานข้อมูล
        collection.update_one(
            {"_id": latest_document["_id"]},
            {"$set": {
                "MriSegmented": result["mask_base64"],
                "ColorBar" : result["colorbar_base64"],
                "Analysis": result["analysis"]
            }}
        )
        
        mri_imagemark_data = base64.b64decode(result["mask_base64"]) 
        pixmap = QPixmap()
        pixmap.loadFromData(mri_imagemark_data)

        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.drawing_label.width(),
                self.drawing_label.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            img_width, img_height = scaled_pixmap.width(), scaled_pixmap.height()
            label_width, label_height = self.drawing_label.width(), self.drawing_label.height()
            crop_x = max((img_width - label_width) // 2, 0)
            crop_y = max((img_height - label_height) // 2, 0)
            crop_width = min(label_width, img_width)
            crop_height = min(label_height, img_height)
            cropped_pixmap = scaled_pixmap.copy(crop_x, crop_y, crop_width, crop_height)
            # กำหนดรูปที่ segment แล้วให้กับ drawing area
            self.drawing_label.setImage(cropped_pixmap)

                # เพิ่มส่วนการแสดง colorbar
        colorbar_data = base64.b64decode(result["colorbar_base64"])
        pixmap_colorbar = QPixmap()
        pixmap_colorbar.loadFromData(colorbar_data)
        if not pixmap_colorbar.isNull():
            scaled_colorbar = pixmap_colorbar.scaled(
                self.colorbar_img.width(),
                self.colorbar_img.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.colorbar_img.setPixmap(scaled_colorbar)    
            
            # แสดงผลวิเคราะห์เพียงครั้งเดียวเท่านั้น
            if not self.analysis_displayed:
                analysis_text = "\n".join(result["analysis"]) if isinstance(result["analysis"], list) else result["analysis"]
                self.text_box.append(analysis_text)
                self.analysis_displayed = True  # ตั้ง flag เพื่อไม่ให้แสดงซ้ำ
            QApplication.restoreOverrideCursor()  # คืนค่าเมาส์เป็นปกติ

        else:
            self.text_box.setText("Error: Could not load image Segment.")
        print("Segment")



    def download_image(self):
        # รวมรูปจาก drawing area (base_image + overlay)
        merged = self.drawing_label.getMergedImage()
        if merged.isNull():
            return
        # เปิด dialog ให้ผู้ใช้เลือกที่บันทึกไฟล์
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if save_path:
            merged.save(save_path, "PNG")
            print("Image saved to:", save_path)
    
    def open_history(self):
        self.history_window = HistoryWindow()
        # เชื่อมต่อสัญญาณ selected_patient_signal กับ slot load_patient_data
        self.history_window.selected_patient_signal.connect(self.load_patient_data)
        self.history_window.show()
    
    def load_patient_data(self, patient_id):
        client = MongoClient('mongodb://localhost:27017/')
        db = client['dicom_database']
        collection = db['dicom_images']
        document = collection.find_one({"PatientID": patient_id})
        
        if not document:
            self.text_box.setText("ไม่พบข้อมูลของ Patient")
            return
        
        # อัพเดตข้อมูลเมตาของผู้ป่วยลงใน text_box (ไม่รวมข้อมูลภาพ)
        metadata_text = "\n".join([f"{key}: {value}" for key, value in document.items()
                                    if key not in ["MriBase", "MriSegmented", "PatientID", "_id", "Analysis","ColorBar"]])
        
        # รวมผลการวิเคราะห์ (ถ้ามี)
        if "Analysis" in document and document["Analysis"]:
            analysis_text = "\n".join(document["Analysis"]) if isinstance(document["Analysis"], list) else document["Analysis"]
            metadata_text += "\n" + analysis_text
        
        formatted_text = "<p style='line-height: 125%;'>" + "</p><p style='line-height: 125%;'>".join(metadata_text.split('\n')) + "</p>"
        self.text_box.setHtml(formatted_text)
        
        # โหลดและแสดงภาพ MRI ดั้งเดิม (ถ้าต้องการแสดงใน img_label1)
        mri_image_data = base64.b64decode(document["MriBase"])
        pixmap = QPixmap()
        pixmap.loadFromData(mri_image_data)
        
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.img_label1.width(),
                self.img_label1.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            img_width, img_height = scaled_pixmap.width(), scaled_pixmap.height()
            label_width, label_height = self.img_label1.width(), self.img_label1.height()
            crop_x = max((img_width - label_width) // 2, 0)
            crop_y = max((img_height - label_height) // 2, 0)
            crop_width = min(label_width, img_width)
            crop_height = min(label_height, img_height)
            cropped_pixmap = scaled_pixmap.copy(crop_x, crop_y, crop_width, crop_height)
            self.img_label1.setPixmap(cropped_pixmap)
        else:
            self.text_box.setText("Error: Could not load image.")
        
        # ตรวจสอบว่ามีภาพ segmentation ที่บันทึกไหม
        if "MriSegmented" in document and document["MriSegmented"]:
            segmented_data = base64.b64decode(document["MriSegmented"])
            pixmap_segment = QPixmap()
            pixmap_segment.loadFromData(segmented_data)
            if not pixmap_segment.isNull():
                scaled_pixmap_seg = pixmap_segment.scaled(
                    self.drawing_label.width(),
                    self.drawing_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
                img_width, img_height = scaled_pixmap_seg.width(), scaled_pixmap_seg.height()
                label_width, label_height = self.drawing_label.width(), self.drawing_label.height()
                crop_x = max((img_width - label_width) // 2, 0)
                crop_y = max((img_height - label_height) // 2, 0)
                crop_width = min(label_width, img_width)
                crop_height = min(label_height, img_height)
                cropped_pixmap_seg = scaled_pixmap_seg.copy(crop_x, crop_y, crop_width, crop_height)
                # แสดงภาพ segmentation ใน drawing area
                self.drawing_label.setImage(cropped_pixmap_seg)

        colorbar_data = base64.b64decode(document["ColorBar"])
        pixmap_colorbar = QPixmap()
        pixmap_colorbar.loadFromData(colorbar_data)
        if not pixmap_colorbar.isNull():
            scaled_colorbar = pixmap_colorbar.scaled(
                self.colorbar_img.width(),
                self.colorbar_img.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.colorbar_img.setPixmap(scaled_colorbar)    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())