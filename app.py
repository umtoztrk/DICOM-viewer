from Qdrant import Server
from pathlib import Path
import os
import sys
import ast
import cv2
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QTreeWidgetItem, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QRectF
from collections import defaultdict
from mainMenuUI import Ui_MainWindow
from widgetFinalv2 import Ui_MainWindow2
from Filters import Filters
from SQLite import SqliteTest
from editReportUI import Ui_EditReport
import pydicom
from PIL import Image, ImageEnhance
import numpy as np
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyQt5.QtWidgets import QMessageBox

class WidgetApp(QtWidgets.QMainWindow, Ui_MainWindow2):
    def __init__(self, parent=None):
        super(WidgetApp, self).__init__(parent)
        self.setupUi(self)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["Code", "Name", "Value"])
        self.lineEdit.textChanged.connect(self.searchForName)
        self.btSave.clicked.connect(self.saveAsMenu)

        self.saveAsTxt.triggered.connect(self.saveTxt)
        self.saveAsCsv.triggered.connect(self.saveCsv)
        self.saveAsXlsx.triggered.connect(self.saveXlsx)
        self.saveAsPdf.triggered.connect(self.savePdf)

        self.tagInfo = None
    
    def saveTxt(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Dosyayı Kaydet", "", "Text Files (*.txt)", options=options)

        if file_path:
            # Dosyayı seçilen konumda kaydet
            with open(file_path, "w", encoding="utf-8") as file:
                # Başlıkları yaz
                file.write("Code        	  Name 					 : Value\n\n")

                # Sözlüğü satır satır yaz
                for key, value in self.tagInfo.items():
                    file.write(f"{key}      {value['name']} : {value['value']}\n")

    def saveCsv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Dosyayı Kaydet", "", "CSV Files (*.csv)", options=options)

        if file_path:
            # Dosyayı seçilen konumda kaydet
            with open(file_path, "w", encoding="utf-8") as file:
                # Başlıkları yaz
                file.write("Code;Name;Value\n")

                # Sözlüğü satır satır yaz
                for key, value in self.tagInfo.items():
                    file.write(f"{key};{value['name']};{value['value']}\n")

    def saveXlsx(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Dosyayı Kaydet", "", "Excel Files (*.xlsx)", options=options)

        if file_path:
            df = pd.DataFrame(list(self.tagInfo.items()), columns=["Anahtar", "Değer"])

            # DataFrame'i Excel dosyasına yaz
            df.to_excel(file_path, index=False, engine='openpyxl')
            
            # Dosyayı seçilen konumda kaydet
            with open(file_path, "w", encoding="utf-8") as file:
                # Başlıkları yaz
                df = pd.DataFrame(self.tagInfo).T  # .T transpozisyonu ile satırlara dönüştürürüz

                # DataFrame'i Excel dosyasına yaz
                df.to_excel(file_path, index=False, engine='openpyxl')

    def savePdf(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Dosyayı Kaydet", "", "PDF Files (*.pdf);;All Files (*)", options=options)

        if file_path:
            # PDF dosyasını oluştur
            c = canvas.Canvas(file_path, pagesize=letter)
            c.setFont("Helvetica", 10)

            # Başlık yazalım
            c.drawString(50, 750, "DICOM Metadata")
            c.drawString(50, 730, "-----------------")

            # Y verisini basma (başlangıç yüksekliği)
            y_position = 710

            # metadata_dict içeriğini PDF'ye ekle
            for key, value in self.tagInfo.items():
                c.drawString(50, y_position, f"Tag: {value['tag']}")
                y_position -= 15
                c.drawString(50, y_position, f"Name: {value['name']}")
                y_position -= 15
                c.drawString(50, y_position, f"Value: {value['value']}")
                y_position -= 30

                # Sayfa sonu kontrolü
                if y_position < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y_position = 750

            # PDF dosyasını kaydet
            c.save()


    def saveAsMenu(self):
        self.saveMenu.exec_(self.btSave.mapToGlobal(self.btSave.rect().bottomLeft()))

    def searchForName(self):
        self.tableWidget.setRowCount(0)
        searchWord = self.lineEdit.text()
        elements = [[key, value['name'], value['value']] for key, value in self.tagInfo.items() if searchWord in value['name'].lower()]
        if elements is None:
            return
        self.tableWidget.setRowCount(len(elements))
            
            # Fill table
        for row, (tag, name, value) in enumerate(elements):
            # Convert tag to string
            tag_str = str(tag)
            
            # Convert value to string, handle special cases
            if isinstance(value, bytes):
                value_str = str(value)[2:-1]  # Remove b'' 
            else:
                value_str = str(value)
            
            # Create table items
            tag_item = QtWidgets.QTableWidgetItem(tag_str)
            name_item = QtWidgets.QTableWidgetItem(name)
            value_item = QtWidgets.QTableWidgetItem(value_str)
            
            # Add items to table
            self.tableWidget.setItem(row, 0, tag_item)
            self.tableWidget.setItem(row, 1, name_item) 
            self.tableWidget.setItem(row, 2, value_item)
            
        # Resize columns to content
            self.tableWidget.resizeColumnsToContents()
    

    def fill_tableV2(self, dicom_fileTags):
        """
        Takes a DICOM file and fills the table with all metadata/annotations
        """
        self.tagInfo = dicom_fileTags
        try:
            # Clear existing table contents
            self.tableWidget.setRowCount(0)
            

            # Get all elements
            elements = [[key, value['name'], value['value']] for key, value in self.tagInfo.items()]
            
            # Set number of rows
            self.tableWidget.setRowCount(len(elements))
            
            # Fill table
            for row, (tag, name, value) in enumerate(elements):
                # Convert tag to string
                tag_str = str(tag)
                
                # Convert value to string, handle special cases
                if isinstance(value, bytes):
                    value_str = str(value)[2:-1]  # Remove b'' 
                else:
                    value_str = str(value)
                
                # Create table items
                tag_item = QtWidgets.QTableWidgetItem(tag_str)
                name_item = QtWidgets.QTableWidgetItem(name)
                value_item = QtWidgets.QTableWidgetItem(value_str)
                
                # Add items to table
                self.tableWidget.setItem(row, 0, tag_item)
                self.tableWidget.setItem(row, 1, name_item) 
                self.tableWidget.setItem(row, 2, value_item)
                
            # Resize columns to content
            self.tableWidget.resizeColumnsToContents()
            
        except Exception as e:
            print(f"Error reading DICOM file: {str(e)}")

filterNames = [
    "Orijinal",
    "Thresholded",
    "Cleaned",
    "Contour",
    "Colored",
    "Normalized",
    "GaussianFilter",
    "HistogramEqualization",
    "AnistropicDiffusion",
    "CLAHE",
    "MorphologicalOperations",
    "GaussianBlur",
    "FourierFiltering",
    "BilateralFilter",
    "FinalSegmentation",
    "ROISegmentation",
    "WaveletTransform",
    "BiasFieldCorrection",
    "Non-LocalMeans",
    "AdaptiveFiltering",
    "HistogramMatching",
    "ResamplingInterpolation",
    "LaplacianofGaussian",
    "SegmentationSmoothing",
    "CannyEdgeDetection",
    "3DReconstruction",
    "PhaseCongruency",
    "GMMSegmentation"
]

class editReport(QtWidgets.QMainWindow, Ui_EditReport):
    def __init__(self, parent=None):
        super(editReport, self).__init__(parent)
        self.setupUi(self)
        self.btSave.clicked.connect(self.saveReport)
        self.btClear.clicked.connect(self.clearReport)
        self.parentW = parent
    def setTxt(self):
        self.btClear.show()
        self.btSave.show()
        self.txtReport.setReadOnly(False)
        self.txtReport.setText(self.parentW.txtReport)
    def onlySetTxt(self):
        self.btClear.hide()
        self.btSave.hide()
        self.txtReport.setReadOnly(True)
        self.txtReport.setText(self.parentW.txtReport)
    def saveReport(self):
        self.parentW.updateReport(self.txtReport.toPlainText())
    def clearReport(self):
        self.txtReport.clear()
        

class MainWindowApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindowApp, self).__init__(parent)
        self.setupUi(self)
        self.Wapp = WidgetApp(self)
        self.qdrantDb = Server()
        self.editReport = editReport(self)
        self.current_image = None
        self.contrast_factor = 0.5
        # Connect existing menu action
        self.actionTag_Goruntule.triggered.connect(self.openWidget)
        # Enable mouse tracking for graphics view
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.wheelEvent = self.wheelEvent
        self.zoom_factor = 1.0  # Add zoom factor tracking
        self.graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)  # Zoom towards cursor

        self.prevContrastValue = 50
        self.is_folder = False
        self.contrastSlider.valueChanged.connect(self.changeContrast)
        self.hideShowPatient.triggered.connect(self.hidePatientInfoStatus)
        self.hideInfoCheck = False
        self.filterCheck = False
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMinMaxButtonsHint)
        self.setFixedSize(1300, 900)
        #self.btFile.clicked.connect(self.select_file)
        self.lbFile.clicked.connect(self.select_file)
        self.treeWidget.setFocusPolicy(Qt.NoFocus)
        self.setFocusPolicy(Qt.StrongFocus)
        self.graphicsView.setFocusPolicy(Qt.NoFocus)
        self.searchPatient.setFocusPolicy(Qt.ClickFocus)
        #self.btFolder.clicked.connect(self.select_folder)
        self.lbFolder.clicked.connect(self.select_folder)
        self.treeWidget.setHeaderHidden(True)  # Hide all headers
        
        self.patientList = []
        self.txtReport = " "
        self.txtReportItem = None
        self.patientInfo = []
        self.dbSearch = False
        self.sq = SqliteTest()
        self.instanceLen = 0
        self.current_dicom = None
        self.treeWidget.currentItemChanged.connect(self.show_selected_image)
        self.treeWidget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.name_dict = defaultdict(lambda: defaultdict(lambda: defaultdict((list))))
        self.actionSave_as_PNG.triggered.connect(self.saveImagePng)
        self.actionSave_as_JPG.triggered.connect(self.saveImageJpeg)
        self.actionSave_as_SVG.triggered.connect(self.saveImageSvg)
        self.filterCombo.currentIndexChanged.connect(self.updateImage)
        self.actionSave.triggered.connect(self.saveAllImages)
        #self.btReport.clicked.connect(self.displayReport)
        self.lbEditReport.clicked.connect(self.displayReport)
        #self.btSearch.clicked.connect(self.patientSearch)
        self.lbSearch.clicked.connect(self.patientSearch)
        self.listPatient.cellDoubleClicked.connect(self.getPoints)
        self.openReport.triggered.connect(self.showReport)

    def showReport(self):
        self.editReport.onlySetTxt()
        self.editReport.show()

    def displayReport(self):
        self.editReport.setTxt()
        self.editReport.show()

    def updateReport(self, text):
        self.txtReport = text
        if self.dbSearch:
            self.qdrantDb.update_radReport_point(int(self.txtReportItem[0].id), self.txtReport)

    def patientSearch(self):
        query = f"%{self.searchPatient.text()}%"
        result = self.sq.allPatients(query)
        self.patientList = result

        if result:
            self.listPatient.setRowCount(len(result))
            for row, record in enumerate(result):
                name_item = QtWidgets.QTableWidgetItem(record[2])
                study_item = QtWidgets.QTableWidgetItem(record[3])
                date_item = QtWidgets.QTableWidgetItem(record[4])
                age_item = QtWidgets.QTableWidgetItem(str(record[5]))
                sex_item = QtWidgets.QTableWidgetItem(record[6])


                self.listPatient.setItem(row, 0, name_item)
                self.listPatient.setItem(row, 1, study_item)
                self.listPatient.setItem(row, 2, date_item)
                self.listPatient.setItem(row, 3, age_item)
                self.listPatient.setItem(row, 4, sex_item)

        self.listPatient.resizeColumnsToContents()

    def getPoints(self, row, column):
        self.name_dict.clear()
        id = None
        searchName = self.listPatient.item(row, 0).text()
        
        for record in self.patientList:
            if record[2] == searchName:
                id = record[1]
                break 

        res = self.sq.listele_patient(id)
        selected_text = self.filterCombo.currentText()
        selected_text = selected_text.replace(" ", "")
        reportResult = self.qdrantDb.search_report(str(id))
        self.txtReportItem = reportResult[0]
        if self.txtReportItem:
            self.txtReport = self.txtReportItem[0].payload['report']

        result = self.qdrantDb.search_filtered(str(id), selected_text)
        pathList = []
        metadata = None
        path = None
        for record in result[0]:
            path = record.payload.get("path")  # 'path' değerini alıyoruz
            pathList.append(path)
            break
        
        for root, _, files in os.walk(path):
            # Alt klasörlerin isimlerini kaydet
            for file in files:
                file_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)
                series_path = os.path.dirname(file_path)
                series_name = os.path.basename(series_path)
                study_path = os.path.dirname(series_path)
                study_name = os.path.basename(study_path)
                patient_path = os.path.dirname(study_path)
                patient_path = os.path.dirname(patient_path)
                patient_name = os.path.basename(patient_path)
                for dicom in res:
                    _, dicom_name, dicom_tags = dicom  # Tuple'dan verileri al
                    if dicom_name == file_name:
                        metadata = json.loads(dicom_tags)
                        break  # Bulunduysa döngüyü sonlandır
                
                file_info = {
                    'file_name': file_name,
                    'image_path': file_path,
                    'dicom': metadata,
                    'instance_num': int(metadata.get('(0020, 0013)', {}).get('value', 0))
                }
                self.name_dict[patient_name][study_name][series_name].append(file_info)
        self.dbSearch = True
        self.setTree()
        self.show_list_items()


    def saveAllImages(self):
        checkFilter = False
        reportPayload = None
        m = self.sq.num_dicoms()
        self.qdrantDb.create_collection()
        for pat_name, study_dict in self.name_dict.items():
            for item in self.patientList:
                if item[2] == pat_name:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Bu hasta zaten kayıtlı.")
                    msg.setWindowTitle("Hata")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
            checkFilter = False
            reportEmb = self.qdrantDb.transform_txtVector(self.txtReport)
            self.sq.ekle_patient(self.patientInfo[0], self.patientInfo[1], self.patientInfo[2], self.patientInfo[3], self.patientInfo[4], self.patientInfo[5])
            for k, filter_name in enumerate(filterNames):
                if filter_name != "Orijinal":
                    checkFilter = True
                id_lists, payload_dict, vec_lists = [], [], []
                for study_desc, series_dict in study_dict.items():
                    if checkFilter == False:
                        self.sq.ekle_study(study_desc, self.sq.num_patients())
                    for series_desc, files in series_dict.items():
                        if checkFilter == False:
                            self.sq.ekle_series(series_desc, self.sq.num_studies())
                        files = sorted(files, key=lambda x: x['instance_num'])
                        for i, file_info in enumerate(files):
                            # Create folder path based on dictionary keys
                            folder_path = os.path.join(
                                "output_images",
                                pat_name,
                                filter_name,
                                study_desc,
                                series_desc
                            )
                            os.makedirs(folder_path, exist_ok=True)
                            
                            if i == 0:
                                reportPayload = {
                                    "name": pat_name,
                                    "p_id": str(file_info['dicom'].get('(0010, 0020)', {}).get('value', 'Bilgi bulunamadı')),
                                    "report": self.txtReport
                                }

                            # Save image in the created folder
                            image_array = file_info['pixel_array']
                            if filter_name == "Orijinal":
                                emb = self.qdrantDb.transform_vector(image_array)
                                vec_lists.append(emb)
                                window_center = int(file_info['dicom'].get('(0028, 1050)').get('value', 'Bilgi bulunamadı'))
                                window_width = int(file_info['dicom'].get('(0028, 1051)').get('value', 'Bilgi bulunamadı'))
                                image_array = MainWindowApp.apply_window(image_array, window_center, window_width)
                            else:
                                image_array = Filters.selectFilter(file_info['pixel_array'], filter_name)
                            
                            if checkFilter == False:
                                json_metadata = json.dumps(file_info['dicom'])
                                self.sq.ekle_dicom(file_info['file_name'], json_metadata, self.sq.num_series())
                            
                            if len(image_array.shape) == 2:  # Grayscale
                                image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
                            image = Image.fromarray((image_array / np.max(image_array) * 255).astype(np.uint8))
                            image_path = os.path.join(folder_path, f"{file_info['file_name']}.png")
                            image.save(image_path)
                            im = image_path
                            image_path = os.path.abspath(os.path.join(image_path, os.pardir, os.pardir, os.pardir))
                            payload_item = {
                                "name": pat_name,
                                "id": str(file_info['dicom'].get('(0010, 0020)', {}).get('value', 'Bilgi bulunamadı')),
                                "filter": filter_name,
                                "study": study_desc,
                                "series": series_desc,
                                "instance": i+1,
                                "path": image_path
                            }
                            if filter_name == "Orijinal":
                                id_lists.append(m)
                                m += 1
                                payload_dict.append(payload_item)

                            elif checkFilter:
                                id_lists.append(m)
                                payload_dict.append(payload_item)
                                checkFilter = False

                if filter_name == "Orijinal":
                    self.qdrantDb.add_point(id_lists, vec_lists, payload_dict, filter_name)
                self.qdrantDb.add_filtered_point(((27 * (self.sq.num_patients()-1)) + k), payload_item)

            self.qdrantDb.add_radReport_point(self.sq.num_patients(), reportEmb, reportPayload)


    def updateImage(self):
        if self.dbSearch is False:
            k = self.filterCombo.currentText().replace(" ", "")
            newFilter = Filters.selectFilter(self.current_dicom['pixel_array'], k)
            self.current_dicom['filtered_array'] = newFilter
            self.filterCheck = True
        else:
            tmpPath = self.current_dicom['image_path']
            x = Path(tmpPath)
            x = list(x.parts)
            x[-4] = self.filterCombo.currentText().replace(" ", "")
            tmpPath = os.path.join(*x)
            self.current_dicom['image_path'] = tmpPath

        self.show_list_items()
        

    def saveImagePng(self):
        file_path, _ = QFileDialog.getSaveFileName(
            None, 
            "Kaydetmek için dosya seçin", "", "PNG Dosyası (*.png)"
        )
        if file_path:  # Eğer kullanıcı bir dosya seçtiyse
            # QGraphicsView'in boyutuna göre bir QPixmap oluştur
            pixmap = QPixmap(self.graphicsView.viewport().size())
            # QPixmap üzerine çizim yapmak için bir QPainter oluştur
            painter = QPainter(pixmap)
            # QGraphicsView'in içeriğini QPixmap'e çiz
            self.graphicsView.render(painter)
            painter.end()  # QPainter'i kapat
            # QPixmap'i dosyaya kaydet
            pixmap.save(file_path)
  
    def saveImageJpeg(self):
            file_path, _ = QFileDialog.getSaveFileName(self, "Dosya Kaydet", "", "JPEG Dosyası (*.jpeg)")
            if file_path:
                    # QGraphicsView'in boyutuna göre bir QPixmap oluştur
                pixmap = QPixmap(self.graphicsView.viewport().size())
                # QPixmap üzerine çizim yapmak için bir QPainter oluştur
                painter = QPainter(pixmap)
                # QGraphicsView'in içeriğini QPixmap'e çiz
                self.graphicsView.render(painter)
                painter.end()  # QPainter'i kapat
                # QPixmap'i dosyaya kaydet
                pixmap.save(file_path)

    def saveImageSvg(self):
        # Kullanıcıdan dosya kaydetme yeri seçmesini iste
        file_path, _ = QFileDialog.getSaveFileName(None, "SVG olarak kaydet", "", "SVG Dosyası (*.svg)")
        if file_path:  # Eğer kullanıcı bir dosya seçtiyse
            # QSvgGenerator nesnesi oluştur
            svg_generator = QSvgGenerator()
            svg_generator.setFileName(file_path)
            svg_generator.setSize(self.graphicsView.viewport().size())
            svg_generator.setViewBox(QRectF(0, 0, self.graphicsView.viewport().width(), self.graphicsView.viewport().height()))
            
            # QSvgGenerator üzerine çizim yapmak için bir QPainter oluştur
            painter = QPainter(svg_generator)
            # QGraphicsView'in içeriğini SVG'ye çiz
            self.graphicsView.render(painter)
            painter.end()  # QPainter'i kapat

    def clearAllLabels(self):
        self.patientName.setText("")
        self.patientId.setText("")
        self.patientDate.setText("")
        self.patientInstitution.setText("")
        self.patientSeries.setText("")
        self.patientStudy.setText("")
        self.patientInstanceNum.setText("")
        self.imageZoom.setText("")
        self.imageContrast.setText("")

    def on_item_double_clicked(self, item, column):
        # Çift tıklanan öğe ve sütun bilgisini al
        # Alt öğeleri kontrol et, varsa ilk çocuğu seçili yap
        if item.childCount() > 0:
            first_child = item.child(0)
            self.treeWidget.setCurrentItem(first_child)

    def changeContrast(self, value):
        factor = value / self.prevContrastValue
        self.prevContrastValue = value
        self.contrast_factor *= factor
        self.update_image_contrast()

    def hidePatientInfoStatus(self):
        if self.hideInfoCheck == True:
            self.hideInfoCheck = False
        else:
            self.hideInfoCheck = True
        self.show_list_items()

    def hidePatientInfo(self):
        if self.hideInfoCheck == False:
            self.patientName.setText("******")
            self.patientId.setText("******")
            self.patientDate.setText("******")
            self.patientInstitution.setText("******")
            self.treeWidget.topLevelItem(0).setText(0, "Anonim")
        else:
            p_name = self.current_dicom['dicom'].get('(0010, 0010)').get('value', 'Bilgi bulunamadı')
            p_id = self.current_dicom['dicom'].get('(0010, 0020)', {}).get('value', 'Bilgi bulunamadı')
            p_birthdate = self.current_dicom['dicom'].get('(0010, 0030)', {}).get('value', 'Bilgi bulunamadı')
            if p_birthdate != "Bilgi yok":
                p_birthdate = f"{p_birthdate[6:]}-{p_birthdate[4:6]}-{p_birthdate[:4]}"
            i_name = self.current_dicom['dicom'].get('(0008, 0080)', {}).get('value', 'Bilgi bulunamadı')

            self.patientName.setText(str(p_name))
            self.patientId.setText(str(p_id))
            self.patientDate.setText(str(p_birthdate))
            self.patientInstitution.setText(str(i_name))
            self.treeWidget.topLevelItem(0).setText(0, str(p_name))

    def show_selected_image(self, current):
        self.graphicsView.setScene(QGraphicsScene())
        if current is None:
            return
        for _, study_dict in self.name_dict.items():
            for _, series_dict in study_dict.items():
                for _, files in series_dict.items():
                    #files = sorted(files, key=lambda x: x['instance_num'])
                    for file_info in files:
                        if file_info['file_name'] == current.text(0):
                            self.current_dicom = file_info
                            self.instanceLen = len(files)
                            self.show_list_items()    #method içine filtre tipini ver
                                
    def select_folder(self):
        self.filterCheck = False
        self.dbSearch = False
        self.patientInfo = []
        self.clearAllLabels()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.is_folder=True
            self.process_dicom_data(folder_path, self.is_folder)

    def select_file(self):
        self.filterCheck = False
        self.dbSearch = False
        self.patientInfo = []
        self.clearAllLabels()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DICOM File",
            "",
            "DICOM Files (*.dcm)"
        )
        if file_path:
            self.is_folder=False
            self.process_dicom_data(file_path, self.is_folder)

    def process_dicom_data(self, path, is_folder):
        self.name_dict.clear()
        self.treeWidget.clear()
        
        # Create nested defaultdict for organizing files
        

        def process_dicom_file(file_path = None):
            try:
                dicom = pydicom.dcmread(file_path)
                if len(self.patientInfo) == 0:
                    self.patientInfo.append(str(dicom.get('PatientID', 'Bilgi bulunamadı')))
                    self.patientInfo.append(str(dicom.get('PatientName', 'Bilgi bulunamadı')))
                    self.patientInfo.append(str(dicom.get('RequestedProcedureDescription', 'Bilgi bulunamadı')))
                    x = dicom.get('StudyDate', 'Bilgi bulunamadı')
                    formatted_date = f"{x[6:8]}-{x[4:6]}-{x[:4]}"
                    self.patientInfo.append(formatted_date)
                    x = dicom.get('PatientBirthDate', 'Bilgi bulunamadı')
                    if x != "Bilgi bulunamadı" and len(x) == 8 and x.isdigit():
                        try:
                            birth_date_obj = datetime.strptime(x, "%Y%m%d")
                            today = datetime.today()
                            age = today.year - birth_date_obj.year - ((today.month, today.day) < (birth_date_obj.month, birth_date_obj.day))
                            self.patientInfo.append(str(age))
                        except ValueError:
                            self.patientInfo.append("Bilinmiyor")
                    else:
                        self.patientInfo.append("Bilinmiyor")
                    if dicom.get('PatientSex', 'Bilgi bulunamadı') == 'M':
                        self.patientInfo.append("Erkek")
                    else:
                        self.patientInfo.append("Kadin")
                    

                metadata_dict = {}

                # Her DICOM elemanını işle
                for elem in dicom:
                    metadata = {
                    "tag": str(elem.tag),
                    "name": str(elem.name),          # DICOM tag (örneğin: '0x00080020')
                    "value": str(elem.value)       # DICOM elemanının değeri (örneğin: '20240101')
                    }
                    if str(elem.tag) == '(0029, 1010)' or str(elem.tag) == '(0029, 1020)' or str(elem.tag) == '(7fe0, 0010)':
                        continue
                    if elem.tag not in metadata_dict :
                        metadata_dict[str(elem.tag)] = metadata  # Eleman adı anahtar olarak kullanılır
                json_metadata = json.dumps(metadata_dict, indent=4)
                reMetadata = json.loads(json_metadata)

                pat_name = str(reMetadata.get('(0010, 0010)').get('value', 'Bilgi bulunamadı'))
                study_desc = str(reMetadata.get('(0008, 1030)', {}).get('value', 'Bilgi bulunamadı'))
                series_desc = str(reMetadata.get('(0008, 103e)', {}).get('value', 'Bilgi bulunamadı'))
                instance_num = int(reMetadata.get('(0020, 0013)', {}).get('value', 0))
                
                pixelArr = dicom.pixel_array.astype(np.float32)

                rescale_slope = float(dicom.RescaleSlope) if 'RescaleSlope' in dicom else 1.0
                rescale_intercept = float(dicom.RescaleIntercept) if 'RescaleIntercept' in dicom else 0.0
                
                # Piksel değerlerini Hounsfield Değerlerine dönüştür
                hu_image = (pixelArr * rescale_slope) + rescale_intercept

                f_name, f_ext = os.path.splitext(file_path)
                # Store DICOM file info
                file_info = {
                    'file_name': os.path.basename(f_name),
                    'pixel_array': hu_image,
                    'filtered_array': None,
                    'dicom': reMetadata,
                    'instance_num': instance_num
                }
                
                self.name_dict[pat_name][study_desc][series_desc].append(file_info)
                return True
            except Exception as e:
                print(f"Hata oluştu: {e}")
                return False

        # Process files based on selection type
        if is_folder:
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    process_dicom_file(file_path)
        else:
            if process_dicom_file(path):
                path = os.path.dirname(path)  # Get the containing folder for status
            else:
                #self.status_label.setText("Error: Selected file is not a valid DICOM file")
                return
        self.setTree()

        
    def setTree(self):
        self.treeWidget.clear()
        # Create tree structure
        for pat_name, study_dict in self.name_dict.items():
            name_item = QTreeWidgetItem([f"{pat_name}"])
            self.treeWidget.addTopLevelItem(name_item)

            for study_desc, series_dict in study_dict.items():
                # Create study-level item
                study_item = QTreeWidgetItem([f"{study_desc}"])
                name_item.addChild(study_item)
                
                for series_desc, files in series_dict.items():
                    # Create series-level item
                    series_item = QTreeWidgetItem([f"{series_desc}"])
                    study_item.addChild(series_item)
                    files = sorted(files, key=lambda x: x['instance_num'])
                    for file_info in files:
                            # Create instance-level item with combined information
                            info_text = (f"{file_info['file_name']}")
                            instance_item = QTreeWidgetItem([info_text])
                            series_item.addChild(instance_item)

        # Expand to study level
        self.treeWidget.expandToDepth(0)


    def wheelEvent(self, event):
        x = event.x()
        y = event.y()
        if (self.current_image is not None and self.dbSearch is False and 330 <= x <= 980 and 100 <= y <= 750) or (self.current_dicom is not None and self.dbSearch is True and 330 <= x <= 980 and 100 <= y <= 750):
            # Zoom functionality
            zoom_in_factor = 1.1
            zoom_out_factor = 1 / zoom_in_factor

            if event.angleDelta().y() > 0:
                # Zoom in
                self.graphicsView.scale(zoom_in_factor, zoom_in_factor)
                self.zoom_factor *= zoom_in_factor
            else:
                # Check if zooming out would go below original size
                new_zoom = self.zoom_factor * zoom_out_factor
                if new_zoom >= 1.0:
                    # Only zoom out if we're still above original size
                    self.graphicsView.scale(zoom_out_factor, zoom_out_factor)
                    self.zoom_factor = new_zoom
                else:
                    # Reset to original size if we would go below
                    scale_factor = 1.0 / self.zoom_factor
                    self.graphicsView.scale(scale_factor, scale_factor)
                    self.zoom_factor = 1.0
            self.imageZoom.setText(f"Zoom: %{int(100 * self.zoom_factor)}")

    def update_image_contrast(self):
        if self.dbSearch is False:
            if self.current_image is not None:
                # Convert to PIL Image for contrast adjustment
                image = Image.fromarray(self.current_image)
                enhancer = ImageEnhance.Contrast(image)
                adjusted_image = enhancer.enhance(self.contrast_factor)
                
                # Convert back to QImage and display
                adjusted_array = np.array(adjusted_image)
                height, width, channels = adjusted_array.shape
                qimage = QImage(adjusted_array.data, width, height, width * channels, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                scene = QGraphicsScene()
                scene.addPixmap(pixmap)

                self.graphicsView.setScene(scene)
                self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            if self.current_dicom is not None:
                self.zoom_factor = 1.0
                self.imageZoom.setText(f"Zoom: %{int(100 * self.zoom_factor)}")
                
                image = Image.open(self.current_dicom['image_path'])
                image = image.convert('RGB')
                
                enhancer = ImageEnhance.Contrast(image)
                adjusted_image = enhancer.enhance(self.contrast_factor)
                
                # Convert back to QImage and display
                adjusted_array = np.array(adjusted_image)
                height, width, channels = adjusted_array.shape
                qimage = QImage(adjusted_array.data, width, height, width * channels, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                scene = QGraphicsScene()
                scene.addPixmap(pixmap)
                
        self.zoom_factor = 1.0
        self.imageZoom.setText(f"Zoom: %{int(100 * self.zoom_factor)}")
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        x = round(self.contrast_factor,2) / 2
        self.imageContrast.setText(f"Contrast: {x}")

    def openWidget(self):
        self.Wapp.show()
        self.Wapp.fill_tableV2(self.current_dicom['dicom'])
            
    def keyPressEvent(self, event):
         # Seçili öğeyi al
        current_item = self.treeWidget.currentItem()

        if current_item:
            # Eğer seçili öğe bir series_item ise
            if current_item.parent() and current_item.parent().parent():  # Yani, 'series_item' seviyesinde
                parent = current_item.parent()
                if parent is None:  # Root level items
                    parent = current_item.treeWidget().invisibleRootItem()
                
                # Get the index of the current item
                current_index = parent.indexOfChild(current_item)
                # Find previous item
                previous_item = parent.child(current_index - 1) if current_index > 0 else None
                
                # Find next item
                next_item = parent.child(current_index + 1) if current_index < parent.childCount() - 1 else None
                # Yukarı tuşu
                if event.key() == Qt.Key_Left or event.key() == Qt.Key_Up:
                    if previous_item is None:
                        try:
                            self.treeWidget.setCurrentItem(current_item)
                        except Exception as e:
                            print(f"Error: {str(e)}")
                    else:
                        self.treeWidget.setCurrentItem(previous_item)

                # Aşağı tuşu
                elif event.key() == Qt.Key_Right or event.key() == Qt.Key_Down:
                    if next_item is None:
                        try:
                            self.treeWidget.setCurrentItem(current_item)
                        except Exception as e:
                            print(f"Error: {str(e)}")
                    else:
                        self.treeWidget.setCurrentItem(next_item)
            
        event.accept()

    def apply_window(image_array, center, width):
        lower_bound = center - (width / 2)
        upper_bound = center + (width / 2)
        windowed_array = np.clip(image_array, lower_bound, upper_bound)
        normalized_array = ((windowed_array - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
        return normalized_array
    
    def parse_window_values(value, check):
        """Window Center ve Window Width değerlerini ayrıştır."""
        if value == "Bilgi bulunamadı" and check == 0:
            return 40
        elif value == "Bilgi bulunamadı" and check == 1:
            return 400

        if "," in value:
            cleaned_string = '[' + ', '.join([str(int(item)) for item in value.strip('[]').split(',')]) + ']'
            value = json.loads(cleaned_string)
            print(value)
            return float(value[0])
        else:
            return float(value)
        
        

    def show_list_items(self):
        if self.current_dicom is None:
            return
        if self.dbSearch is False:
            image_array = self.current_dicom['pixel_array']

            if self.filterCheck:
                image_array = self.current_dicom['filtered_array']
                pixel_array = image_array
                
            if (np.array_equal(image_array, self.current_dicom['pixel_array'])):
                # Rescale Slope ve Rescale Intercept değerlerini kontrol et
                """rescale_slope = float(self.current_dicom['dicom'].get('(0028, 1053)').get('value', '1'))
                rescale_intercept = float(self.current_dicom['dicom'].get('(0028, 1052)').get('value', '0'))
                
                # Piksel değerlerini Hounsfield Değerlerine dönüştür
                hu_image = (image_array * rescale_slope) + rescale_intercept"""

                try:
                    window_center = MainWindowApp.parse_window_values(str(self.current_dicom['dicom'].get('(0028, 1050)').get('value', 'Bilgi bulunamadı')), 0)
                    window_width = MainWindowApp.parse_window_values(str(self.current_dicom['dicom'].get('(0028, 1051)').get('value', 'Bilgi bulunamadı')), 1)
                except Exception as e:
                    print(f"Window değerleri ayrıştırılırken hata oluştu: {e}")

                pixel_array = MainWindowApp.apply_window(image_array, window_center, window_width)
            else:
                pixel_array = (image_array / np.max(image_array) * 255).astype(np.uint8)

            if len(pixel_array.shape) == 2:  # Gri tonlamalı (H, W)
                pixel_array = np.stack((pixel_array,) * 3, axis=-1)  # (H, W, 3)

            self.current_image = pixel_array  # Store current image
            

            height, width, channels = pixel_array.shape
            qimage = QImage(pixel_array.data, width, height, width * channels, QImage.Format_RGB888)
            
            # QPixmap'e dönüştür ve QGraphicsView'de göster
            pixmap = QPixmap.fromImage(qimage)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)  # Görüntüyü çerçeveye sığdır
        else:
            scene = QGraphicsScene()
            pixmap = QPixmap(self.current_dicom['image_path'])  # Burada 'image.png' dosyasının yolunu belirtin
            scene.addPixmap(pixmap)
            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        self.contrast_factor = 1.0  # Reset contrast factor
        self.imageZoom.setText(f"Zoom: %100")
        self.imageContrast.setText(f"Contrast: 0.5")
        self.contrastSlider.setValue(50)
    
        self.zoom_factor = 1.0  # Reset zoom when showing new image
        
        ser_description = self.current_dicom['dicom'].get('(0008, 103e)', {}).get('value', 'Bilgi bulunamadı')
        std_description = self.current_dicom['dicom'].get('(0008, 1030)', {}).get('value', 'Bilgi bulunamadı')

        self.hidePatientInfo()

        self.patientSeries.setText(str(ser_description))
        self.patientStudy.setText(str(std_description))
        
        if (self.is_folder and self.dbSearch is False) or self.dbSearch is True:
            self.patientInstanceNum.setText(f"Series: {self.current_dicom['instance_num']}/{self.instanceLen}")
        else:
            self.patientInstanceNum.setText(f"Series: 1/1")

app = QtWidgets.QApplication(sys.argv)        
mainApp = MainWindowApp()

if __name__ == "__main__":
    mainApp.show()
    sys.exit(app.exec_())
