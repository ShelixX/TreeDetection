import torch.cuda
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QTimer
import sys
import cv2
from ultralytics import YOLO

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle("Trees detection")
        self.x = 1080
        self.y = 720
        self.stopVideo = False
        self.setFixedSize(self.x, self.y)

        p = self.palette()
        p.setColor(QPalette.Window, QColor("#2f2f2f"))
        p.setColor(QPalette.Button, QColor("#444444"))
        p.setColor(QPalette.ButtonText, QColor("#ffffff"))
        self.setPalette(p)

        self.init_ui()

    def init_ui(self):
        openVideoBtn = QPushButton('Open Video')
        openVideoBtn.clicked.connect(self.open_video_file)
        self.style_button(openVideoBtn)

        openImageBtn = QPushButton('Open Image')
        openImageBtn.clicked.connect(self.open_image_file)
        self.style_button(openImageBtn)

        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.style_label(self.label)

        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        hboxLayout.addWidget(openVideoBtn)
        hboxLayout.addWidget(openImageBtn)

        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.imageLabel)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)
        self.setLayout(vboxLayout)
        self.imageLabel.setVisible(False)
        self.label.setVisible(False)

    def style_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #777777;
            }
        """)

    def style_label(self, label):
        label.setStyleSheet("""
            QLabel {
                background-color: #555555;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size:32px;
            }
        """)

    def open_video_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4)")
        if filename:
            self.imageLabel.setVisible(True)
            self.stopVideo = False
            self.frames = []
            cap = cv2.VideoCapture(filename)
            while cap.isOpened() and not self.stopVideo:
                success, frame = cap.read()
                if success:
                    frame = cv2.resize(frame, (1920, 1080), fx=0, fy=0,
                                       interpolation=cv2.INTER_NEAREST)
                    results = self.detect(frame)
                    self.frames.append(results[0])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
            cap.release()
            self.label.setVisible(True)
            self.current_frame_index = 0
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(17)

    def update_frame(self):
        if self.current_frame_index < len(self.frames) and not self.stopVideo:
            annotated_frame = self.frames[self.current_frame_index].plot()
            image = QImage(annotated_frame, annotated_frame.shape[1], annotated_frame.shape[0],
                           QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.x-300, self.y-200)
            self.imageLabel.setPixmap(scaled_pixmap)
            self.label.setText("Trees count: " + str(len(self.frames[self.current_frame_index].boxes.data)))
            self.current_frame_index += 1
        else:
            self.timer.stop()  # Остановка таймера, если все кадры показаны
            self.stopVideo = False

    def open_image_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if filename:
            self.stopVideo = True  # Остановка вывода кадров видео
            source = cv2.imread(filename)
            frame = cv2.resize(source, (1920, 1080), fx=0, fy=0,
                                 interpolation=cv2.INTER_NEAREST)
            results = self.detect(frame)
            annotated_frame = results[0].plot()
            image = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.x-300, self.y-200)
            self.imageLabel.setPixmap(scaled_pixmap)
            self.label.setText("Number of trees: " + str(len(results[0].boxes.data)))
            self.imageLabel.setVisible(True)
            self.label.setVisible(True)

    # Функция, которая возвращает результаты по обнаружению деревьев на картинке
    def detect(self, image):
        # Если система имеет cuda ядра:
        if torch.cuda.is_available():
            result = model(image, augment=True, imgsz=960, iou=0.2, device=0)
            # image - картинка, на которой необходимо обнаружить деревья
            # augment - улучшает надёжность обнаружения
            # imgsz - изменяет размер картинки. Правильное изменение размера может улучшить точность обнаружения и скорость обработки данных.
            # iou - меньшие значения уменьшенают количество обнаружений за счет устранения перекрывающихся ячеек, что полезно для уменьшения количества дубликатов.
            # device - указывает устройство для обработки.
            # Позволяет выбирать между CPU, конкретным GPU или другими вычислительными устройствами для выполнения модели.
            # Значение 0 указывает, что выбрана первая GPU (для выбора нескольких надо написать [0, 1, 2], в зависимости от количества доступных GPU).
            # Для выбора процессора используется запись device="cpu".
        else:
            result = model(image, augment=True, imgsz=640, iou=0.2, device="cpu")
        return result

    def closeEvent(self, event):
        sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = YOLO("model/model.pt")
    window = Window()
    window.show()
    sys.exit(app.exec_())