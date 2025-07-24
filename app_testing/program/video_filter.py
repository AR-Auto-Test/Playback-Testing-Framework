import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import cv2
import os

videos_dir = "../video_clips"
output_file = "data_video_list.txt"

class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, video_path, delay=30):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.delay = delay

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (360, 800))
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_signal.emit(qt_image)
                QThread.msleep(self.delay)  # 添加延迟
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def stop(self):
        self.cap.release()
        self.quit()
        self.wait()
        
    def set_delay(self, delay):
        self.delay = delay

class VideoClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Filter")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: gray;")

        self.video_name_label = QLabel(self)
        self.video_name_label.setAlignment(Qt.AlignCenter)
        self.video_name_label.setStyleSheet("font-size: 12pt; padding: 10px;")

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.keep_button = QPushButton("Keep", self)
        self.keep_button.setStyleSheet("color: black;")
        self.keep_button.clicked.connect(lambda: self.classify_video('keep'))

        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setStyleSheet("color: black;")
        self.delete_button.clicked.connect(lambda: self.classify_video('delete'))

        self.quit_button = QPushButton("Confirm and Quit", self)
        self.quit_button.setStyleSheet("color: black;")
        self.quit_button.clicked.connect(self.confirm_and_quit)

        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setRange(5, 50)
        self.speed_slider.setValue(30)
        self.speed_slider.valueChanged.connect(self.set_video_speed)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.keep_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.quit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_name_label)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.speed_slider)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.video_paths = self.load_video_paths()
        self.current_video_path = None
        self.video_thread = None

        self.load_next_video()

    def set_video_speed(self, value):
        if self.video_thread is not None:
            self.video_thread.set_delay(value)
            
    def load_video_paths(self):
        video_paths = [os.path.join(videos_dir, filename) for filename in os.listdir(videos_dir) if filename.endswith(".mp4")]
        return video_paths

    def load_next_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.deleteLater()
            self.video_thread = None

        if self.video_paths:
            self.current_video_path = self.video_paths.pop(0)
            while self.is_classified(self.current_video_path):
                self.current_video_path = self.video_paths.pop(0)
            self.video_name_label.setText(os.path.basename(self.current_video_path))
            self.video_thread = VideoThread(self.current_video_path, delay=self.speed_slider.value())
            self.video_thread.frame_signal.connect(self.update_video_frame)
            self.video_thread.start()

    def update_video_frame(self, frame):
        self.video_label.setPixmap(QPixmap.fromImage(frame))

    def classify_video(self, label):
        with open(output_file, "a") as f:
            f.write(f"{os.path.splitext(self.current_video_path)[0]}:{label}\n")
        self.load_next_video()

    def is_classified(self, video_path):
        if not os.path.exists(output_file):
            return False
        with open(output_file, "r") as f:
            classified_videos = f.read()
        return os.path.splitext(video_path)[0] in classified_videos

    def confirm_and_quit(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier = VideoClassifier()
    classifier.show()
    sys.exit(app.exec_())