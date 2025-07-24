"""
To install PyQt on your device, use "pip install PyQt5" 
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import cv2
import os
import queue
import time

MAX_QUEUE_SIZE = 200

# Change the path if needed
videos_dir = "video_clips"
gt_file = "gt.txt"

class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self, video_path, frame_queue):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while not self.stop_event and cap.isOpened():
            if self.frame_queue.qsize() < MAX_QUEUE_SIZE:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (350, 600))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_signal.emit(q_image)
                    
                    time.sleep(0.01) 
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()

    def stop(self):
        self.stop_event = True

class VideoClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Classifier")
        self.setGeometry(100, 100, 1200, 800)

        self.progress_label = QLabel("", self)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 14pt; font-weight: bold;")

        self.video_name_label = QLabel("", self)
        self.video_name_label.setAlignment(Qt.AlignCenter)
        self.video_name_label.setStyleSheet("font-size: 12pt;")

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.clicked.connect(self.delete_video)

        self.performance_label = QLabel("Performance", self)
        self.performance_buttons = [QPushButton(str(i), self) for i in range(1, 5)]
        for button in self.performance_buttons:
            button.clicked.connect(lambda _, b=button: self.select_performance(b.text()))

        self.placement_label = QLabel("Placement", self)
        self.placement_buttons = [QPushButton(str(i), self) for i in range(1, 5)]
        for button in self.placement_buttons:
            button.clicked.connect(lambda _, b=button: self.select_placement(b.text()))

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_classification)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close)

        performance_layout = QHBoxLayout()
        performance_layout.addWidget(self.performance_label)
        for button in self.performance_buttons:
            performance_layout.addWidget(button)

        placement_layout = QHBoxLayout()
        placement_layout.addWidget(self.placement_label)
        for button in self.placement_buttons:
            placement_layout.addWidget(button)

        button_layout = QHBoxLayout()
        
        button_layout.addWidget(self.delete_button)
        button_layout.addLayout(performance_layout)
        button_layout.addLayout(placement_layout)
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.quit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.video_name_label)
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.video_info = self.load_video_info()
        self.total_num = len(self.video_info)
        self.classified_num = sum(1 for _, label in self.video_info.items() if label)

        self.current_video_path = None
        self.frame_queue = queue.Queue()
        self.video_thread = None

        self.performance_selection = None
        self.placement_selection = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.update_progress()
        self.load_next_video()

    def load_video_info(self):
        if not os.path.exists(gt_file):
            return {}
        with open(gt_file, "r") as f:
            lines = f.readlines()
        video_info = {line.split(':')[0]: line.strip().split(':')[1] for line in lines}
        return video_info

    def load_next_video(self):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        self.clear_queue()

        video_paths = [video for video, label in self.video_info.items() if not label]
        if video_paths:
            self.current_video_path = video_paths[0] + ".mp4"
            self.video_name_label.setText(os.path.basename(self.current_video_path))
            self.video_thread = VideoThread(os.path.join(videos_dir, self.current_video_path), self.frame_queue)
            self.video_thread.frame_signal.connect(self.display_frame)
            self.video_thread.start()
            self.reset_selections()
        else:
            self.video_label.clear()
            self.progress_label.setText("Task completed")

    def clear_queue(self):
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def display_frame(self, q_image):
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def update_frame(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.display_frame(frame)

    def select_performance(self, label):
        self.performance_selection = label
        for button in self.performance_buttons:
            button.setStyleSheet("background-color: #FFFFFF;" if button.text() != label else "background-color: #7CD9B8;")

    def select_placement(self, label):
        self.placement_selection = label
        for button in self.placement_buttons:
            button.setStyleSheet("background-color: #FFFFFF;" if button.text() != label else "background-color: #7CD9B8;")

    def confirm_classification(self):
        if self.performance_selection is not None and self.placement_selection is not None:
            video_path_without_ext = os.path.splitext(self.current_video_path)[0]
            self.video_info[video_path_without_ext] = f"{self.performance_selection},{self.placement_selection}"
            self.classified_num += 1

            with open(gt_file, "r") as file:
                lines = file.readlines()
            with open(gt_file, "w") as file:
                for line in lines:
                    video_path = line.split(':')[0]
                    if video_path == video_path_without_ext:
                        file.write(f"{video_path}:{self.performance_selection},{self.placement_selection}\n")
                    else:
                        file.write(line)

            self.update_progress()

            if self.classified_num < self.total_num:
                self.load_next_video()
            else:
                self.video_label.clear()
                self.progress_label.setText("Task completed")

    def delete_video(self):
        video_path_without_ext = os.path.splitext(self.current_video_path)[0]
        self.video_info[video_path_without_ext] = "delete"
        self.classified_num += 1

        with open(gt_file, "r") as file:
            lines = file.readlines()
        with open(gt_file, "w") as file:
            for line in lines:
                video_path = line.split(':')[0]
                if video_path == video_path_without_ext:
                    file.write(f"{video_path}:delete\n")
                else:
                    file.write(line)

        self.update_progress()

        if self.classified_num < self.total_num:
            self.load_next_video()
        else:
            self.video_label.clear()
            self.progress_label.setText("Task completed")

    def reset_selections(self):
        self.performance_selection = None
        self.placement_selection = None
        for button in self.performance_buttons + self.placement_buttons:
            button.setStyleSheet("background-color: #FFFFFF;")

    def update_progress(self):
        self.progress_label.setText(f"{self.classified_num}/{self.total_num}")

    def closeEvent(self, event):
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    classifier = VideoClassifier()
    classifier.show()
    sys.exit(app.exec_())