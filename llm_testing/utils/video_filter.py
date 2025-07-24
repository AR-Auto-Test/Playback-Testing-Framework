import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import threading
import queue

MAX_QUEUE_SIZE = 200

videos_dir = "video_clips"
output_file = "data_video_list.txt"

class VideoClassifier(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Filter")
        self.geometry("1200x800")
        
        self.progress_label = tk.Label(self, text="", font=('Arial', 14), pady=10)
        self.progress_label.pack()
        
        # 添加用于显示视频文件名的标签
        self.video_name_label = tk.Label(self, text="", font=('Arial', 12), pady=10)
        self.video_name_label.pack()

        self.video_label = tk.Label(self)
        self.video_label.pack(expand=True, fill=tk.BOTH)

        frame = tk.Frame(self)
        frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.delete_button = tk.Button(frame, text="Keep", width=10, height=10, command=lambda: self.classify_video('keep'))
        self.delete_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(frame, text="Delete", width=10, height=10, command=lambda: self.classify_video('delete'))
        self.delete_button.pack(side=tk.RIGHT)

        self.quit_button = tk.Button(self, text="Quit", width=15, command=self.confirm_and_quit)
        self.quit_button.place(x=1050, y=10)

        self.video_paths = self.load_video_paths()
        #self.video_info = self.load_video_info()
        self.total_num = len(self.video_paths)
        self.classified_num = sum(1 for _ in open(output_file))
        self.current_video_path = None
        self.frame_queue = queue.Queue()
        self.video_thread = None
        self.stop_event = threading.Event()

        self.after(100, self.check_frame_queue)
        self.update_progress()
        self.load_next_video()
        
    def load_video_paths(self):
        video_paths = [os.path.join(videos_dir, filename) for filename in os.listdir(videos_dir) if filename.endswith(".mp4")]
        return video_paths
    

    def load_next_video(self):
        if self.video_thread is not None:
            self.stop_event.set()
            self.video_thread.join()
            if self.cap is not None:
                self.cap.release() # 释放现有的视频捕获对象
        self.stop_event.clear()  # 重置事件，为下一个线程准备
            
        self.clear_queue()  # 清空队列

        if self.video_paths:
            self.current_video_path = self.video_paths.pop(0)
            while self.is_classified(self.current_video_path):
                self.current_video_path = self.video_paths.pop(0)
            self.video_name_label.config(text=os.path.basename(self.current_video_path))
            self.cap = cv2.VideoCapture(self.current_video_path)
            self.stop_event.clear()
            self.video_thread = threading.Thread(target=self.read_frames)
            self.video_thread.start()
        else:
            self.video_label.pack_forget()
            self.progress_label.config(text="Task completed")
            
    
    def clear_queue(self):
        try:
            while True:  # 循环直到队列为空，引发queue.Empty异常
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass  # 队列已经为空，可以继续

    def read_frames(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            if self.frame_queue.qsize() < MAX_QUEUE_SIZE:  # MAX_QUEUE_SIZE 是你设定的队列大小上限
                ret, frame = self.cap.read()
                if ret:
                    self.frame_queue.put(frame)
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def check_frame_queue(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame = cv2.resize(frame, (350, 600))
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.after(10, self.check_frame_queue)

    def classify_video(self, label):
        with open(output_file, "a") as f:
            f.write(f"{os.path.splitext(self.current_video_path)[0]}:{label}\n")
        
        self.classified_num += 1
        self.update_progress()
            
        if self.classified_num < self.total_num:
            self.load_next_video()
        else:
            self.video_label.pack_forget()
            self.progress_label.config(text="Task completed")
        
    def is_classified(self, video_path):
        if not os.path.exists(output_file):
            return False
        with open(output_file, "r") as f:
            classified_videos = f.read()
        return os.path.splitext(video_path)[0] in classified_videos

    def confirm_and_quit(self):
        self.stop_event.set()
        self.video_thread.join()
        self.destroy()
        
    def update_progress(self):
        self.progress_label.config(text=f"{self.classified_num}/{self.total_num}")

if __name__ == "__main__":
    app = VideoClassifier()
    app.mainloop()
