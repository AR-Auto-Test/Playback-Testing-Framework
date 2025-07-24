import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import threading
import queue

MAX_QUEUE_SIZE = 200

# Configuration for classification categories
CATEGORIES = [
    {"name": "placement", "options": [0, 1], "display_name": "Placement"},
    {"name": "movement", "options": [0, 1], "display_name": "Movement"},
    {"name": "occlusion", "options": [0, 1], "display_name": "Occlusion"},
    {"name": "lighting", "options": [0, 1], "display_name": "Lighting"},
    {"name": "rendering", "options": [0, 1], "display_name": "Rendering"},
    {"name": "black", "options": [0, 1], "display_name": "Black"}
]

# Change the path if needed
videos_dir = "video_clips_6s_2"
gt_file = "pre_process/gt_llm.txt"

class VideoClassifier(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Classifier")
        self.geometry("1200x800")
        
        # Initialize variables
        self.current_video_path = None
        self.frame_queue = queue.Queue()
        self.video_thread = None
        self.stop_event = threading.Event()
        self.selections = {cat["name"]: None for cat in CATEGORIES}
        self.category_buttons = {cat["name"]: [] for cat in CATEGORIES}
        
        # Setup UI components
        self.setup_ui()
        
        # Load video information
        self.video_info = self.load_video_info()
        self.total_num = len(self.video_info)
        self.classified_num = sum(1 for _, label in self.video_info.items() if label)
        
        # Start processing
        self.after(100, self.check_frame_queue)
        self.update_progress()
        self.load_next_video()

    def setup_ui(self):
        # Progress and video name labels
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(top_frame, text="", font=('Arial', 14))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.quit_button = tk.Button(top_frame, text="Quit", width=15, command=self.confirm_and_quit)
        self.quit_button.pack(side=tk.RIGHT, padx=10)

        self.video_name_label = tk.Label(self, text="", font=('Arial', 12))
        self.video_name_label.pack(pady=5)

        # Video display
        self.video_label = tk.Label(self)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # Classification buttons in two rows
        self.create_classification_frames()

        # Control buttons at bottom
        button_frame = tk.Frame(self)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.delete_button = tk.Button(button_frame, text="Delete", width=10, height=2,
                                     command=self.delete_video)
        self.delete_button.pack(side=tk.RIGHT, padx=5)

        self.confirm_button = tk.Button(button_frame, text="Confirm", width=10, height=2,
                                      command=self.confirm_classification)
        self.confirm_button.pack(side=tk.RIGHT, padx=5)

    def create_classification_frames(self):
        # Create main container for classification buttons
        container = tk.Frame(self)
        container.pack(fill=tk.X, pady=5)

        # First row frame
        row1_frame = tk.Frame(container)
        row1_frame.pack(fill=tk.X, pady=2)

        # Second row frame
        row2_frame = tk.Frame(container)
        row2_frame.pack(fill=tk.X, pady=2)

        # Split categories into two rows
        half = len(CATEGORIES) // 2
        first_row = CATEGORIES[:half]
        second_row = CATEGORIES[half:]

        # Create first row of categories
        for category in first_row:
            frame = tk.Frame(row1_frame)
            frame.pack(side=tk.LEFT, padx=10)

            label = tk.Label(frame, text=category["display_name"], width=10)
            label.pack()

            button_frame = tk.Frame(frame)
            button_frame.pack()

            self.category_buttons[category["name"]] = []
            for option in category["options"]:
                btn = tk.Button(
                    button_frame,
                    text=str(option),
                    width=10,
                    command=lambda cat=category["name"], opt=option: self.select_option(cat, opt)
                )
                btn.pack(side=tk.LEFT, padx=2)
                self.category_buttons[category["name"]].append(btn)

        # Create second row of categories
        for category in second_row:
            frame = tk.Frame(row2_frame)
            frame.pack(side=tk.LEFT, padx=10)

            label = tk.Label(frame, text=category["display_name"], width=10)
            label.pack()

            button_frame = tk.Frame(frame)
            button_frame.pack()

            self.category_buttons[category["name"]] = []
            for option in category["options"]:
                btn = tk.Button(
                    button_frame,
                    text=str(option),
                    width=10,
                    command=lambda cat=category["name"], opt=option: self.select_option(cat, opt)
                )
                btn.pack(side=tk.LEFT, padx=2)
                self.category_buttons[category["name"]].append(btn)

    def check_frame_queue(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame = cv2.resize(frame, (270, 480))  # Adjusted size
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.after(10, self.check_frame_queue)
    

    def get_video_files(self):
        if not os.path.exists(videos_dir):
            messagebox.showerror("Error", f"Video directory {videos_dir} not found!")
            return []
        
        video_files = [f[:-4] for f in os.listdir(videos_dir) if f.endswith('.mp4')]
        return video_files

    def load_video_info(self):
        all_videos = self.get_video_files()
        video_info = {}

        if not all_videos:
            return video_info

        # Initialize all videos as unlabeled
        for video in all_videos:
            video_info[video] = ""

        # Load existing labels if gt_file exists
        if os.path.exists(gt_file):
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        video_path, label = line.strip().split(':')
                        if video_path in video_info:
                            video_info[video_path] = label

        # Create gt_file if it doesn't exist
        if not os.path.exists(gt_file):
            try:
                with open(gt_file, "w") as f:
                    for video in all_videos:
                        f.write(f"{video}:\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create {gt_file}: {str(e)}")

        return video_info

    def select_option(self, category, value):
        self.selections[category] = value
        for button in self.category_buttons[category]:
            if button['text'] == str(value):
                button.config(relief=tk.SUNKEN)
            else:
                button.config(relief=tk.RAISED)

    def confirm_classification(self):
        if None not in self.selections.values():
            video_path_without_ext = os.path.splitext(self.current_video_path)[0]
            
            classification = ",".join(str(self.selections[cat["name"]]) 
                                   for cat in CATEGORIES)
            
            self.video_info[video_path_without_ext] = classification
            self.classified_num += 1

            with open(gt_file, "r") as file:
                lines = file.readlines()
            with open(gt_file, "w") as file:
                for line in lines:
                    video_path = line.split(':')[0]
                    if video_path == video_path_without_ext:
                        file.write(f"{video_path}:{classification}\n")
                    else:
                        file.write(line)

            self.update_progress()

            if self.classified_num < self.total_num:
                self.load_next_video()
            else:
                self.video_label.pack_forget()
                self.progress_label.config(text="Task completed")

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
            self.video_label.pack_forget()
            self.progress_label.config(text="Task completed")

    def reset_selections(self):
        """Reset all selections to default value 1"""
        for category in self.selections:
            self.selections[category] = 1  # 默认设为1
            for button in self.category_buttons[category]:
                if button['text'] == "1":
                    button.config(relief=tk.SUNKEN)  # 选中状态
                else:
                    button.config(relief=tk.RAISED)  # 未选中状态

    def load_next_video(self):
        if self.video_thread is not None:
            self.stop_event.set()
            self.video_thread.join()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        self.stop_event.clear()
        self.clear_queue()
        
        video_paths = [video for video, label in self.video_info.items() if not label.strip()]
        if video_paths:
            self.current_video_path = video_paths[0] + ".mp4"
            video_full_path = os.path.join(videos_dir, self.current_video_path)
            
            if not os.path.exists(video_full_path):
                messagebox.showerror("Error", f"Video file not found: {video_full_path}")
                return
                
            self.video_name_label.config(text=os.path.basename(self.current_video_path))
            self.cap = cv2.VideoCapture(video_full_path)
            self.stop_event.clear()
            self.video_thread = threading.Thread(target=self.read_frames)
            self.video_thread.start()
            self.reset_selections()
        else:
            self.video_label.pack_forget()
            self.progress_label.config(text="Task completed")

    def clear_queue(self):
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

    def read_frames(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            if self.frame_queue.qsize() < MAX_QUEUE_SIZE:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_queue.put(frame)
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_progress(self):
        self.progress_label.config(text=f"{self.classified_num}/{self.total_num}")

    def confirm_and_quit(self):
        if self.video_thread is not None:
            self.stop_event.set()
            self.video_thread.join()
        self.destroy()

if __name__ == "__main__":
    app = VideoClassifier()
    app.mainloop()