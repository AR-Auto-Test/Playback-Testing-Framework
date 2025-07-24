import subprocess
import threading
import time
import os
import signal
import sys
import random
import psutil
from datetime import datetime

# Global variable
current_app = "ru.sibakovskaya.vrgirls"
current_recording = "unknown"


# Global flags
exit_program = False
pause_simulation = False
recording_end = False

# Simulation screen event area
# Portrait Setting
_TOP = 900
_BOTTOM = 1500
_LEFT = 300
_RIGHT = 800
# Landscape Setting
#_TOP = 250
#_BOTTOM = 850
#_LEFT = 1100
#_RIGHT = 1900

def main():
    init()
    
    print("Starting screen recording...")
    signal.signal(signal.SIGINT, signal_handler)

    # Starting threads
    touch_thread = threading.Thread(target=simulate_touch_and_swipe)
    record_thread = threading.Thread(target=screen_recording)
    log_thread = threading.Thread(target=monitor_logs)

    touch_thread.start()
    record_thread.start()
    log_thread.start()

    touch_thread.join()
    record_thread.join()
    log_thread.join()
    
    pull_all_record()
    
        
def init():
    # Clear the old log, in case of delayed log message output
    print("[INFO] Cleaning old log...")
    os.system("adb logcat -c")
    # In case of old recording exists due to exception
    try:
        mp4_files = subprocess.check_output("adb shell ls /sdcard/*.mp4".split()).decode().splitlines()

        # Pull each file to the local directory and then delete it from the device
        for file in mp4_files:
            # Delete the file from the device
            os.system(f"adb shell rm {file}")
    except:
        print("No mp4 File Left on Device")
    
"""
Handling Ctrl+C
"""
def signal_handler(sig, frame):
    global exit_program
    exit_program = True
    print("Exiting program...")  
    

def checkHomeScreen():
    try:
        # Run adb command to get the current foreground activity
        result = subprocess.check_output("adb shell dumpsys activity activities | grep mFocusedApp", shell=True).decode('utf-8')

        # Check if the current focus is on the home screen
        # This is a more general check for any launcher activity
        return 'nexuslauncher' in result.lower()
    except subprocess.CalledProcessError:
        print("Failed to execute adb command")
        return False

def screen_recording():
    global exit_program, recording_end
    while not exit_program:
        print("[INFO] Start New Recording")
        recorder = subprocess.Popen("adb shell screenrecord /sdcard/temp_screenrecord.mp4 --time-limit 180", shell=True)
        
        # Print the beginning of the recording
        # For the purpose of locating in data collecting later
        os.system("adb shell log -t 'XiaoyiYang_data' 'XiaoyiYang_playback_start'")
        
        # Wait for recording to finish or for stop event
        while recorder.poll() is None and not exit_program and not recording_end:
            time.sleep(1)


        # Pull and delete the recording
        if recording_end:
            recording_end = False
            
        print("[INFO] Terminating Recorder")
        #recorder.terminate()
        
        # Terminate the adb screen record
        try:
            parent_proc = psutil.Process(recorder.pid)
            for child_proc in parent_proc.children(recursive = True):
                child_proc.kill()
            parent_proc.kill()
        except:
            print("[WARN] This recorder has already ended")
        
        stop_record_thread = threading.Thread(target=stopRecordScreen)
        stop_record_thread.start()
        #stopRecordScreen()
    # Ensure the recording pulling can be completed
    stop_record_thread.join()
    print("[END] Record Thread")

    
        
def stopRecordScreen():
    global current_recording
    print("[INFO] Saving Screen Record...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_filename = f"{current_recording}_{timestamp}.mp4"
    
    # Simply rename the recordings, no pull
    # pull all recordings in the end
    subprocess.run(f"adb shell mv /sdcard/temp_screenrecord.mp4 /sdcard/{local_filename}", shell=True)
    
    # Pull the test recording from phone
    # May cause the mp4 file broken
    #subprocess.run(f"adb pull /sdcard/temp_screenrecord.mp4 ../../screenRecordings/{local_filename}", shell=True)
    
    
    
            
    
"""
Randomly select Touch or Swipe event

"""
def simulate_touch_and_swipe():
    global pause_simulation, exit_program
    ratio = 0.9
    #delay = 1.8
    # For ru.vrgirls, this app will crash if replace model to often
    delay = 3
    while not exit_program:
        if not pause_simulation:
            if random.random() <= ratio:
                simulate_touch()
            else:
                for i in range(5):
                    simulate_swipe()
        time.sleep(delay) 
    print("[END] Touch Thread")
    
    
"""
Used to randomly generate a screen touch event
"""
def simulate_touch():
    x_touch = random.randint(_LEFT, _RIGHT)
    y_touch = random.randint(_TOP, _BOTTOM)
    cmd = "adb shell input tap %d %d" % (
        x_touch,
        y_touch
    )
    #print(cmd)
    os.system(cmd)
    
"""
Used to randomly generate a screen swipe event
"""
def simulate_swipe():
    x_start = random.randint(_LEFT, _RIGHT)
    y_start = random.randint(_TOP, _BOTTOM)
    x_end = random.randint(_LEFT, _RIGHT)
    y_end = random.randint(_TOP, _BOTTOM)
    duration = 500
    cmd = "adb shell input touchscreen swipe %d %d %d %d %d" % (
        x_start,
        y_start,
        x_end,
        y_end,
        duration
    )
    #print(cmd)
    os.system(cmd)

"""
Monitor the log in real time to check when the current playback has ended
"""
def monitor_logs():
    global pause_simulation, exit_program, recording_end, current_recording
    with subprocess.Popen(["adb", "logcat", "*:I"], stdout=subprocess.PIPE, bufsize=1, text=True) as logcat:
        while not exit_program:
            line = logcat.stdout.readline()
            if not line:
                break  # Stop if no more output
                
            if "Xiaoyi Yang_Playback_Path" in line:
                temp_path = line.split(":")[-1].strip()
                current_recording = temp_path.split("/")[-1].strip()[0:-4]
                print("[INFO] Finishing Recording:", current_recording)
                print("[INFO] Switching to Next Recording")
                continue
                
            if "XiaoyiYang_Playback_End" in line:
                print("[INFO] Changing Recording")
                recording_end = True
                continue
                
            if "beginning of crash" in line:
                print("[WARN] App Crashed")
                continue
                
            if "XiaoyiYang_Playback_Terminated" in line:
                print("[INFO] Playback Treminating...")
                exit_program = True
                break
        try:
            parent_proc = psutil.Process(logcat.pid)
            for child_proc in parent_proc.children(recursive = True):
                child_proc.kill()
            parent_proc.kill()
        except:
            print("[WARN] Logcat Has Already Ended")
    print("[END] Log Thread")
                
def pull_all_record():
    print("[INFO] Testing End")
    print("[INFO] Start To Pull All Records")
    
    # Find all mp4 files in /sdcard/
    mp4_files = subprocess.check_output("adb shell ls /sdcard/*.mp4".split()).decode().splitlines()

    # Pull each file to the local directory and then delete it from the device
    for file in mp4_files:
        # Pull the file
        os.system(f"adb pull {file} ../../screenRecordings/{current_app}/outdoor/big")
        #os.system(f"adb pull {file} ../../screenRecordings/com.rooom/outdoor/big")

        # Delete the file from the device
        os.system(f"adb shell rm {file}")

if __name__ == "__main__":
    main()