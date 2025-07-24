import sys, os
import subprocess

# Assume thess directories exists on the Android device, no validation in this program
scnshot_path = "/storage/emulated/0/Pictures/screenshots/"
dest_path = "../screenshots/"


def main():
    try:
        # Clear the screenshots every time during experimenting
        # Comment out this method when in collecting process
        clearComputer()
        
        # Get the list of existing screenshot on the device
        lst = getList()
        
        # Pull the screenshots from device using adb
        download = pullScreenshots(lst)
        
        # Clear the screenshots every time during experimenting
        # Comment out this method when in collecting process
        clearDevice(download)
    except Exception as e:
        print(e)
        
def clearDevice(run):
    if not run:
        return
    # adb shell rm /storage/emulated/0/Pictures/screenshots/*
    cmd = ['adb', 'shell', 'rm', scnshot_path + '*']
    
    subprocess.run(cmd)
    
def clearComputer():
    for f in os.listdir(dest_path):
        os.remove(os.path.join(dest_path, f))
    
def getList():
    cmd = ['adb', 'shell', 'ls', scnshot_path]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("Standard Output:\n" + result.stdout)
    print("Standard Error:\n" + result.stderr)
    
    return result.stdout.split()

def pullScreenshots(lst):
    if len(lst) > 0:
        for item in lst:
            item = item.rstrip()
            cmd = ['adb', 'pull', scnshot_path + item, dest_path + item]
            subprocess.run(cmd)
        return True
    print("No Screenshot on the device")
    return False
    
if __name__ == "__main__":
    main()