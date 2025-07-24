import sys, os
import subprocess

# Assume thess directories exists on the Android device, no validation in this program
app2pull = "com.ashleyfurniturehomestore.ecomm"
#app2pull = "com.ar.augment"
apks_path = "../apks/"


def main():
    # If apks (or the directory) exist, return
    localApp2Pull = app2pull
    if localApp2Pull.endswith(".app"):
        localApp2Pull = localApp2Pull[:-4]
    
    targetPath = os.path.join(apks_path, localApp2Pull)
    if os.path.exists(targetPath):
        print("[Info] Apks already exist.")
        return
    
    os.makedirs(targetPath)
    
    try:
        # adb shell pm path com.example.someapp
        cmd = ['adb', 'shell', 'pm', 'path', app2pull]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        output_list = result.stdout.split()
        #print(output_list)
        
        new_output_list = [s[8:] if s.startswith('package:') else s for s in output_list]
        
        for package in new_output_list:
            name = package.split('/')[-1]
            cmd = ['adb', 'pull', package, os.path.join(targetPath, name)]
            #print(cmd)
            
            subprocess.run(cmd)
        
    except Exception as e:
        print(e)
        
    print("[Info] The apks pulling completed")
    
if __name__ == "__main__":
    main()