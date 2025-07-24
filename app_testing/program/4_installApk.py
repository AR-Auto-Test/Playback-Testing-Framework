import sys, os
import subprocess

apk_signedPath = "../3_signedApk"
targetApp = "com.bobsfurniture.retail"
#targetApp = "com.rooom.app"


#targetApp = "com.google.ar.core.examples.java.helloar"

def main():
    print("[INFO] Current App:", targetApp)
    # Remove app if exist
    localTargetApp = targetApp
    if localTargetApp.endswith(".app"):
        localTargetApp = localTargetApp[:-4]
    
    if check_app_installed(targetApp):
        print("[INFO] Uninstalling existing app......")
        uninstallApp(targetApp)
    
    print("[INFO] Installing app......")
    
    if check_apk_amount(localTargetApp) <= 1:
        try:
            cmd = "adb install"
            for file in os.listdir(os.path.join(apk_signedPath, localTargetApp)):
                if file.endswith('.apk'):
                    cmd = cmd + ' ' + os.path.join(apk_signedPath, localTargetApp, file)
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)
    else:
        try:
            cmd = "adb install-multiple"
            for file in os.listdir(os.path.join(apk_signedPath, localTargetApp)):
                if file.endswith('.apk'):
                    cmd = cmd + ' ' + os.path.join(apk_signedPath, localTargetApp, file)
            print(cmd)
            os.system(cmd)
        except Exception as e:
            print(e)
        
def check_apk_amount(localTargetApp):
    amt = 0
    for file in os.listdir(os.path.join(apk_signedPath, localTargetApp)):
        if file.endswith('.apk'):
            amt += 1
    return amt
    
        
def check_app_installed(package_name):
    try:
        output = subprocess.check_output(["adb", "shell", "pm", "list", "packages"]).decode('utf-8')
        if package_name in output:
            return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to check installed apps: {str(e)}")
    return False

def uninstallApp(app):
    cmd = "adb uninstall {apkid}".format(apkid = app)
    os.system(cmd)
    
if __name__ == "__main__":
    main()