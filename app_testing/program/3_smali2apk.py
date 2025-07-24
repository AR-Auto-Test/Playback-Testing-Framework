"""
This file integrates all the required code in 3_rebuildApk.py and 5_asignKey.py. 
This programe uses a universe key names test.keystore
"""
#import re
import os
import shutil

rebuildApkPath = "../1_rebuildApk"
alignedApkPath = "../2_alignedApk"
apk_signedPath = "../3_signedApk"
decodeFilePath = "../smali"
apkFilePath = "../apks"
keyPath = "../keys"
apkNameList = []

failedFiles = []


def main():
    checkPathExist(rebuildApkPath)
    checkPathExist(alignedApkPath)
    checkPathExist(apk_signedPath)
        
    rebuildApk()
    
    removePath(rebuildApkPath)
    removePath(alignedApkPath)
    
    reportFailures()
    
def checkPathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def rebuildApk():
    for file in os.listdir(decodeFilePath):
        print(file)
        if file == "escape":
            continue

        if os.path.isfile(os.path.join(decodeFilePath, file)):
            continue

        if file in os.listdir(apk_signedPath):
            print("Exists!!!!")

        else:
            checkPathExist(os.path.join(alignedApkPath, file))
            checkPathExist(os.path.join(apk_signedPath, file))
            
            for subfile in os.listdir(os.path.join(decodeFilePath, file)):
                # rebuild
                if ".DS_Store" in subfile:
                    continue
                    
                
                # Switch working mode by changing between True and False
                if False:
                    # recompile all apks
                    try:
                        cmd = "/usr/local/bin/apktool.txt b %s -o %s.apk" % (os.path.join(decodeFilePath, file, subfile), os.path.join(rebuildApkPath, file, subfile))

                        # Use aapt2 in some cases
                        # The project contains a new resource that is not compatible with aapt1.
                        # https://github.com/iBotPeaches/Apktool/issues/1978
                        #cmd = "/usr/local/bin/apktool.txt b --use-aapt2 %s -o %s.apk" % (os.path.join(decodeFilePath, file, subfile), os.path.join(rebuildApkPath, file, subfile))
                        print(cmd)
                        os.system(cmd)
                    except Exception as e:
                        failedFiles.append(subfile)

                    # align
                    aligncmd = "zipalign -p -f -v 4 %s.apk %s.apk" % (os.path.join(rebuildApkPath, file, subfile), os.path.join(alignedApkPath, file, subfile))
                    print(aligncmd)
                    os.system(aligncmd)
                else:
                    # only handle the base.apk, the others are irrelevant
                    if subfile == "base":
                        try:
                            #cmd = "/usr/local/bin/apktool.txt b %s -o %s.apk" % (os.path.join(decodeFilePath, file, subfile), os.path.join(rebuildApkPath, file, subfile))

                            # Use aapt2 in some cases
                            # The project contains a new resource that is not compatible with aapt1.
                            # https://github.com/iBotPeaches/Apktool/issues/1978
                            cmd = "/usr/local/bin/apktool.txt b --use-aapt2 %s -o %s.apk" % (os.path.join(decodeFilePath, file, subfile), os.path.join(rebuildApkPath, file, subfile))
                            print(cmd)
                            os.system(cmd)
                        except Exception as e:
                            failedFiles.append(subfile)

                        # align
                        aligncmd = "zipalign -p -f -v 4 %s.apk %s.apk" % (os.path.join(rebuildApkPath, file, subfile), os.path.join(alignedApkPath, file, subfile))
                        print(aligncmd)
                        os.system(aligncmd)
                        #subprocess.run(cmd.split(' '), shell = True)
                    else:
                        # Copy the original apk to ../2_alignedApk
                        # Because the original apks are surely aligned
                        shutil.copyfile(os.path.join(apkFilePath, file, subfile + ".apk"), os.path.join(alignedApkPath, file, subfile+ ".apk"))


                
                
                # sign
                key = "test.keystore"
                #cmd = ("jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore %s -storepass 123456 -signedjar %s.apk %s %s" % 
                signcmd = ("apksigner sign --ks %s --ks-key-alias %s --ks-pass pass:123456 --min-sdk-version 25 --out %s.apk %s.apk" %
                       (os.path.join(keyPath, key),
                        key,
                        os.path.join(apk_signedPath, file, subfile), 
                        os.path.join(alignedApkPath,file, subfile)))
                print(signcmd)
                os.system(signcmd)
            
        
def removePath(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))
        
def reportFailures():
    fail = False
    print("**********************End Checking********************")
    if len(failedFiles) > 0:
        fail = True
        for file in failedFiles:
            print("%s failed to rebuild" % file)
        
    for file in os.listdir(decodeFilePath):
        if file == "escape":
            continue
            
        if os.path.isfile(os.path.join(decodeFilePath, file)):
            continue
            
        for subfile in os.listdir(os.path.join(decodeFilePath, file)):
                if ".DS_Store" in subfile:
                    continue
                    
                if (subfile + ".apk") not in os.listdir(os.path.join(apk_signedPath, file)):
                    fail = True
                    print("%s/%s failed to rebuild" % (file, subfile))
    
    if not fail:
        print("**********************Good********************")
        print("Ready to install")

        

if __name__ == "__main__":
    main()