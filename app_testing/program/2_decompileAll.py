import sys, os
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool, cpu_count

apkPath = '../apks'
smaliPath = '../smali'

def decompile(app):
    try:
        smaliAppPath = os.path.join(smaliPath, app)
        apkAppPath = os.path.join(apkPath, app)
        # if the smali code is already there, ignore
        if not os.path.exists(smaliAppPath):
            os.mkdir(smaliAppPath)
            
            for apk in os.listdir(apkAppPath):
                if apk[-3:] == "apk":
                    #folderName = filename[0:-5]

                    # no -r to decompile resources
                    cmd = "/usr/local/bin/apktool.txt d %s -o %s " % (os.path.join(apkAppPath, apk), os.path.join(smaliAppPath, apk[0:-4]))
                    
                    # with -r, no decompile resources(for apps like wayfair)
                    #cmd = "/usr/local/bin/apktool.txt d %s -r -o %s " % (os.path.join(apkAppPath, apk), os.path.join(smaliAppPath, apk[0:-4]))
                    os.system(cmd)
    except Exception as e:
        print('cant decomplile' + filename)

if __name__ == "__main__":
    
    
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count())
    results = pool.map(decompile, os.listdir(apkPath))
    pool.close()
    pool.join()
            