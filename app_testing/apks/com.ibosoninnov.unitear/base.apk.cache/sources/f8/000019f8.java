package com.google.android.play.core.assetpacks;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzbh {
    private static final com.google.android.play.core.internal.zzag zza = new com.google.android.play.core.internal.zzag("AssetPackStorage");
    private static final long zzb;
    private static final long zzc;
    private final Context zzd;
    private final zzed zze;

    static {
        TimeUnit timeUnit = TimeUnit.DAYS;
        zzb = timeUnit.toMillis(14L);
        zzc = timeUnit.toMillis(28L);
    }

    public zzbh(Context context, zzed zzedVar) {
        this.zzd = context;
        this.zze = zzedVar;
    }

    private static long zzH(File file, boolean z) {
        File[] listFiles;
        if (file.exists()) {
            ArrayList arrayList = new ArrayList();
            if (z && file.listFiles().length > 1) {
                zza.zze("Multiple pack versions found, using highest version code.", new Object[0]);
            }
            try {
                for (File file2 : file.listFiles()) {
                    if (!file2.getName().equals("stale.tmp")) {
                        arrayList.add(Long.valueOf(file2.getName()));
                    }
                }
            } catch (NumberFormatException e2) {
                zza.zzc(e2, "Corrupt asset pack directories.", new Object[0]);
            }
            if (arrayList.isEmpty()) {
                return -1L;
            }
            Collections.sort(arrayList);
            return ((Long) arrayList.get(arrayList.size() - 1)).longValue();
        }
        return -1L;
    }

    private final File zzI(String str) {
        return new File(zzL(), str);
    }

    private final File zzJ(String str, int i, long j) {
        return new File(zzj(str, i, j), "merge.tmp");
    }

    private final File zzK(String str, int i, long j) {
        return new File(new File(new File(zzM(), str), String.valueOf(i)), String.valueOf(j));
    }

    private final File zzL() {
        return new File(this.zzd.getFilesDir(), "assetpacks");
    }

    private final File zzM() {
        return new File(zzL(), "_tmp");
    }

    private static List zzN(PackageInfo packageInfo, String str) {
        ArrayList arrayList = new ArrayList();
        String[] strArr = packageInfo.splitNames;
        if (strArr == null) {
            return arrayList;
        }
        int i = (-Arrays.binarySearch(strArr, str)) - 1;
        while (true) {
            String[] strArr2 = packageInfo.splitNames;
            if (i >= strArr2.length || !strArr2[i].startsWith(str)) {
                break;
            }
            arrayList.add(packageInfo.applicationInfo.splitSourceDirs[i]);
            i++;
        }
        return arrayList;
    }

    private final List zzO() {
        File[] listFiles;
        ArrayList arrayList = new ArrayList();
        try {
        } catch (IOException e2) {
            zza.zzb("Could not process directory while scanning installed packs. %s", e2);
        }
        if (zzL().exists() && zzL().listFiles() != null) {
            for (File file : zzL().listFiles()) {
                if (!file.getCanonicalPath().equals(zzM().getCanonicalPath())) {
                    arrayList.add(file);
                }
            }
            return arrayList;
        }
        return arrayList;
    }

    private static void zzP(File file) {
        File[] listFiles;
        if (file.listFiles() == null || file.listFiles().length <= 1) {
            return;
        }
        long zzH = zzH(file, false);
        for (File file2 : file.listFiles()) {
            if (!file2.getName().equals(String.valueOf(zzH)) && !file2.getName().equals("stale.tmp")) {
                zzQ(file2);
            }
        }
    }

    private static boolean zzQ(File file) {
        File[] listFiles = file.listFiles();
        boolean z = true;
        if (listFiles != null) {
            for (File file2 : listFiles) {
                z &= zzQ(file2);
            }
        }
        if (file.delete()) {
            return z;
        }
        return false;
    }

    public final void zzA(String str, int i, long j, int i2) {
        File zzJ = zzJ(str, i, j);
        Properties properties = new Properties();
        properties.put("numberOfMerges", String.valueOf(i2));
        zzJ.getParentFile().mkdirs();
        zzJ.createNewFile();
        FileOutputStream fileOutputStream = new FileOutputStream(zzJ);
        properties.store(fileOutputStream, (String) null);
        fileOutputStream.close();
    }

    public final void zzB(String str, int i, long j) {
        File[] listFiles;
        File[] listFiles2;
        File zzI = zzI(str);
        if (zzI.exists()) {
            for (File file : zzI.listFiles()) {
                if (!file.getName().equals(String.valueOf(i)) && !file.getName().equals("stale.tmp")) {
                    zzQ(file);
                } else if (file.getName().equals(String.valueOf(i))) {
                    for (File file2 : file.listFiles()) {
                        if (!file2.getName().equals(String.valueOf(j))) {
                            zzQ(file2);
                        }
                    }
                }
            }
        }
    }

    public final void zzC(List list) {
        int zza2 = this.zze.zza();
        for (File file : zzO()) {
            if (!list.contains(file.getName()) && zzH(file, true) != zza2) {
                zzQ(file);
            }
        }
    }

    public final boolean zzD(String str) {
        if (zzI(str).exists()) {
            return zzQ(zzI(str));
        }
        return true;
    }

    public final boolean zzE(String str, int i, long j) {
        if (zzK(str, i, j).exists()) {
            return zzQ(zzK(str, i, j));
        }
        return true;
    }

    public final boolean zzF(String str, int i, long j) {
        if (zzh(str, i, j).exists()) {
            return zzQ(zzh(str, i, j));
        }
        return true;
    }

    public final boolean zzG(String str) {
        return zzr(str) != null;
    }

    public final int zza(String str) {
        return (int) zzH(zzI(str), true);
    }

    public final int zzb(String str, int i, long j) {
        File zzJ = zzJ(str, i, j);
        if (zzJ.exists()) {
            Properties properties = new Properties();
            FileInputStream fileInputStream = new FileInputStream(zzJ);
            try {
                properties.load(fileInputStream);
                fileInputStream.close();
                if (properties.getProperty("numberOfMerges") != null) {
                    try {
                        return Integer.parseInt(properties.getProperty("numberOfMerges"));
                    } catch (NumberFormatException e2) {
                        throw new zzck("Merge checkpoint file corrupt.", e2);
                    }
                }
                throw new zzck("Merge checkpoint file corrupt.");
            } catch (Throwable th) {
                try {
                    fileInputStream.close();
                } catch (Throwable unused) {
                }
                throw th;
            }
        }
        return 0;
    }

    public final long zzc(String str) {
        return zzH(zzg(str, (int) zzH(zzI(str), true)), true);
    }

    public final AssetLocation zzd(String str, String str2, List list) {
        if (list == null) {
            return null;
        }
        String path = new File("assets", str2).getPath();
        Iterator it = list.iterator();
        while (it.hasNext()) {
            String str3 = (String) it.next();
            try {
                AssetLocation zza2 = zzbt.zza(str3, path);
                if (zza2 != null) {
                    return zza2;
                }
            } catch (IOException e2) {
                zza.zzc(e2, "Failed to parse APK file '%s' looking for asset '%s'.", str3, str2);
                return null;
            }
        }
        zza.zza("The asset %s is not present in Asset Pack %s. Searched in APKs: %s", str2, str, list);
        return null;
    }

    public final AssetLocation zze(String str, String str2, AssetPackLocation assetPackLocation) {
        File file = new File(assetPackLocation.assetsPath(), str2);
        if (file.exists()) {
            return new zzbl(file.getPath(), 0L, file.length());
        }
        zza.zza("The asset %s is not present in Asset Pack %s. Searched in folder: %s", str2, str, assetPackLocation.assetsPath());
        return null;
    }

    public final AssetPackLocation zzf(String str) {
        String zzr = zzr(str);
        if (zzr == null) {
            return null;
        }
        File file = new File(zzr, "assets");
        if (!file.isDirectory()) {
            zza.zzb("Failed to find assets directory: %s", file);
            return null;
        }
        return new zzbm(0, zzr, file.getCanonicalPath());
    }

    public final File zzg(String str, int i) {
        return new File(zzI(str), String.valueOf(i));
    }

    public final File zzh(String str, int i, long j) {
        return new File(zzg(str, i), String.valueOf(j));
    }

    public final File zzi(String str, int i, long j) {
        return new File(zzh(str, i, j), "_metadata");
    }

    public final File zzj(String str, int i, long j) {
        return new File(zzK(str, i, j), "_packs");
    }

    public final File zzk(String str, int i, long j) {
        return new File(zzi(str, i, j), "properties.dat");
    }

    public final File zzl(String str, int i, long j) {
        return new File(new File(zzK(str, i, j), "_slices"), "_metadata");
    }

    public final File zzm(String str, int i, long j, String str2) {
        return new File(zzo(str, i, j, str2), "checkpoint_ext.dat");
    }

    public final File zzn(String str, int i, long j, String str2) {
        return new File(zzo(str, i, j, str2), "checkpoint.dat");
    }

    public final File zzo(String str, int i, long j, String str2) {
        return new File(zzl(str, i, j), str2);
    }

    public final File zzp(String str, int i, long j, String str2) {
        return new File(new File(new File(zzK(str, i, j), "_slices"), "_unverified"), str2);
    }

    public final File zzq(String str, int i, long j, String str2) {
        return new File(new File(new File(zzK(str, i, j), "_slices"), "_verified"), str2);
    }

    public final String zzr(String str) {
        int length;
        File file = new File(zzL(), str);
        if (!file.exists()) {
            zza.zza("Pack not found with pack name: %s", str);
            return null;
        }
        File file2 = new File(file, String.valueOf(this.zze.zza()));
        if (!file2.exists()) {
            zza.zza("Pack not found with pack name: %s app version: %s", str, Integer.valueOf(this.zze.zza()));
            return null;
        }
        File[] listFiles = file2.listFiles();
        if (listFiles == null || (length = listFiles.length) == 0) {
            zza.zza("No pack version found for pack name: %s app version: %s", str, Integer.valueOf(this.zze.zza()));
            return null;
        } else if (length > 1) {
            zza.zzb("Multiple pack versions found for pack name: %s app version: %s", str, Integer.valueOf(this.zze.zza()));
            return null;
        } else {
            return listFiles[0].getCanonicalPath();
        }
    }

    public final List zzs(String str) {
        PackageInfo packageInfo;
        String str2 = null;
        try {
            packageInfo = this.zzd.getPackageManager().getPackageInfo(this.zzd.getPackageName(), 0);
        } catch (PackageManager.NameNotFoundException unused) {
            zza.zzb("Could not find PackageInfo.", new Object[0]);
            packageInfo = null;
        }
        if (packageInfo == null) {
            return null;
        }
        ArrayList arrayList = new ArrayList();
        String[] strArr = packageInfo.splitNames;
        if (strArr != null && packageInfo.applicationInfo.splitSourceDirs != null) {
            int binarySearch = Arrays.binarySearch(strArr, str);
            if (binarySearch < 0) {
                zza.zza("Asset Pack '%s' is not installed.", str);
            } else {
                str2 = packageInfo.applicationInfo.splitSourceDirs[binarySearch];
            }
        } else {
            zza.zza("No splits present for package %s.", str);
        }
        if (str2 == null) {
            arrayList.add(packageInfo.applicationInfo.sourceDir);
            arrayList.addAll(zzN(packageInfo, "config."));
            return arrayList;
        }
        arrayList.add(str2);
        arrayList.addAll(zzN(packageInfo, String.valueOf(str).concat(".config.")));
        return arrayList;
    }

    public final Map zzt() {
        HashMap hashMap = new HashMap();
        for (File file : zzO()) {
            String name = file.getName();
            int zzH = (int) zzH(zzI(name), true);
            long zzH2 = zzH(zzg(name, zzH), true);
            if (zzh(name, zzH, zzH2).exists()) {
                hashMap.put(name, Long.valueOf(zzH2));
            }
        }
        return hashMap;
    }

    public final Map zzu() {
        HashMap hashMap = new HashMap();
        for (String str : zzv().keySet()) {
            hashMap.put(str, Long.valueOf(zzc(str)));
        }
        return hashMap;
    }

    public final Map zzv() {
        HashMap hashMap = new HashMap();
        try {
            for (File file : zzO()) {
                AssetPackLocation zzf = zzf(file.getName());
                if (zzf != null) {
                    hashMap.put(file.getName(), zzf);
                }
            }
        } catch (IOException e2) {
            zza.zzb("Could not process directory while scanning installed packs: %s", e2);
        }
        return hashMap;
    }

    public final void zzw() {
        for (File file : zzO()) {
            if (file.listFiles() != null) {
                zzP(file);
                long zzH = zzH(file, false);
                if (this.zze.zza() != zzH) {
                    try {
                        new File(new File(file, String.valueOf(zzH)), "stale.tmp").createNewFile();
                    } catch (IOException unused) {
                        zza.zzb("Could not write staleness marker.", new Object[0]);
                    }
                }
                for (File file2 : file.listFiles()) {
                    zzP(file2);
                }
            }
        }
    }

    public final void zzx() {
        File[] listFiles;
        if (zzM().exists()) {
            for (File file : zzM().listFiles()) {
                if (System.currentTimeMillis() - file.lastModified() > zzb) {
                    zzQ(file);
                } else {
                    zzP(file);
                }
            }
        }
    }

    public final void zzy() {
        File[] listFiles;
        for (File file : zzO()) {
            if (file.listFiles() != null) {
                for (File file2 : file.listFiles()) {
                    File file3 = new File(file2, "stale.tmp");
                    if (file3.exists() && System.currentTimeMillis() - file3.lastModified() > zzc) {
                        zzQ(file2);
                    }
                }
            }
        }
    }

    public final void zzz() {
        zzQ(zzL());
    }
}