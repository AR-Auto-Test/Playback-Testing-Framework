package com.google.android.gms.internal.measurement;

import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.StrictMode;
import android.util.Log;
import b.f.h;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzho {
    private static volatile zzig zza;

    private zzho() {
    }

    /* JADX WARN: Can't wrap try/catch for region: R(16:6|(3:10|11|12)|18|(1:22)|23|24|25|26|27|28|(1:30)(1:84)|31|(10:33|34|35|36|37|38|(2:39|(3:41|(3:59|60|61)(8:43|44|(2:46|(1:49))|50|(1:52)(1:58)|(1:54)|55|56)|57)(1:62))|63|64|65)(1:83)|66|11|12) */
    /* JADX WARN: Code restructure failed: missing block: B:30:0x0068, code lost:
        r3 = move-exception;
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x0069, code lost:
        android.util.Log.e("HermeticFileOverrides", "no data dir", r3);
        r3 = com.google.android.gms.internal.measurement.zzig.zzc();
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static zzig zza(Context context) {
        zzig zzigVar;
        zzig zzc;
        synchronized (zzho.class) {
            zzigVar = zza;
            if (zzigVar == null) {
                String str = Build.TYPE;
                String str2 = Build.TAGS;
                if ((!str.equals("eng") && !str.equals("userdebug")) || (!str2.contains("dev-keys") && !str2.contains("test-keys"))) {
                    zzc = zzig.zzc();
                    zzigVar = zzc;
                    zza = zzigVar;
                }
                if (zzhb.zzb() && !context.isDeviceProtectedStorage()) {
                    context = context.createDeviceProtectedStorageContext();
                }
                StrictMode.ThreadPolicy allowThreadDiskReads = StrictMode.allowThreadDiskReads();
                StrictMode.allowThreadDiskWrites();
                File file = new File(context.getDir("phenotype_hermetic", 0), "overrides.txt");
                zzig zzc2 = file.exists() ? zzig.zzd(file) : zzig.zzc();
                if (zzc2.zzb()) {
                    File file2 = (File) zzc2.zza();
                    try {
                        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(file2)));
                        try {
                            h hVar = new h();
                            HashMap hashMap = new HashMap();
                            while (true) {
                                String readLine = bufferedReader.readLine();
                                if (readLine == null) {
                                    break;
                                }
                                String[] split = readLine.split(" ", 3);
                                if (split.length != 3) {
                                    Log.e("HermeticFileOverrides", "Invalid: " + readLine);
                                } else {
                                    String str3 = new String(split[0]);
                                    String decode = Uri.decode(new String(split[1]));
                                    String str4 = (String) hashMap.get(split[2]);
                                    if (str4 == null) {
                                        String str5 = new String(split[2]);
                                        str4 = Uri.decode(str5);
                                        if (str4.length() < 1024 || str4 == str5) {
                                            hashMap.put(str5, str4);
                                        }
                                    }
                                    if (!(hVar.e(str3) >= 0)) {
                                        hVar.put(str3, new h());
                                    }
                                    ((h) hVar.getOrDefault(str3, null)).put(decode, str4);
                                }
                            }
                            String obj = file2.toString();
                            String packageName = context.getPackageName();
                            Log.w("HermeticFileOverrides", "Parsed " + obj + " for Android package " + packageName);
                            zzhh zzhhVar = new zzhh(hVar);
                            bufferedReader.close();
                            zzc = zzig.zzd(zzhhVar);
                        } catch (Throwable th) {
                            try {
                                bufferedReader.close();
                            } catch (Throwable th2) {
                                try {
                                    Throwable.class.getDeclaredMethod("addSuppressed", Throwable.class).invoke(th, th2);
                                } catch (Exception unused) {
                                }
                            }
                            throw th;
                        }
                    } catch (IOException e2) {
                        throw new RuntimeException(e2);
                    }
                } else {
                    zzc = zzig.zzc();
                }
                StrictMode.setThreadPolicy(allowThreadDiskReads);
                zzigVar = zzc;
                zza = zzigVar;
            }
        }
        return zzigVar;
    }
}