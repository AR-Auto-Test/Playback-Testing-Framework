package c.e.b.p000if;

import android.content.Context;
import android.util.Log;
import java.io.File;
import java.util.Date;

/* compiled from: CacheHelper.java */
/* renamed from: c.e.b.if.f  reason: invalid package */
/* loaded from: classes2.dex */
public class f {
    public static void a(Context context, int i) {
        Log.i("CacheHelper", String.format("Starting cache prune, deleting files older than %d seconds", Integer.valueOf(i)));
        Log.i("CacheHelper", String.format("Cache pruning completed, %d files deleted", Integer.valueOf(b(context.getDataDir(), i))));
    }

    public static int b(File file, int i) {
        int i2;
        File[] listFiles;
        if (file == null || !file.isDirectory()) {
            return 0;
        }
        try {
            i2 = 0;
            for (File file2 : file.listFiles()) {
                try {
                    if (file2.isDirectory()) {
                        i2 += b(file2, i);
                    }
                    if (file2.lastModified() < new Date().getTime() - (i * 1000) && file2.getAbsolutePath().toLowerCase().contains("app_webview")) {
                        Log.d("CacheHelper", file2.getAbsolutePath() + "  Last used " + ((System.currentTimeMillis() - file2.lastModified()) / 1000) + " size=" + Integer.parseInt(String.valueOf(file2.length() / 1024)));
                        if (file2.delete()) {
                            i2++;
                        }
                    }
                } catch (Exception e2) {
                    e = e2;
                    Log.e("CacheHelper", String.format("Failed to clean the cache, error %s", e.getMessage()));
                    return i2;
                }
            }
        } catch (Exception e3) {
            e = e3;
            i2 = 0;
        }
        return i2;
    }
}