package c.a.a.a0;

import android.content.Context;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Objects;

/* compiled from: NetworkCache.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public final Context f2950a;

    public b(Context context) {
        this.f2950a = context.getApplicationContext();
    }

    public static String a(String str, a aVar, boolean z) {
        String str2;
        StringBuilder x = c.b.a.a.a.x("lottie_cache_");
        x.append(str.replaceAll("\\W+", ""));
        if (z) {
            Objects.requireNonNull(aVar);
            str2 = ".temp" + aVar.f2949e;
        } else {
            str2 = aVar.f2949e;
        }
        x.append(str2);
        return x.toString();
    }

    public final File b() {
        File file = new File(this.f2950a.getCacheDir(), "lottie_network_cache");
        if (file.isFile()) {
            file.delete();
        }
        if (!file.exists()) {
            file.mkdirs();
        }
        return file;
    }

    public File c(String str, InputStream inputStream, a aVar) {
        File file = new File(b(), a(str, aVar, true));
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(file);
            byte[] bArr = new byte[1024];
            while (true) {
                int read = inputStream.read(bArr);
                if (read != -1) {
                    fileOutputStream.write(bArr, 0, read);
                } else {
                    fileOutputStream.flush();
                    fileOutputStream.close();
                    return file;
                }
            }
        } finally {
            inputStream.close();
        }
    }
}