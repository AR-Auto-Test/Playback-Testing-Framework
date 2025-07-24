package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/* compiled from: BitmapEncoder.java */
/* loaded from: classes.dex */
public class c implements c.c.a.m.s<Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public static final c.c.a.m.o<Integer> f3945a = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.BitmapEncoder.CompressionQuality", 90);

    /* renamed from: b  reason: collision with root package name */
    public static final c.c.a.m.o<Bitmap.CompressFormat> f3946b = new c.c.a.m.o<>("com.bumptech.glide.load.resource.bitmap.BitmapEncoder.CompressionFormat", null, c.c.a.m.o.f3539a);

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f3947c;

    public c(c.c.a.m.v.c0.b bVar) {
        this.f3947c = bVar;
    }

    /* JADX WARN: Code restructure failed: missing block: B:31:0x0069, code lost:
        if (r6 != null) goto L20;
     */
    @Override // c.c.a.m.d
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean a(Object obj, File file, c.c.a.m.p pVar) {
        FileOutputStream fileOutputStream;
        Bitmap bitmap = (Bitmap) ((c.c.a.m.v.w) obj).get();
        Bitmap.CompressFormat compressFormat = (Bitmap.CompressFormat) pVar.c(f3946b);
        if (compressFormat == null) {
            if (bitmap.hasAlpha()) {
                compressFormat = Bitmap.CompressFormat.PNG;
            } else {
                compressFormat = Bitmap.CompressFormat.JPEG;
            }
        }
        bitmap.getWidth();
        bitmap.getHeight();
        int i = c.c.a.s.f.f4187b;
        long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        int intValue = ((Integer) pVar.c(f3945a)).intValue();
        boolean z = false;
        c.c.a.m.u.c cVar = null;
        try {
            try {
                fileOutputStream = new FileOutputStream(file);
                try {
                    cVar = this.f3947c != null ? new c.c.a.m.u.c(fileOutputStream, this.f3947c) : fileOutputStream;
                    bitmap.compress(compressFormat, intValue, cVar);
                    cVar.close();
                    z = true;
                } catch (IOException e2) {
                    e = e2;
                    cVar = fileOutputStream;
                    if (Log.isLoggable("BitmapEncoder", 3)) {
                        Log.d("BitmapEncoder", "Failed to encode Bitmap", e);
                    }
                } catch (Throwable th) {
                    th = th;
                    if (fileOutputStream != null) {
                        try {
                            fileOutputStream.close();
                        } catch (IOException unused) {
                        }
                    }
                    throw th;
                }
            } catch (IOException e3) {
                e = e3;
            }
            try {
                cVar.close();
            } catch (IOException unused2) {
            }
            if (Log.isLoggable("BitmapEncoder", 2)) {
                Log.v("BitmapEncoder", "Compressed with type: " + compressFormat + " of size " + c.c.a.s.j.d(bitmap) + " in " + c.c.a.s.f.a(elapsedRealtimeNanos) + ", options format: " + pVar.c(f3946b) + ", hasAlpha: " + bitmap.hasAlpha());
            }
            return z;
        } catch (Throwable th2) {
            th = th2;
            fileOutputStream = null;
        }
    }

    @Override // c.c.a.m.s
    public c.c.a.m.c b(c.c.a.m.p pVar) {
        return c.c.a.m.c.TRANSFORMED;
    }
}