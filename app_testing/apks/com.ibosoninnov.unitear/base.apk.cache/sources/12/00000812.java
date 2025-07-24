package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.Paint;
import android.util.Log;
import java.security.MessageDigest;

/* compiled from: CenterInside.java */
/* loaded from: classes.dex */
public class j extends f {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f3963b = "com.bumptech.glide.load.resource.bitmap.CenterInside".getBytes(c.c.a.m.m.f3537a);

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        messageDigest.update(f3963b);
    }

    @Override // c.c.a.m.x.c.f
    public Bitmap c(c.c.a.m.v.c0.d dVar, Bitmap bitmap, int i, int i2) {
        Paint paint = a0.f3938a;
        if (bitmap.getWidth() <= i && bitmap.getHeight() <= i2) {
            if (Log.isLoggable("TransformationUtils", 2)) {
                Log.v("TransformationUtils", "requested target size larger or equal to input, returning input");
                return bitmap;
            }
            return bitmap;
        }
        if (Log.isLoggable("TransformationUtils", 2)) {
            Log.v("TransformationUtils", "requested target size too big for input, fit centering instead");
        }
        return a0.b(dVar, bitmap, i, i2);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        return obj instanceof j;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return -670243078;
    }
}