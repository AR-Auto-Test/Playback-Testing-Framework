package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Paint;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.security.MessageDigest;

/* compiled from: CenterCrop.java */
/* loaded from: classes.dex */
public class i extends f {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f3962b = "com.bumptech.glide.load.resource.bitmap.CenterCrop".getBytes(c.c.a.m.m.f3537a);

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        messageDigest.update(f3962b);
    }

    @Override // c.c.a.m.x.c.f
    public Bitmap c(c.c.a.m.v.c0.d dVar, Bitmap bitmap, int i, int i2) {
        float width;
        float height;
        Paint paint = a0.f3938a;
        if (bitmap.getWidth() == i && bitmap.getHeight() == i2) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        int width2 = bitmap.getWidth() * i2;
        int height2 = bitmap.getHeight() * i;
        float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        if (width2 > height2) {
            width = i2 / bitmap.getHeight();
            f2 = (i - (bitmap.getWidth() * width)) * 0.5f;
            height = 0.0f;
        } else {
            width = i / bitmap.getWidth();
            height = (i2 - (bitmap.getHeight() * width)) * 0.5f;
        }
        matrix.setScale(width, width);
        matrix.postTranslate((int) (f2 + 0.5f), (int) (height + 0.5f));
        Bitmap e2 = dVar.e(i, i2, a0.d(bitmap));
        e2.setHasAlpha(bitmap.hasAlpha());
        a0.a(bitmap, e2, matrix);
        return e2;
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        return obj instanceof i;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return -599754482;
    }
}