package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.BitmapShader;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.graphics.Shader;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.concurrent.locks.Lock;

/* compiled from: RoundedCorners.java */
/* loaded from: classes.dex */
public final class y extends f {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f4014b = "com.bumptech.glide.load.resource.bitmap.RoundedCorners".getBytes(c.c.a.m.m.f3537a);

    /* renamed from: c  reason: collision with root package name */
    public final int f4015c;

    public y(int i) {
        b.v.u.c.d(i > 0, "roundingRadius must be greater than 0.");
        this.f4015c = i;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        messageDigest.update(f4014b);
        messageDigest.update(ByteBuffer.allocate(4).putInt(this.f4015c).array());
    }

    @Override // c.c.a.m.x.c.f
    public Bitmap c(c.c.a.m.v.c0.d dVar, Bitmap bitmap, int i, int i2) {
        Bitmap e2;
        int i3 = this.f4015c;
        Paint paint = a0.f3938a;
        b.v.u.c.d(i3 > 0, "roundingRadius must be greater than 0.");
        Bitmap.Config c2 = a0.c(bitmap);
        Bitmap.Config c3 = a0.c(bitmap);
        if (c3.equals(bitmap.getConfig())) {
            e2 = bitmap;
        } else {
            e2 = dVar.e(bitmap.getWidth(), bitmap.getHeight(), c3);
            new Canvas(e2).drawBitmap(bitmap, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (Paint) null);
        }
        Bitmap e3 = dVar.e(e2.getWidth(), e2.getHeight(), c2);
        e3.setHasAlpha(true);
        Shader.TileMode tileMode = Shader.TileMode.CLAMP;
        BitmapShader bitmapShader = new BitmapShader(e2, tileMode, tileMode);
        Paint paint2 = new Paint();
        paint2.setAntiAlias(true);
        paint2.setShader(bitmapShader);
        RectF rectF = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, e3.getWidth(), e3.getHeight());
        Lock lock = a0.f3941d;
        lock.lock();
        try {
            Canvas canvas = new Canvas(e3);
            canvas.drawColor(0, PorterDuff.Mode.CLEAR);
            float f2 = i3;
            canvas.drawRoundRect(rectF, f2, f2, paint2);
            canvas.setBitmap(null);
            lock.unlock();
            if (!e2.equals(bitmap)) {
                dVar.d(e2);
            }
            return e3;
        } catch (Throwable th) {
            a0.f3941d.unlock();
            throw th;
        }
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        return (obj instanceof y) && this.f4015c == ((y) obj).f4015c;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        int i = this.f4015c;
        char[] cArr = c.c.a.s.j.f4197a;
        return ((i + 527) * 31) - 569625254;
    }
}