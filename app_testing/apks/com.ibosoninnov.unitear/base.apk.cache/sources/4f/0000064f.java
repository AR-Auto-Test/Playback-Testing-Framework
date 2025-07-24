package c.a.a.c0;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.PathMeasure;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.io.Closeable;

/* compiled from: Utils.java */
/* loaded from: classes.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    public static final PathMeasure f3031a = new PathMeasure();

    /* renamed from: b  reason: collision with root package name */
    public static final Path f3032b = new Path();

    /* renamed from: c  reason: collision with root package name */
    public static final Path f3033c = new Path();

    /* renamed from: d  reason: collision with root package name */
    public static final float[] f3034d = new float[4];

    /* renamed from: e  reason: collision with root package name */
    public static final float f3035e = (float) (Math.sqrt(2.0d) / 2.0d);

    /* renamed from: f  reason: collision with root package name */
    public static float f3036f = -1.0f;

    public static void a(Path path, float f2, float f3, float f4) {
        PathMeasure pathMeasure = f3031a;
        pathMeasure.setPath(path, false);
        float length = pathMeasure.getLength();
        if (f2 == 1.0f && f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            c.a.a.c.a("applyTrimPathIfNeeded");
        } else if (length >= 1.0f && Math.abs((f3 - f2) - 1.0f) >= 0.01d) {
            float f5 = f2 * length;
            float f6 = f3 * length;
            float f7 = f4 * length;
            float min = Math.min(f5, f6) + f7;
            float max = Math.max(f5, f6) + f7;
            if (min >= length && max >= length) {
                min = f.d(min, length);
                max = f.d(max, length);
            }
            if (min < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                min = f.d(min, length);
            }
            if (max < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                max = f.d(max, length);
            }
            int i = (min > max ? 1 : (min == max ? 0 : -1));
            if (i == 0) {
                path.reset();
                c.a.a.c.a("applyTrimPathIfNeeded");
                return;
            }
            if (i >= 0) {
                min -= length;
            }
            Path path2 = f3032b;
            path2.reset();
            pathMeasure.getSegment(min, max, path2, true);
            if (max > length) {
                Path path3 = f3033c;
                path3.reset();
                pathMeasure.getSegment(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, max % length, path3, true);
                path2.addPath(path3);
            } else if (min < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                Path path4 = f3033c;
                path4.reset();
                pathMeasure.getSegment(min + length, length, path4, true);
                path2.addPath(path4);
            }
            path.set(path2);
            c.a.a.c.a("applyTrimPathIfNeeded");
        } else {
            c.a.a.c.a("applyTrimPathIfNeeded");
        }
    }

    public static void b(Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (RuntimeException e2) {
                throw e2;
            } catch (Exception unused) {
            }
        }
    }

    public static float c() {
        if (f3036f == -1.0f) {
            f3036f = Resources.getSystem().getDisplayMetrics().density;
        }
        return f3036f;
    }

    public static float d(Matrix matrix) {
        float[] fArr = f3034d;
        fArr[0] = 0.0f;
        fArr[1] = 0.0f;
        float f2 = f3035e;
        fArr[2] = f2;
        fArr[3] = f2;
        matrix.mapPoints(fArr);
        return (float) Math.hypot(fArr[2] - fArr[0], fArr[3] - fArr[1]);
    }

    public static Bitmap e(Bitmap bitmap, int i, int i2) {
        if (bitmap.getWidth() == i && bitmap.getHeight() == i2) {
            return bitmap;
        }
        Bitmap createScaledBitmap = Bitmap.createScaledBitmap(bitmap, i, i2, true);
        bitmap.recycle();
        return createScaledBitmap;
    }
}