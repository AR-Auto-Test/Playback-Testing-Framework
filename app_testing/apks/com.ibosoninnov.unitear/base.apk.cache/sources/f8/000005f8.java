package b.z;

import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.PathMeasure;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: PatternPathMotion.java */
/* loaded from: classes.dex */
public class f extends e {

    /* renamed from: a  reason: collision with root package name */
    public final Path f2884a;

    /* renamed from: b  reason: collision with root package name */
    public final Matrix f2885b;

    public f(Path path) {
        Path path2 = new Path();
        this.f2884a = path2;
        Matrix matrix = new Matrix();
        this.f2885b = matrix;
        PathMeasure pathMeasure = new PathMeasure(path, false);
        float[] fArr = new float[2];
        pathMeasure.getPosTan(pathMeasure.getLength(), fArr, null);
        float f2 = fArr[0];
        float f3 = fArr[1];
        pathMeasure.getPosTan(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, fArr, null);
        float f4 = fArr[0];
        float f5 = fArr[1];
        if (f4 == f2 && f5 == f3) {
            throw new IllegalArgumentException("pattern must not end at the starting point");
        }
        matrix.setTranslate(-f4, -f5);
        float f6 = f2 - f4;
        float f7 = f3 - f5;
        float sqrt = 1.0f / ((float) Math.sqrt((f7 * f7) + (f6 * f6)));
        matrix.postScale(sqrt, sqrt);
        matrix.postRotate((float) Math.toDegrees(-Math.atan2(f7, f6)));
        path.transform(matrix, path2);
    }

    @Override // b.z.e
    public Path getPath(float f2, float f3, float f4, float f5) {
        float f6 = f4 - f2;
        float f7 = f5 - f3;
        float sqrt = (float) Math.sqrt((f7 * f7) + (f6 * f6));
        double atan2 = Math.atan2(f7, f6);
        this.f2885b.setScale(sqrt, sqrt);
        this.f2885b.postRotate((float) Math.toDegrees(atan2));
        this.f2885b.postTranslate(f2, f3);
        Path path = new Path();
        this.f2884a.transform(this.f2885b, path);
        return path;
    }
}