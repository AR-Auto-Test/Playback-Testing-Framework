package b.e.c;

import android.graphics.drawable.Drawable;

/* compiled from: RoundRectDrawableWithShadow.java */
/* loaded from: classes.dex */
public class e extends Drawable {

    /* renamed from: a  reason: collision with root package name */
    public static final double f1736a = Math.cos(Math.toRadians(45.0d));

    public static float a(float f2, float f3, boolean z) {
        if (z) {
            return (float) (((1.0d - f1736a) * f3) + f2);
        }
        return f2;
    }

    public static float b(float f2, float f3, boolean z) {
        if (z) {
            return (float) (((1.0d - f1736a) * f3) + (f2 * 1.5f));
        }
        return f2 * 1.5f;
    }
}