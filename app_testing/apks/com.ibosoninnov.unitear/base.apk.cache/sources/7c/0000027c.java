package b.d.a.e;

import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: ZoomStateImpl.java */
/* loaded from: classes.dex */
public class x1 implements b.d.b.c1 {

    /* renamed from: a  reason: collision with root package name */
    public float f1235a;

    /* renamed from: b  reason: collision with root package name */
    public final float f1236b;

    /* renamed from: c  reason: collision with root package name */
    public final float f1237c;

    /* renamed from: d  reason: collision with root package name */
    public float f1238d;

    public x1(float f2, float f3) {
        this.f1236b = f2;
        this.f1237c = f3;
    }

    public void a(float f2) {
        float f3 = this.f1236b;
        if (f2 <= f3) {
            float f4 = this.f1237c;
            if (f2 >= f4) {
                this.f1235a = f2;
                int i = (f3 > f4 ? 1 : (f3 == f4 ? 0 : -1));
                float f5 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                if (i != 0) {
                    if (f2 == f3) {
                        f5 = 1.0f;
                    } else if (f2 != f4) {
                        float f6 = 1.0f / f4;
                        f5 = ((1.0f / f2) - f6) / ((1.0f / f3) - f6);
                    }
                }
                this.f1238d = f5;
                return;
            }
        }
        throw new IllegalArgumentException("Requested zoomRatio " + f2 + " is not within valid range [" + this.f1237c + " , " + this.f1236b + "]");
    }
}