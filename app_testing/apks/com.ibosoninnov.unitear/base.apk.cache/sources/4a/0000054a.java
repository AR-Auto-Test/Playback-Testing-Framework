package b.r.a.a;

import android.view.animation.Interpolator;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: LookupTableInterpolator.java */
/* loaded from: classes.dex */
public abstract class d implements Interpolator {

    /* renamed from: a  reason: collision with root package name */
    public final float[] f2563a;

    /* renamed from: b  reason: collision with root package name */
    public final float f2564b;

    public d(float[] fArr) {
        this.f2563a = fArr;
        this.f2564b = 1.0f / (fArr.length - 1);
    }

    @Override // android.animation.TimeInterpolator
    public float getInterpolation(float f2) {
        if (f2 >= 1.0f) {
            return 1.0f;
        }
        if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        float[] fArr = this.f2563a;
        int min = Math.min((int) ((fArr.length - 1) * f2), fArr.length - 2);
        float f3 = this.f2564b;
        float f4 = (f2 - (min * f3)) / f3;
        float[] fArr2 = this.f2563a;
        return c.b.a.a.a.a(fArr2[min + 1], fArr2[min], f4, fArr2[min]);
    }
}