package c.a.a.x.c;

import java.util.List;

/* compiled from: FloatKeyframeAnimation.java */
/* loaded from: classes.dex */
public class c extends f<Float> {
    public c(List<c.a.a.d0.a<Float>> list) {
        super(list);
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        return Float.valueOf(k(aVar, f2));
    }

    public float j() {
        return k(a(), c());
    }

    public float k(c.a.a.d0.a<Float> aVar, float f2) {
        Float f3;
        if (aVar.f3046b != null && aVar.f3047c != null) {
            c.a.a.d0.c<A> cVar = this.f3227e;
            if (cVar != 0 && (f3 = (Float) cVar.a(aVar.f3049e, aVar.f3050f.floatValue(), aVar.f3046b, aVar.f3047c, f2, d(), this.f3226d)) != null) {
                return f3.floatValue();
            }
            if (aVar.f3051g == -3987645.8f) {
                aVar.f3051g = aVar.f3046b.floatValue();
            }
            float f4 = aVar.f3051g;
            if (aVar.f3052h == -3987645.8f) {
                aVar.f3052h = aVar.f3047c.floatValue();
            }
            return c.a.a.c0.f.e(f4, aVar.f3052h, f2);
        }
        throw new IllegalStateException("Missing values for keyframe.");
    }
}