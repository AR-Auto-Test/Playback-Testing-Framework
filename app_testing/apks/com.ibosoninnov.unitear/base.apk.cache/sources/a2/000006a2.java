package c.a.a.x.c;

import java.util.List;

/* compiled from: ScaleKeyframeAnimation.java */
/* loaded from: classes.dex */
public class k extends f<c.a.a.d0.d> {
    public final c.a.a.d0.d i;

    public k(List<c.a.a.d0.a<c.a.a.d0.d>> list) {
        super(list);
        this.i = new c.a.a.d0.d();
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        T t;
        c.a.a.d0.d dVar;
        T t2 = aVar.f3046b;
        if (t2 != 0 && (t = aVar.f3047c) != 0) {
            c.a.a.d0.d dVar2 = (c.a.a.d0.d) t2;
            c.a.a.d0.d dVar3 = (c.a.a.d0.d) t;
            c.a.a.d0.c<A> cVar = this.f3227e;
            if (cVar == 0 || (dVar = (c.a.a.d0.d) cVar.a(aVar.f3049e, aVar.f3050f.floatValue(), dVar2, dVar3, f2, d(), this.f3226d)) == null) {
                c.a.a.d0.d dVar4 = this.i;
                float e2 = c.a.a.c0.f.e(dVar2.f3057a, dVar3.f3057a, f2);
                float e3 = c.a.a.c0.f.e(dVar2.f3058b, dVar3.f3058b, f2);
                dVar4.f3057a = e2;
                dVar4.f3058b = e3;
                return this.i;
            }
            return dVar;
        }
        throw new IllegalStateException("Missing values for keyframe.");
    }
}