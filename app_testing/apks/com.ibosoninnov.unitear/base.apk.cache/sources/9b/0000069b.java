package c.a.a.x.c;

import java.util.List;
import java.util.Objects;

/* compiled from: GradientColorKeyframeAnimation.java */
/* loaded from: classes.dex */
public class d extends f<c.a.a.z.k.c> {
    public final c.a.a.z.k.c i;

    public d(List<c.a.a.d0.a<c.a.a.z.k.c>> list) {
        super(list);
        c.a.a.z.k.c cVar = list.get(0).f3046b;
        int length = cVar != null ? cVar.f3308b.length : 0;
        this.i = new c.a.a.z.k.c(new float[length], new int[length]);
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        c.a.a.z.k.c cVar = this.i;
        c.a.a.z.k.c cVar2 = (c.a.a.z.k.c) aVar.f3046b;
        c.a.a.z.k.c cVar3 = (c.a.a.z.k.c) aVar.f3047c;
        Objects.requireNonNull(cVar);
        if (cVar2.f3308b.length == cVar3.f3308b.length) {
            for (int i = 0; i < cVar2.f3308b.length; i++) {
                cVar.f3307a[i] = c.a.a.c0.f.e(cVar2.f3307a[i], cVar3.f3307a[i], f2);
                cVar.f3308b[i] = b.v.u.c.h(f2, cVar2.f3308b[i], cVar3.f3308b[i]);
            }
            return this.i;
        }
        StringBuilder x = c.b.a.a.a.x("Cannot interpolate between gradients. Lengths vary (");
        x.append(cVar2.f3308b.length);
        x.append(" vs ");
        throw new IllegalArgumentException(c.b.a.a.a.s(x, cVar3.f3308b.length, ")"));
    }
}