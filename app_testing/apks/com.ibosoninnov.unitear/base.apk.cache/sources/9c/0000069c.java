package c.a.a.x.c;

import android.graphics.PointF;
import java.util.List;

/* compiled from: IntegerKeyframeAnimation.java */
/* loaded from: classes.dex */
public class e extends f<Integer> {
    public e(List<c.a.a.d0.a<Integer>> list) {
        super(list);
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        return Integer.valueOf(j(aVar, f2));
    }

    public int j(c.a.a.d0.a<Integer> aVar, float f2) {
        Integer num;
        if (aVar.f3046b != null && aVar.f3047c != null) {
            c.a.a.d0.c<A> cVar = this.f3227e;
            if (cVar != 0 && (num = (Integer) cVar.a(aVar.f3049e, aVar.f3050f.floatValue(), aVar.f3046b, aVar.f3047c, f2, d(), this.f3226d)) != null) {
                return num.intValue();
            }
            if (aVar.i == 784923401) {
                aVar.i = aVar.f3046b.intValue();
            }
            int i = aVar.i;
            if (aVar.j == 784923401) {
                aVar.j = aVar.f3047c.intValue();
            }
            int i2 = aVar.j;
            PointF pointF = c.a.a.c0.f.f3030a;
            return (int) ((f2 * (i2 - i)) + i);
        }
        throw new IllegalStateException("Missing values for keyframe.");
    }
}