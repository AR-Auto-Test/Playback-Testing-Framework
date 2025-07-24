package c.a.a.x.c;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.List;

/* compiled from: ColorKeyframeAnimation.java */
/* loaded from: classes.dex */
public class b extends f<Integer> {
    public b(List<c.a.a.d0.a<Integer>> list) {
        super(list);
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        return Integer.valueOf(j(aVar, f2));
    }

    public int j(c.a.a.d0.a<Integer> aVar, float f2) {
        Integer num;
        Integer num2 = aVar.f3046b;
        if (num2 != null && aVar.f3047c != null) {
            int intValue = num2.intValue();
            int intValue2 = aVar.f3047c.intValue();
            c.a.a.d0.c<A> cVar = this.f3227e;
            if (cVar != 0 && (num = (Integer) cVar.a(aVar.f3049e, aVar.f3050f.floatValue(), Integer.valueOf(intValue), Integer.valueOf(intValue2), f2, d(), this.f3226d)) != null) {
                return num.intValue();
            }
            return b.v.u.c.h(c.a.a.c0.f.b(f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f), intValue, intValue2);
        }
        throw new IllegalStateException("Missing values for keyframe.");
    }
}