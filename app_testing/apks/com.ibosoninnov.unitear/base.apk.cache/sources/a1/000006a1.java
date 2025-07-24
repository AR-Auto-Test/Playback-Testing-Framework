package c.a.a.x.c;

import android.graphics.PointF;
import java.util.List;

/* compiled from: PointKeyframeAnimation.java */
/* loaded from: classes.dex */
public class j extends f<PointF> {
    public final PointF i;

    public j(List<c.a.a.d0.a<PointF>> list) {
        super(list);
        this.i = new PointF();
    }

    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        T t;
        PointF pointF;
        T t2 = aVar.f3046b;
        if (t2 != 0 && (t = aVar.f3047c) != 0) {
            PointF pointF2 = (PointF) t2;
            PointF pointF3 = (PointF) t;
            c.a.a.d0.c<A> cVar = this.f3227e;
            if (cVar == 0 || (pointF = (PointF) cVar.a(aVar.f3049e, aVar.f3050f.floatValue(), pointF2, pointF3, f2, d(), this.f3226d)) == null) {
                PointF pointF4 = this.i;
                float f3 = pointF2.x;
                float a2 = c.b.a.a.a.a(pointF3.x, f3, f2, f3);
                float f4 = pointF2.y;
                pointF4.set(a2, ((pointF3.y - f4) * f2) + f4);
                return this.i;
            }
            return pointF;
        }
        throw new IllegalStateException("Missing values for keyframe.");
    }
}