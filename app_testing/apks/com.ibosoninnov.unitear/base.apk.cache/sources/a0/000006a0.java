package c.a.a.x.c;

import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.PointF;
import java.util.List;

/* compiled from: PathKeyframeAnimation.java */
/* loaded from: classes.dex */
public class i extends f<PointF> {
    public final PointF i;
    public final float[] j;
    public h k;
    public PathMeasure l;

    public i(List<? extends c.a.a.d0.a<PointF>> list) {
        super(list);
        this.i = new PointF();
        this.j = new float[2];
        this.l = new PathMeasure();
    }

    /* JADX DEBUG: Multi-variable search result rejected for r5v0, resolved type: T */
    /* JADX DEBUG: Multi-variable search result rejected for r6v0, resolved type: T */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.x.c.a
    public Object f(c.a.a.d0.a aVar, float f2) {
        PointF pointF;
        h hVar = (h) aVar;
        Path path = hVar.o;
        if (path == null) {
            return (PointF) aVar.f3046b;
        }
        c.a.a.d0.c<A> cVar = this.f3227e;
        if (cVar == 0 || (pointF = (PointF) cVar.a(hVar.f3049e, hVar.f3050f.floatValue(), hVar.f3046b, hVar.f3047c, d(), f2, this.f3226d)) == null) {
            if (this.k != hVar) {
                this.l.setPath(path, false);
                this.k = hVar;
            }
            PathMeasure pathMeasure = this.l;
            pathMeasure.getPosTan(pathMeasure.getLength() * f2, this.j, null);
            PointF pointF2 = this.i;
            float[] fArr = this.j;
            pointF2.set(fArr[0], fArr[1]);
            return this.i;
        }
        return pointF;
    }
}