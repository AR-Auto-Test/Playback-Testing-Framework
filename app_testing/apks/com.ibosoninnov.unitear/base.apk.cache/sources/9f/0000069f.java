package c.a.a.x.c;

import android.graphics.Path;
import android.graphics.PathMeasure;
import android.graphics.PointF;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: PathKeyframe.java */
/* loaded from: classes.dex */
public class h extends c.a.a.d0.a<PointF> {
    public Path o;
    public final c.a.a.d0.a<PointF> p;

    public h(c.a.a.d dVar, c.a.a.d0.a<PointF> aVar) {
        super(dVar, aVar.f3046b, aVar.f3047c, aVar.f3048d, aVar.f3049e, aVar.f3050f);
        this.p = aVar;
        e();
    }

    public void e() {
        T t;
        T t2 = this.f3047c;
        boolean z = (t2 == 0 || (t = this.f3046b) == 0 || !((PointF) t).equals(((PointF) t2).x, ((PointF) t2).y)) ? false : true;
        T t3 = this.f3047c;
        if (t3 == 0 || z) {
            return;
        }
        PointF pointF = (PointF) this.f3046b;
        PointF pointF2 = (PointF) t3;
        c.a.a.d0.a<PointF> aVar = this.p;
        PointF pointF3 = aVar.m;
        PointF pointF4 = aVar.n;
        PathMeasure pathMeasure = c.a.a.c0.g.f3031a;
        Path path = new Path();
        path.moveTo(pointF.x, pointF.y);
        if (pointF3 != null && pointF4 != null && (pointF3.length() != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || pointF4.length() != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            float f2 = pointF.x;
            float f3 = pointF2.x;
            float f4 = pointF2.y;
            path.cubicTo(pointF3.x + f2, pointF.y + pointF3.y, f3 + pointF4.x, f4 + pointF4.y, f3, f4);
        } else {
            path.lineTo(pointF2.x, pointF2.y);
        }
        this.o = path;
    }
}