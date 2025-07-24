package c.a.a.x.c;

import android.graphics.Path;
import android.graphics.PointF;
import java.util.List;

/* compiled from: ShapeKeyframeAnimation.java */
/* loaded from: classes.dex */
public class l extends a<c.a.a.z.k.k, Path> {
    public final c.a.a.z.k.k i;
    public final Path j;

    public l(List<c.a.a.d0.a<c.a.a.z.k.k>> list) {
        super(list);
        this.i = new c.a.a.z.k.k();
        this.j = new Path();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [c.a.a.d0.a, float] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.x.c.a
    public Path f(c.a.a.d0.a<c.a.a.z.k.k> aVar, float f2) {
        c.a.a.z.k.k kVar = aVar.f3046b;
        c.a.a.z.k.k kVar2 = aVar.f3047c;
        c.a.a.z.k.k kVar3 = this.i;
        if (kVar3.f3357b == null) {
            kVar3.f3357b = new PointF();
        }
        kVar3.f3358c = kVar.f3358c || kVar2.f3358c;
        if (kVar.f3356a.size() != kVar2.f3356a.size()) {
            StringBuilder x = c.b.a.a.a.x("Curves must have the same number of control points. Shape 1: ");
            x.append(kVar.f3356a.size());
            x.append("\tShape 2: ");
            x.append(kVar2.f3356a.size());
            c.a.a.c0.c.b(x.toString());
        }
        int min = Math.min(kVar.f3356a.size(), kVar2.f3356a.size());
        if (kVar3.f3356a.size() < min) {
            for (int size = kVar3.f3356a.size(); size < min; size++) {
                kVar3.f3356a.add(new c.a.a.z.a());
            }
        } else if (kVar3.f3356a.size() > min) {
            for (int size2 = kVar3.f3356a.size() - 1; size2 >= min; size2--) {
                List<c.a.a.z.a> list = kVar3.f3356a;
                list.remove(list.size() - 1);
            }
        }
        PointF pointF = kVar.f3357b;
        PointF pointF2 = kVar2.f3357b;
        float e2 = c.a.a.c0.f.e(pointF.x, pointF2.x, f2);
        float e3 = c.a.a.c0.f.e(pointF.y, pointF2.y, f2);
        if (kVar3.f3357b == null) {
            kVar3.f3357b = new PointF();
        }
        kVar3.f3357b.set(e2, e3);
        for (int size3 = kVar3.f3356a.size() - 1; size3 >= 0; size3--) {
            c.a.a.z.a aVar2 = kVar.f3356a.get(size3);
            c.a.a.z.a aVar3 = kVar2.f3356a.get(size3);
            PointF pointF3 = aVar2.f3258a;
            PointF pointF4 = aVar2.f3259b;
            PointF pointF5 = aVar2.f3260c;
            PointF pointF6 = aVar3.f3258a;
            PointF pointF7 = aVar3.f3259b;
            PointF pointF8 = aVar3.f3260c;
            kVar3.f3356a.get(size3).f3258a.set(c.a.a.c0.f.e(pointF3.x, pointF6.x, f2), c.a.a.c0.f.e(pointF3.y, pointF6.y, f2));
            kVar3.f3356a.get(size3).f3259b.set(c.a.a.c0.f.e(pointF4.x, pointF7.x, f2), c.a.a.c0.f.e(pointF4.y, pointF7.y, f2));
            kVar3.f3356a.get(size3).f3260c.set(c.a.a.c0.f.e(pointF5.x, pointF8.x, f2), c.a.a.c0.f.e(pointF5.y, pointF8.y, f2));
        }
        c.a.a.z.k.k kVar4 = this.i;
        Path path = this.j;
        path.reset();
        PointF pointF9 = kVar4.f3357b;
        path.moveTo(pointF9.x, pointF9.y);
        c.a.a.c0.f.f3030a.set(pointF9.x, pointF9.y);
        for (int i = 0; i < kVar4.f3356a.size(); i++) {
            c.a.a.z.a aVar4 = kVar4.f3356a.get(i);
            PointF pointF10 = aVar4.f3258a;
            PointF pointF11 = aVar4.f3259b;
            PointF pointF12 = aVar4.f3260c;
            if (pointF10.equals(c.a.a.c0.f.f3030a) && pointF11.equals(pointF12)) {
                path.lineTo(pointF12.x, pointF12.y);
            } else {
                path.cubicTo(pointF10.x, pointF10.y, pointF11.x, pointF11.y, pointF12.x, pointF12.y);
            }
            c.a.a.c0.f.f3030a.set(pointF12.x, pointF12.y);
        }
        if (kVar4.f3358c) {
            path.close();
        }
        return this.j;
    }
}