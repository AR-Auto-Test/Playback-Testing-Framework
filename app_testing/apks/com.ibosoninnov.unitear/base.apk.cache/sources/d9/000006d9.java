package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.RectF;
import c.a.a.j;
import c.a.a.z.k.m;
import java.util.Collections;
import java.util.List;

/* compiled from: ShapeLayer.java */
/* loaded from: classes.dex */
public class g extends b {
    public final c.a.a.x.b.d x;

    public g(j jVar, e eVar) {
        super(jVar, eVar);
        c.a.a.x.b.d dVar = new c.a.a.x.b.d(jVar, this, new m("__container", eVar.f3395a, false));
        this.x = dVar;
        dVar.b(Collections.emptyList(), Collections.emptyList());
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        this.x.d(rectF, this.m, z);
    }

    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
        this.x.f(canvas, matrix, i);
    }

    @Override // c.a.a.z.l.b
    public void o(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        this.x.c(eVar, i, list, eVar2);
    }
}