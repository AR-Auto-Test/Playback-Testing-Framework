package c.a.a.x.c;

import android.graphics.PointF;
import java.util.Collections;

/* compiled from: SplitDimensionPathKeyframeAnimation.java */
/* loaded from: classes.dex */
public class m extends a<PointF, PointF> {
    public final PointF i;
    public final a<Float, Float> j;
    public final a<Float, Float> k;

    public m(a<Float, Float> aVar, a<Float, Float> aVar2) {
        super(Collections.emptyList());
        this.i = new PointF();
        this.j = aVar;
        this.k = aVar2;
        h(this.f3226d);
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.x.c.a
    public PointF e() {
        return this.i;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [c.a.a.d0.a, float] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.x.c.a
    public PointF f(c.a.a.d0.a<PointF> aVar, float f2) {
        return this.i;
    }

    @Override // c.a.a.x.c.a
    public void h(float f2) {
        this.j.h(f2);
        this.k.h(f2);
        this.i.set(this.j.e().floatValue(), this.k.e().floatValue());
        for (int i = 0; i < this.f3223a.size(); i++) {
            this.f3223a.get(i).a();
        }
    }
}