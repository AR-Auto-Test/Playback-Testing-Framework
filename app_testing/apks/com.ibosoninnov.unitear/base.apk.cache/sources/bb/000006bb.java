package c.a.a.z.j;

import android.graphics.PointF;
import java.util.List;

/* compiled from: AnimatableSplitDimensionPathValue.java */
/* loaded from: classes.dex */
public class i implements m<PointF, PointF> {

    /* renamed from: a  reason: collision with root package name */
    public final b f3287a;

    /* renamed from: b  reason: collision with root package name */
    public final b f3288b;

    public i(b bVar, b bVar2) {
        this.f3287a = bVar;
        this.f3288b = bVar2;
    }

    @Override // c.a.a.z.j.m
    public c.a.a.x.c.a<PointF, PointF> a() {
        return new c.a.a.x.c.m(this.f3287a.a(), this.f3288b.a());
    }

    @Override // c.a.a.z.j.m
    public List<c.a.a.d0.a<PointF>> b() {
        throw new UnsupportedOperationException("Cannot call getKeyframes on AnimatableSplitDimensionPathValue.");
    }

    @Override // c.a.a.z.j.m
    public boolean c() {
        return this.f3287a.c() && this.f3288b.c();
    }
}