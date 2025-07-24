package c.a.a.z.j;

import android.graphics.PointF;
import java.util.List;

/* compiled from: AnimatablePathValue.java */
/* loaded from: classes.dex */
public class e implements m<PointF, PointF> {

    /* renamed from: a  reason: collision with root package name */
    public final List<c.a.a.d0.a<PointF>> f3286a;

    public e(List<c.a.a.d0.a<PointF>> list) {
        this.f3286a = list;
    }

    @Override // c.a.a.z.j.m
    public c.a.a.x.c.a<PointF, PointF> a() {
        if (this.f3286a.get(0).d()) {
            return new c.a.a.x.c.j(this.f3286a);
        }
        return new c.a.a.x.c.i(this.f3286a);
    }

    @Override // c.a.a.z.j.m
    public List<c.a.a.d0.a<PointF>> b() {
        return this.f3286a;
    }

    @Override // c.a.a.z.j.m
    public boolean c() {
        return this.f3286a.size() == 1 && this.f3286a.get(0).d();
    }
}