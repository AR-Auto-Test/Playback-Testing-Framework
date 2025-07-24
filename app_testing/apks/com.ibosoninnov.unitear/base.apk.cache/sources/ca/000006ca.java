package c.a.a.z.k;

import android.graphics.PointF;

/* compiled from: RectangleShape.java */
/* loaded from: classes.dex */
public class i implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3346a;

    /* renamed from: b  reason: collision with root package name */
    public final c.a.a.z.j.m<PointF, PointF> f3347b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.j.f f3348c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.j.b f3349d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f3350e;

    public i(String str, c.a.a.z.j.m<PointF, PointF> mVar, c.a.a.z.j.f fVar, c.a.a.z.j.b bVar, boolean z) {
        this.f3346a = str;
        this.f3347b = mVar;
        this.f3348c = fVar;
        this.f3349d = bVar;
        this.f3350e = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new c.a.a.x.b.o(jVar, bVar, this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("RectangleShape{position=");
        x.append(this.f3347b);
        x.append(", size=");
        x.append(this.f3348c);
        x.append('}');
        return x.toString();
    }
}