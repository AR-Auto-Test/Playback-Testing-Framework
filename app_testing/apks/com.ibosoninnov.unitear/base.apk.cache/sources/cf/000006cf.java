package c.a.a.z.k;

import c.a.a.x.b.q;

/* compiled from: ShapePath.java */
/* loaded from: classes.dex */
public class n implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3368a;

    /* renamed from: b  reason: collision with root package name */
    public final int f3369b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.j.h f3370c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f3371d;

    public n(String str, int i, c.a.a.z.j.h hVar, boolean z) {
        this.f3368a = str;
        this.f3369b = i;
        this.f3370c = hVar;
        this.f3371d = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new q(jVar, bVar, this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ShapePath{name=");
        x.append(this.f3368a);
        x.append(", index=");
        x.append(this.f3369b);
        x.append('}');
        return x.toString();
    }
}