package c.a.a.z.k;

import c.a.a.x.b.s;

/* compiled from: ShapeTrimPath.java */
/* loaded from: classes.dex */
public class p implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3380a;

    /* renamed from: b  reason: collision with root package name */
    public final int f3381b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.j.b f3382c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.j.b f3383d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.z.j.b f3384e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f3385f;

    public p(String str, int i, c.a.a.z.j.b bVar, c.a.a.z.j.b bVar2, c.a.a.z.j.b bVar3, boolean z) {
        this.f3380a = str;
        this.f3381b = i;
        this.f3382c = bVar;
        this.f3383d = bVar2;
        this.f3384e = bVar3;
        this.f3385f = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new s(bVar, this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Trim Path: {start: ");
        x.append(this.f3382c);
        x.append(", end: ");
        x.append(this.f3383d);
        x.append(", offset: ");
        x.append(this.f3384e);
        x.append("}");
        return x.toString();
    }
}