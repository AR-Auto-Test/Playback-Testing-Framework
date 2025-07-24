package c.a.a.z.k;

import android.graphics.Path;

/* compiled from: ShapeFill.java */
/* loaded from: classes.dex */
public class l implements b {

    /* renamed from: a  reason: collision with root package name */
    public final boolean f3359a;

    /* renamed from: b  reason: collision with root package name */
    public final Path.FillType f3360b;

    /* renamed from: c  reason: collision with root package name */
    public final String f3361c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.j.a f3362d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.z.j.d f3363e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f3364f;

    public l(String str, boolean z, Path.FillType fillType, c.a.a.z.j.a aVar, c.a.a.z.j.d dVar, boolean z2) {
        this.f3361c = str;
        this.f3359a = z;
        this.f3360b = fillType;
        this.f3362d = aVar;
        this.f3363e = dVar;
        this.f3364f = z2;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new c.a.a.x.b.g(jVar, bVar, this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ShapeFill{color=, fillEnabled=");
        x.append(this.f3359a);
        x.append('}');
        return x.toString();
    }
}