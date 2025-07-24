package c.a.a.z.k;

import android.graphics.PointF;

/* compiled from: PolystarShape.java */
/* loaded from: classes.dex */
public class h implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3338a;

    /* renamed from: b  reason: collision with root package name */
    public final int f3339b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.j.b f3340c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.j.m<PointF, PointF> f3341d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.z.j.b f3342e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.z.j.b f3343f;

    /* renamed from: g  reason: collision with root package name */
    public final c.a.a.z.j.b f3344g;

    /* renamed from: h  reason: collision with root package name */
    public final c.a.a.z.j.b f3345h;
    public final c.a.a.z.j.b i;
    public final boolean j;

    /* JADX WARN: Incorrect types in method signature: (Ljava/lang/String;Ljava/lang/Object;Lc/a/a/z/j/b;Lc/a/a/z/j/m<Landroid/graphics/PointF;Landroid/graphics/PointF;>;Lc/a/a/z/j/b;Lc/a/a/z/j/b;Lc/a/a/z/j/b;Lc/a/a/z/j/b;Lc/a/a/z/j/b;Z)V */
    public h(String str, int i, c.a.a.z.j.b bVar, c.a.a.z.j.m mVar, c.a.a.z.j.b bVar2, c.a.a.z.j.b bVar3, c.a.a.z.j.b bVar4, c.a.a.z.j.b bVar5, c.a.a.z.j.b bVar6, boolean z) {
        this.f3338a = str;
        this.f3339b = i;
        this.f3340c = bVar;
        this.f3341d = mVar;
        this.f3342e = bVar2;
        this.f3343f = bVar3;
        this.f3344g = bVar4;
        this.f3345h = bVar5;
        this.i = bVar6;
        this.j = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new c.a.a.x.b.n(jVar, bVar, this);
    }
}