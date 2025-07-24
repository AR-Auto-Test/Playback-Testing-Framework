package c.a.a.z.k;

import c.a.a.x.b.r;
import java.util.List;

/* compiled from: ShapeStroke.java */
/* loaded from: classes.dex */
public class o implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3372a;

    /* renamed from: b  reason: collision with root package name */
    public final c.a.a.z.j.b f3373b;

    /* renamed from: c  reason: collision with root package name */
    public final List<c.a.a.z.j.b> f3374c;

    /* renamed from: d  reason: collision with root package name */
    public final c.a.a.z.j.a f3375d;

    /* renamed from: e  reason: collision with root package name */
    public final c.a.a.z.j.d f3376e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.z.j.b f3377f;

    /* renamed from: g  reason: collision with root package name */
    public final int f3378g;

    /* renamed from: h  reason: collision with root package name */
    public final int f3379h;
    public final float i;
    public final boolean j;

    /* JADX WARN: Incorrect types in method signature: (Ljava/lang/String;Lc/a/a/z/j/b;Ljava/util/List<Lc/a/a/z/j/b;>;Lc/a/a/z/j/a;Lc/a/a/z/j/d;Lc/a/a/z/j/b;Ljava/lang/Object;Ljava/lang/Object;FZ)V */
    public o(String str, c.a.a.z.j.b bVar, List list, c.a.a.z.j.a aVar, c.a.a.z.j.d dVar, c.a.a.z.j.b bVar2, int i, int i2, float f2, boolean z) {
        this.f3372a = str;
        this.f3373b = bVar;
        this.f3374c = list;
        this.f3375d = aVar;
        this.f3376e = dVar;
        this.f3377f = bVar2;
        this.f3378g = i;
        this.f3379h = i2;
        this.i = f2;
        this.j = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new r(jVar, bVar, this);
    }
}