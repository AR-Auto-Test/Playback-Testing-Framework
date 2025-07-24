package c.a.a.z.k;

import android.graphics.PointF;

/* compiled from: CircleShape.java */
/* loaded from: classes.dex */
public class a implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3302a;

    /* renamed from: b  reason: collision with root package name */
    public final c.a.a.z.j.m<PointF, PointF> f3303b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.z.j.f f3304c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f3305d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f3306e;

    public a(String str, c.a.a.z.j.m<PointF, PointF> mVar, c.a.a.z.j.f fVar, boolean z, boolean z2) {
        this.f3302a = str;
        this.f3303b = mVar;
        this.f3304c = fVar;
        this.f3305d = z;
        this.f3306e = z2;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new c.a.a.x.b.f(jVar, bVar, this);
    }
}