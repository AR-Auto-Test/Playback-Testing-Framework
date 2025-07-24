package c.c.a.m.x.g;

import c.c.a.l.a;

/* compiled from: GifBitmapProvider.java */
/* loaded from: classes.dex */
public final class b implements a.InterfaceC0064a {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f4034a;

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f4035b;

    public b(c.c.a.m.v.c0.d dVar, c.c.a.m.v.c0.b bVar) {
        this.f4034a = dVar;
        this.f4035b = bVar;
    }

    public byte[] a(int i) {
        c.c.a.m.v.c0.b bVar = this.f4035b;
        if (bVar == null) {
            return new byte[i];
        }
        return (byte[]) bVar.d(i, byte[].class);
    }
}