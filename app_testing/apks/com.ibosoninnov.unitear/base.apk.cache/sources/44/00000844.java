package c.c.a.m.x.g;

import android.graphics.Bitmap;
import c.c.a.m.v.s;
import c.c.a.m.x.g.g;

/* compiled from: GifDrawableResource.java */
/* loaded from: classes.dex */
public class e extends c.c.a.m.x.e.b<c> implements s {
    public e(c cVar) {
        super(cVar);
    }

    @Override // c.c.a.m.v.w
    public void a() {
        ((c) this.f4023b).stop();
        c cVar = (c) this.f4023b;
        cVar.f4039e = true;
        g gVar = cVar.f4036b.f4043a;
        gVar.f4047c.clear();
        Bitmap bitmap = gVar.l;
        if (bitmap != null) {
            gVar.f4049e.d(bitmap);
            gVar.l = null;
        }
        gVar.f4050f = false;
        g.a aVar = gVar.i;
        if (aVar != null) {
            gVar.f4048d.i(aVar);
            gVar.i = null;
        }
        g.a aVar2 = gVar.k;
        if (aVar2 != null) {
            gVar.f4048d.i(aVar2);
            gVar.k = null;
        }
        g.a aVar3 = gVar.n;
        if (aVar3 != null) {
            gVar.f4048d.i(aVar3);
            gVar.n = null;
        }
        gVar.f4045a.clear();
        gVar.j = true;
    }

    @Override // c.c.a.m.v.w
    public int c() {
        g gVar = ((c) this.f4023b).f4036b.f4043a;
        return gVar.f4045a.g() + gVar.o;
    }

    @Override // c.c.a.m.v.w
    public Class<c> d() {
        return c.class;
    }

    @Override // c.c.a.m.x.e.b, c.c.a.m.v.s
    public void initialize() {
        ((c) this.f4023b).b().prepareToDraw();
    }
}