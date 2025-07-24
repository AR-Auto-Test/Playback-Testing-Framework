package c.c.a.m.v;

import c.c.a.m.u.d;
import c.c.a.m.v.g;
import c.c.a.m.w.n;
import java.io.File;
import java.util.List;

/* compiled from: DataCacheGenerator.java */
/* loaded from: classes.dex */
public class d implements g, d.a<Object> {

    /* renamed from: b  reason: collision with root package name */
    public final List<c.c.a.m.m> f3642b;

    /* renamed from: c  reason: collision with root package name */
    public final h<?> f3643c;

    /* renamed from: d  reason: collision with root package name */
    public final g.a f3644d;

    /* renamed from: e  reason: collision with root package name */
    public int f3645e;

    /* renamed from: f  reason: collision with root package name */
    public c.c.a.m.m f3646f;

    /* renamed from: g  reason: collision with root package name */
    public List<c.c.a.m.w.n<File, ?>> f3647g;

    /* renamed from: h  reason: collision with root package name */
    public int f3648h;
    public volatile n.a<?> i;
    public File j;

    public d(h<?> hVar, g.a aVar) {
        List<c.c.a.m.m> a2 = hVar.a();
        this.f3645e = -1;
        this.f3642b = a2;
        this.f3643c = hVar;
        this.f3644d = aVar;
    }

    @Override // c.c.a.m.v.g
    public boolean b() {
        while (true) {
            List<c.c.a.m.w.n<File, ?>> list = this.f3647g;
            if (list != null) {
                if (this.f3648h < list.size()) {
                    this.i = null;
                    boolean z = false;
                    while (!z) {
                        if (!(this.f3648h < this.f3647g.size())) {
                            break;
                        }
                        List<c.c.a.m.w.n<File, ?>> list2 = this.f3647g;
                        int i = this.f3648h;
                        this.f3648h = i + 1;
                        File file = this.j;
                        h<?> hVar = this.f3643c;
                        this.i = list2.get(i).b(file, hVar.f3696e, hVar.f3697f, hVar.i);
                        if (this.i != null && this.f3643c.g(this.i.f3865c.a())) {
                            this.i.f3865c.e(this.f3643c.o, this);
                            z = true;
                        }
                    }
                    return z;
                }
            }
            int i2 = this.f3645e + 1;
            this.f3645e = i2;
            if (i2 >= this.f3642b.size()) {
                return false;
            }
            c.c.a.m.m mVar = this.f3642b.get(this.f3645e);
            h<?> hVar2 = this.f3643c;
            File b2 = hVar2.b().b(new e(mVar, hVar2.n));
            this.j = b2;
            if (b2 != null) {
                this.f3646f = mVar;
                this.f3647g = this.f3643c.f3694c.f3427c.f(b2);
                this.f3648h = 0;
            }
        }
    }

    @Override // c.c.a.m.u.d.a
    public void c(Exception exc) {
        this.f3644d.a(this.f3646f, exc, this.i.f3865c, c.c.a.m.a.DATA_DISK_CACHE);
    }

    @Override // c.c.a.m.v.g
    public void cancel() {
        n.a<?> aVar = this.i;
        if (aVar != null) {
            aVar.f3865c.cancel();
        }
    }

    @Override // c.c.a.m.u.d.a
    public void f(Object obj) {
        this.f3644d.d(this.f3646f, obj, this.i.f3865c, c.c.a.m.a.DATA_DISK_CACHE, this.f3646f);
    }

    public d(List<c.c.a.m.m> list, h<?> hVar, g.a aVar) {
        this.f3645e = -1;
        this.f3642b = list;
        this.f3643c = hVar;
        this.f3644d = aVar;
    }
}