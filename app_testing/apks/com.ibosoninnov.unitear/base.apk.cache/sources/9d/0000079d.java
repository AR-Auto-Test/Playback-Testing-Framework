package c.c.a.m.v;

import c.c.a.m.u.d;
import c.c.a.m.v.g;
import c.c.a.m.w.n;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/* compiled from: ResourceCacheGenerator.java */
/* loaded from: classes.dex */
public class x implements g, d.a<Object> {

    /* renamed from: b  reason: collision with root package name */
    public final g.a f3805b;

    /* renamed from: c  reason: collision with root package name */
    public final h<?> f3806c;

    /* renamed from: d  reason: collision with root package name */
    public int f3807d;

    /* renamed from: e  reason: collision with root package name */
    public int f3808e = -1;

    /* renamed from: f  reason: collision with root package name */
    public c.c.a.m.m f3809f;

    /* renamed from: g  reason: collision with root package name */
    public List<c.c.a.m.w.n<File, ?>> f3810g;

    /* renamed from: h  reason: collision with root package name */
    public int f3811h;
    public volatile n.a<?> i;
    public File j;
    public y k;

    public x(h<?> hVar, g.a aVar) {
        this.f3806c = hVar;
        this.f3805b = aVar;
    }

    @Override // c.c.a.m.v.g
    public boolean b() {
        List<Class<?>> orDefault;
        List<Class<?>> d2;
        List<c.c.a.m.m> a2 = this.f3806c.a();
        if (a2.isEmpty()) {
            return false;
        }
        h<?> hVar = this.f3806c;
        c.c.a.g gVar = hVar.f3694c.f3427c;
        Class<?> cls = hVar.f3695d.getClass();
        Class<?> cls2 = hVar.f3698g;
        Class<?> cls3 = hVar.k;
        c.c.a.p.d dVar = gVar.f3447h;
        c.c.a.s.i andSet = dVar.f4116a.getAndSet(null);
        if (andSet == null) {
            andSet = new c.c.a.s.i(cls, cls2, cls3);
        } else {
            andSet.f4194a = cls;
            andSet.f4195b = cls2;
            andSet.f4196c = cls3;
        }
        synchronized (dVar.f4117b) {
            orDefault = dVar.f4117b.getOrDefault(andSet, null);
        }
        dVar.f4116a.set(andSet);
        ArrayList arrayList = orDefault;
        if (orDefault == null) {
            ArrayList arrayList2 = new ArrayList();
            c.c.a.m.w.p pVar = gVar.f3440a;
            synchronized (pVar) {
                d2 = pVar.f3866a.d(cls);
            }
            Iterator it = ((ArrayList) d2).iterator();
            while (it.hasNext()) {
                Iterator it2 = ((ArrayList) gVar.f3442c.b((Class) it.next(), cls2)).iterator();
                while (it2.hasNext()) {
                    Class cls4 = (Class) it2.next();
                    if (!((ArrayList) gVar.f3445f.a(cls4, cls3)).isEmpty() && !arrayList2.contains(cls4)) {
                        arrayList2.add(cls4);
                    }
                }
            }
            c.c.a.p.d dVar2 = gVar.f3447h;
            List<Class<?>> unmodifiableList = Collections.unmodifiableList(arrayList2);
            synchronized (dVar2.f4117b) {
                dVar2.f4117b.put(new c.c.a.s.i(cls, cls2, cls3), unmodifiableList);
            }
            arrayList = arrayList2;
        }
        if (arrayList.isEmpty()) {
            if (File.class.equals(this.f3806c.k)) {
                return false;
            }
            StringBuilder x = c.b.a.a.a.x("Failed to find any load path from ");
            x.append(this.f3806c.f3695d.getClass());
            x.append(" to ");
            x.append(this.f3806c.k);
            throw new IllegalStateException(x.toString());
        }
        while (true) {
            List<c.c.a.m.w.n<File, ?>> list = this.f3810g;
            if (list != null) {
                if (this.f3811h < list.size()) {
                    this.i = null;
                    boolean z = false;
                    while (!z) {
                        if (!(this.f3811h < this.f3810g.size())) {
                            break;
                        }
                        List<c.c.a.m.w.n<File, ?>> list2 = this.f3810g;
                        int i = this.f3811h;
                        this.f3811h = i + 1;
                        File file = this.j;
                        h<?> hVar2 = this.f3806c;
                        this.i = list2.get(i).b(file, hVar2.f3696e, hVar2.f3697f, hVar2.i);
                        if (this.i != null && this.f3806c.g(this.i.f3865c.a())) {
                            this.i.f3865c.e(this.f3806c.o, this);
                            z = true;
                        }
                    }
                    return z;
                }
            }
            int i2 = this.f3808e + 1;
            this.f3808e = i2;
            if (i2 >= arrayList.size()) {
                int i3 = this.f3807d + 1;
                this.f3807d = i3;
                if (i3 >= a2.size()) {
                    return false;
                }
                this.f3808e = 0;
            }
            c.c.a.m.m mVar = a2.get(this.f3807d);
            Class<?> cls5 = arrayList.get(this.f3808e);
            c.c.a.m.t<Z> f2 = this.f3806c.f(cls5);
            h<?> hVar3 = this.f3806c;
            this.k = new y(hVar3.f3694c.f3426b, mVar, hVar3.n, hVar3.f3696e, hVar3.f3697f, f2, cls5, hVar3.i);
            File b2 = hVar3.b().b(this.k);
            this.j = b2;
            if (b2 != null) {
                this.f3809f = mVar;
                this.f3810g = this.f3806c.f3694c.f3427c.f(b2);
                this.f3811h = 0;
            }
        }
    }

    @Override // c.c.a.m.u.d.a
    public void c(Exception exc) {
        this.f3805b.a(this.k, exc, this.i.f3865c, c.c.a.m.a.RESOURCE_DISK_CACHE);
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
        this.f3805b.d(this.f3809f, obj, this.i.f3865c, c.c.a.m.a.RESOURCE_DISK_CACHE, this.k);
    }
}