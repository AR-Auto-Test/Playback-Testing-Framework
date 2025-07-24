package c.c.a.m.v;

import android.util.Log;
import c.c.a.g;
import c.c.a.m.v.i;
import c.c.a.m.w.n;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: DecodePath.java */
/* loaded from: classes.dex */
public class j<DataType, ResourceType, Transcode> {

    /* renamed from: a  reason: collision with root package name */
    public final Class<DataType> f3726a;

    /* renamed from: b  reason: collision with root package name */
    public final List<? extends c.c.a.m.r<DataType, ResourceType>> f3727b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.x.h.e<ResourceType, Transcode> f3728c;

    /* renamed from: d  reason: collision with root package name */
    public final b.j.i.d<List<Throwable>> f3729d;

    /* renamed from: e  reason: collision with root package name */
    public final String f3730e;

    /* compiled from: DecodePath.java */
    /* loaded from: classes.dex */
    public interface a<ResourceType> {
    }

    public j(Class<DataType> cls, Class<ResourceType> cls2, Class<Transcode> cls3, List<? extends c.c.a.m.r<DataType, ResourceType>> list, c.c.a.m.x.h.e<ResourceType, Transcode> eVar, b.j.i.d<List<Throwable>> dVar) {
        this.f3726a = cls;
        this.f3727b = list;
        this.f3728c = eVar;
        this.f3729d = dVar;
        StringBuilder x = c.b.a.a.a.x("Failed DecodePath{");
        x.append(cls.getSimpleName());
        x.append("->");
        x.append(cls2.getSimpleName());
        x.append("->");
        x.append(cls3.getSimpleName());
        x.append("}");
        this.f3730e = x.toString();
    }

    public w<Transcode> a(c.c.a.m.u.e<DataType> eVar, int i, int i2, c.c.a.m.p pVar, a<ResourceType> aVar) {
        w<ResourceType> wVar;
        c.c.a.m.t tVar;
        c.c.a.m.c cVar;
        c.c.a.m.m eVar2;
        List<Throwable> b2 = this.f3729d.b();
        Objects.requireNonNull(b2, "Argument must not be null");
        List<Throwable> list = b2;
        try {
            w<ResourceType> b3 = b(eVar, i, i2, pVar, list);
            this.f3729d.a(list);
            i.b bVar = (i.b) aVar;
            i iVar = i.this;
            c.c.a.m.a aVar2 = bVar.f3707a;
            Objects.requireNonNull(iVar);
            Class<?> cls = b3.get().getClass();
            c.c.a.m.s sVar = null;
            if (aVar2 != c.c.a.m.a.RESOURCE_DISK_CACHE) {
                c.c.a.m.t f2 = iVar.f3700b.f(cls);
                tVar = f2;
                wVar = f2.b(iVar.i, b3, iVar.m, iVar.n);
            } else {
                wVar = b3;
                tVar = null;
            }
            if (!b3.equals(wVar)) {
                b3.a();
            }
            boolean z = false;
            if (iVar.f3700b.f3694c.f3427c.f3443d.a(wVar.d()) != null) {
                sVar = iVar.f3700b.f3694c.f3427c.f3443d.a(wVar.d());
                if (sVar != null) {
                    cVar = sVar.b(iVar.p);
                } else {
                    throw new g.d(wVar.d());
                }
            } else {
                cVar = c.c.a.m.c.NONE;
            }
            c.c.a.m.s sVar2 = sVar;
            h<R> hVar = iVar.f3700b;
            c.c.a.m.m mVar = iVar.y;
            List<n.a<?>> c2 = hVar.c();
            int size = c2.size();
            int i3 = 0;
            while (true) {
                if (i3 >= size) {
                    break;
                } else if (c2.get(i3).f3863a.equals(mVar)) {
                    z = true;
                    break;
                } else {
                    i3++;
                }
            }
            w<ResourceType> wVar2 = wVar;
            if (iVar.o.d(!z, aVar2, cVar)) {
                if (sVar2 != null) {
                    int ordinal = cVar.ordinal();
                    if (ordinal == 0) {
                        eVar2 = new e(iVar.y, iVar.j);
                    } else if (ordinal == 1) {
                        eVar2 = new y(iVar.f3700b.f3694c.f3426b, iVar.y, iVar.j, iVar.m, iVar.n, tVar, cls, iVar.p);
                    } else {
                        throw new IllegalArgumentException("Unknown strategy: " + cVar);
                    }
                    v<Z> e2 = v.e(wVar);
                    i.c<?> cVar2 = iVar.f3705g;
                    cVar2.f3709a = eVar2;
                    cVar2.f3710b = sVar2;
                    cVar2.f3711c = e2;
                    wVar2 = e2;
                } else {
                    throw new g.d(wVar.get().getClass());
                }
            }
            return this.f3728c.a(wVar2, pVar);
        } catch (Throwable th) {
            this.f3729d.a(list);
            throw th;
        }
    }

    public final w<ResourceType> b(c.c.a.m.u.e<DataType> eVar, int i, int i2, c.c.a.m.p pVar, List<Throwable> list) {
        int size = this.f3727b.size();
        w<ResourceType> wVar = null;
        for (int i3 = 0; i3 < size; i3++) {
            c.c.a.m.r<DataType, ResourceType> rVar = this.f3727b.get(i3);
            try {
                if (rVar.a(eVar.a(), pVar)) {
                    wVar = rVar.b(eVar.a(), i, i2, pVar);
                }
            } catch (IOException | OutOfMemoryError | RuntimeException e2) {
                if (Log.isLoggable("DecodePath", 2)) {
                    Log.v("DecodePath", "Failed to decode data for " + rVar, e2);
                }
                list.add(e2);
            }
            if (wVar != null) {
                break;
            }
        }
        if (wVar != null) {
            return wVar;
        }
        throw new r(this.f3730e, new ArrayList(list));
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("DecodePath{ dataClass=");
        x.append(this.f3726a);
        x.append(", decoders=");
        x.append(this.f3727b);
        x.append(", transcoder=");
        x.append(this.f3728c);
        x.append('}');
        return x.toString();
    }
}