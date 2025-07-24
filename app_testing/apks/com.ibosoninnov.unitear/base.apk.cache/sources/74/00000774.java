package c.c.a.m.v;

import c.c.a.g;
import c.c.a.m.v.i;
import c.c.a.m.v.l;
import c.c.a.m.w.n;
import c.c.a.m.x.h.f;
import c.c.a.p.a;
import c.c.a.p.e;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/* compiled from: DecodeHelper.java */
/* loaded from: classes.dex */
public final class h<Transcode> {

    /* renamed from: a  reason: collision with root package name */
    public final List<n.a<?>> f3692a = new ArrayList();

    /* renamed from: b  reason: collision with root package name */
    public final List<c.c.a.m.m> f3693b = new ArrayList();

    /* renamed from: c  reason: collision with root package name */
    public c.c.a.d f3694c;

    /* renamed from: d  reason: collision with root package name */
    public Object f3695d;

    /* renamed from: e  reason: collision with root package name */
    public int f3696e;

    /* renamed from: f  reason: collision with root package name */
    public int f3697f;

    /* renamed from: g  reason: collision with root package name */
    public Class<?> f3698g;

    /* renamed from: h  reason: collision with root package name */
    public i.d f3699h;
    public c.c.a.m.p i;
    public Map<Class<?>, c.c.a.m.t<?>> j;
    public Class<Transcode> k;
    public boolean l;
    public boolean m;
    public c.c.a.m.m n;
    public c.c.a.f o;
    public k p;
    public boolean q;
    public boolean r;

    public List<c.c.a.m.m> a() {
        if (!this.m) {
            this.m = true;
            this.f3693b.clear();
            List<n.a<?>> c2 = c();
            int size = c2.size();
            for (int i = 0; i < size; i++) {
                n.a<?> aVar = c2.get(i);
                if (!this.f3693b.contains(aVar.f3863a)) {
                    this.f3693b.add(aVar.f3863a);
                }
                for (int i2 = 0; i2 < aVar.f3864b.size(); i2++) {
                    if (!this.f3693b.contains(aVar.f3864b.get(i2))) {
                        this.f3693b.add(aVar.f3864b.get(i2));
                    }
                }
            }
        }
        return this.f3693b;
    }

    public c.c.a.m.v.d0.a b() {
        return ((l.c) this.f3699h).a();
    }

    public List<n.a<?>> c() {
        if (!this.l) {
            this.l = true;
            this.f3692a.clear();
            List f2 = this.f3694c.f3427c.f(this.f3695d);
            int size = f2.size();
            for (int i = 0; i < size; i++) {
                n.a<?> b2 = ((c.c.a.m.w.n) f2.get(i)).b(this.f3695d, this.f3696e, this.f3697f, this.i);
                if (b2 != null) {
                    this.f3692a.add(b2);
                }
            }
        }
        return this.f3692a;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r19v0, resolved type: java.lang.Class<Data> */
    /* JADX WARN: Multi-variable type inference failed */
    public <Data> u<Data, ?, Transcode> d(Class<Data> cls) {
        u<Data, ?, Transcode> uVar;
        ArrayList arrayList;
        c.c.a.m.x.h.e eVar;
        c.c.a.g gVar = this.f3694c.f3427c;
        Class<?> cls2 = this.f3698g;
        Class cls3 = (Class<Transcode>) this.k;
        c.c.a.p.c cVar = gVar.i;
        c.c.a.s.i andSet = cVar.f4115c.getAndSet(null);
        if (andSet == null) {
            andSet = new c.c.a.s.i();
        }
        andSet.f4194a = cls;
        andSet.f4195b = cls2;
        andSet.f4196c = cls3;
        synchronized (cVar.f4114b) {
            uVar = (u<Data, ?, Transcode>) cVar.f4114b.getOrDefault(andSet, null);
        }
        cVar.f4115c.set(andSet);
        Objects.requireNonNull(gVar.i);
        if (c.c.a.p.c.f4113a.equals(uVar)) {
            return null;
        }
        if (uVar == null) {
            ArrayList arrayList2 = new ArrayList();
            Iterator it = ((ArrayList) gVar.f3442c.b(cls, cls2)).iterator();
            while (it.hasNext()) {
                Class<?> cls4 = (Class) it.next();
                Iterator it2 = ((ArrayList) gVar.f3445f.a(cls4, cls3)).iterator();
                while (it2.hasNext()) {
                    Class<?> cls5 = (Class) it2.next();
                    c.c.a.p.e eVar2 = gVar.f3442c;
                    synchronized (eVar2) {
                        arrayList = new ArrayList();
                        for (String str : eVar2.f4118a) {
                            List<e.a<?, ?>> list = eVar2.f4119b.get(str);
                            if (list != null) {
                                for (e.a<?, ?> aVar : list) {
                                    if (aVar.a(cls, cls4)) {
                                        arrayList.add(aVar.f4122c);
                                    }
                                }
                            }
                        }
                    }
                    c.c.a.m.x.h.f fVar = gVar.f3445f;
                    synchronized (fVar) {
                        if (cls5.isAssignableFrom(cls4)) {
                            eVar = c.c.a.m.x.h.g.f4074a;
                        } else {
                            for (f.a<?, ?> aVar2 : fVar.f4070a) {
                                if (aVar2.a(cls4, cls5)) {
                                    eVar = aVar2.f4073c;
                                }
                            }
                            throw new IllegalArgumentException("No transcoder registered to transcode from " + cls4 + " to " + cls5);
                        }
                    }
                    arrayList2.add(new j(cls, cls4, cls5, arrayList, eVar, gVar.j));
                }
            }
            u<Data, ?, Transcode> uVar2 = arrayList2.isEmpty() ? null : new u<>(cls, cls2, cls3, arrayList2, gVar.j);
            c.c.a.p.c cVar2 = gVar.i;
            synchronized (cVar2.f4114b) {
                cVar2.f4114b.put(new c.c.a.s.i(cls, cls2, cls3), uVar2 != null ? uVar2 : c.c.a.p.c.f4113a);
            }
            return uVar2;
        }
        return uVar;
    }

    /* JADX WARN: Code restructure failed: missing block: B:9:0x0025, code lost:
        r1 = (c.c.a.m.d<X>) r3.f4111b;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public <X> c.c.a.m.d<X> e(X x) {
        c.c.a.m.d<X> dVar;
        c.c.a.p.a aVar = this.f3694c.f3427c.f3441b;
        Class<?> cls = x.getClass();
        synchronized (aVar) {
            Iterator<a.C0083a<?>> it = aVar.f4109a.iterator();
            while (true) {
                if (!it.hasNext()) {
                    dVar = null;
                    break;
                }
                a.C0083a<?> next = it.next();
                if (next.f4110a.isAssignableFrom(cls)) {
                    break;
                }
            }
        }
        if (dVar != null) {
            return dVar;
        }
        throw new g.e(x.getClass());
    }

    public <Z> c.c.a.m.t<Z> f(Class<Z> cls) {
        c.c.a.m.t<Z> tVar = (c.c.a.m.t<Z>) this.j.get(cls);
        if (tVar == null) {
            Iterator<Map.Entry<Class<?>, c.c.a.m.t<?>>> it = this.j.entrySet().iterator();
            while (true) {
                if (!it.hasNext()) {
                    break;
                }
                Map.Entry<Class<?>, c.c.a.m.t<?>> next = it.next();
                if (next.getKey().isAssignableFrom(cls)) {
                    tVar = (c.c.a.m.t<Z>) next.getValue();
                    break;
                }
            }
        }
        if (tVar == null) {
            if (this.j.isEmpty() && this.q) {
                throw new IllegalArgumentException("Missing transformation for " + cls + ". If you wish to ignore unknown resource types, use the optional transformation methods.");
            }
            return (c.c.a.m.x.b) c.c.a.m.x.b.f3935b;
        }
        return tVar;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r1v0, resolved type: java.lang.Class<?> */
    /* JADX WARN: Multi-variable type inference failed */
    public boolean g(Class<?> cls) {
        return d(cls) != null;
    }
}