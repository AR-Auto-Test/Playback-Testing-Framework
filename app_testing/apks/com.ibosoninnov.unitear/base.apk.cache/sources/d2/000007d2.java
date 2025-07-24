package c.c.a.m.w;

import c.c.a.g;
import c.c.a.m.w.n;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/* compiled from: MultiModelLoaderFactory.java */
/* loaded from: classes.dex */
public class r {

    /* renamed from: a  reason: collision with root package name */
    public static final c f3879a = new c();

    /* renamed from: b  reason: collision with root package name */
    public static final n<Object, Object> f3880b = new a();

    /* renamed from: c  reason: collision with root package name */
    public final List<b<?, ?>> f3881c;

    /* renamed from: d  reason: collision with root package name */
    public final c f3882d;

    /* renamed from: e  reason: collision with root package name */
    public final Set<b<?, ?>> f3883e;

    /* renamed from: f  reason: collision with root package name */
    public final b.j.i.d<List<Throwable>> f3884f;

    /* compiled from: MultiModelLoaderFactory.java */
    /* loaded from: classes.dex */
    public static class a implements n<Object, Object> {
        @Override // c.c.a.m.w.n
        public boolean a(Object obj) {
            return false;
        }

        @Override // c.c.a.m.w.n
        public n.a<Object> b(Object obj, int i, int i2, c.c.a.m.p pVar) {
            return null;
        }
    }

    /* compiled from: MultiModelLoaderFactory.java */
    /* loaded from: classes.dex */
    public static class b<Model, Data> {

        /* renamed from: a  reason: collision with root package name */
        public final Class<Model> f3885a;

        /* renamed from: b  reason: collision with root package name */
        public final Class<Data> f3886b;

        /* renamed from: c  reason: collision with root package name */
        public final o<? extends Model, ? extends Data> f3887c;

        public b(Class<Model> cls, Class<Data> cls2, o<? extends Model, ? extends Data> oVar) {
            this.f3885a = cls;
            this.f3886b = cls2;
            this.f3887c = oVar;
        }
    }

    /* compiled from: MultiModelLoaderFactory.java */
    /* loaded from: classes.dex */
    public static class c {
    }

    public r(b.j.i.d<List<Throwable>> dVar) {
        c cVar = f3879a;
        this.f3881c = new ArrayList();
        this.f3883e = new HashSet();
        this.f3884f = dVar;
        this.f3882d = cVar;
    }

    public final <Model, Data> n<Model, Data> a(b<?, ?> bVar) {
        n<Model, Data> nVar = (n<Model, Data>) bVar.f3887c.b(this);
        Objects.requireNonNull(nVar, "Argument must not be null");
        return nVar;
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public synchronized <Model, Data> n<Model, Data> b(Class<Model> cls, Class<Data> cls2) {
        try {
            ArrayList arrayList = new ArrayList();
            boolean z = false;
            for (b<?, ?> bVar : this.f3881c) {
                if (this.f3883e.contains(bVar)) {
                    z = true;
                } else if (bVar.f3885a.isAssignableFrom(cls) && bVar.f3886b.isAssignableFrom(cls2)) {
                    this.f3883e.add(bVar);
                    arrayList.add(a(bVar));
                    this.f3883e.remove(bVar);
                }
            }
            if (arrayList.size() > 1) {
                c cVar = this.f3882d;
                b.j.i.d<List<Throwable>> dVar = this.f3884f;
                Objects.requireNonNull(cVar);
                return new q(arrayList, dVar);
            } else if (arrayList.size() == 1) {
                return (n) arrayList.get(0);
            } else if (z) {
                return (n<Model, Data>) f3880b;
            } else {
                throw new g.c((Class<?>) cls, (Class<?>) cls2);
            }
        } catch (Throwable th) {
            this.f3883e.clear();
            throw th;
        }
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public synchronized <Model> List<n<Model, ?>> c(Class<Model> cls) {
        ArrayList arrayList;
        try {
            arrayList = new ArrayList();
            for (b<?, ?> bVar : this.f3881c) {
                if (!this.f3883e.contains(bVar) && bVar.f3885a.isAssignableFrom(cls)) {
                    this.f3883e.add(bVar);
                    n<? extends Object, ? extends Object> b2 = bVar.f3887c.b(this);
                    Objects.requireNonNull(b2, "Argument must not be null");
                    arrayList.add(b2);
                    this.f3883e.remove(bVar);
                }
            }
        } catch (Throwable th) {
            this.f3883e.clear();
            throw th;
        }
        return arrayList;
    }

    public synchronized List<Class<?>> d(Class<?> cls) {
        ArrayList arrayList;
        arrayList = new ArrayList();
        for (b<?, ?> bVar : this.f3881c) {
            if (!arrayList.contains(bVar.f3886b) && bVar.f3885a.isAssignableFrom(cls)) {
                arrayList.add(bVar.f3886b);
            }
        }
        return arrayList;
    }
}