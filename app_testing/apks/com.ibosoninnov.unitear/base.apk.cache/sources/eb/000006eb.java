package c.c.a;

import c.c.a.m.s;
import c.c.a.m.u.e;
import c.c.a.m.w.n;
import c.c.a.m.w.o;
import c.c.a.m.w.p;
import c.c.a.m.w.r;
import c.c.a.m.x.h.f;
import c.c.a.p.a;
import c.c.a.p.e;
import c.c.a.p.f;
import c.c.a.s.k.a;
import com.bumptech.glide.load.ImageHeaderParser;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/* compiled from: Registry.java */
/* loaded from: classes.dex */
public class g {

    /* renamed from: a  reason: collision with root package name */
    public final p f3440a;

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.p.a f3441b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.p.e f3442c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.p.f f3443d;

    /* renamed from: e  reason: collision with root package name */
    public final c.c.a.m.u.f f3444e;

    /* renamed from: f  reason: collision with root package name */
    public final c.c.a.m.x.h.f f3445f;

    /* renamed from: g  reason: collision with root package name */
    public final c.c.a.p.b f3446g;

    /* renamed from: h  reason: collision with root package name */
    public final c.c.a.p.d f3447h = new c.c.a.p.d();
    public final c.c.a.p.c i = new c.c.a.p.c();
    public final b.j.i.d<List<Throwable>> j;

    /* compiled from: Registry.java */
    /* loaded from: classes.dex */
    public static class a extends RuntimeException {
        public a(String str) {
            super(str);
        }
    }

    /* compiled from: Registry.java */
    /* loaded from: classes.dex */
    public static final class b extends a {
        public b() {
            super("Failed to find image header parser.");
        }
    }

    /* compiled from: Registry.java */
    /* loaded from: classes.dex */
    public static class c extends a {
        /* JADX WARN: Illegal instructions before constructor call */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public c(Object obj) {
            super(r0.toString());
            StringBuilder x = c.b.a.a.a.x("Failed to find any ModelLoaders registered for model class: ");
            x.append(obj.getClass());
        }

        public <M> c(M m, List<n<M, ?>> list) {
            super("Found ModelLoaders for model class: " + list + ", but none that handle this specific model instance: " + m);
        }

        public c(Class<?> cls, Class<?> cls2) {
            super("Failed to find any ModelLoaders for model: " + cls + " and data: " + cls2);
        }
    }

    /* compiled from: Registry.java */
    /* loaded from: classes.dex */
    public static class d extends a {
        public d(Class<?> cls) {
            super("Failed to find result encoder for resource class: " + cls + ", you may need to consider registering a new Encoder for the requested type or DiskCacheStrategy.DATA/DiskCacheStrategy.NONE if caching your transformed resource is unnecessary.");
        }
    }

    /* compiled from: Registry.java */
    /* loaded from: classes.dex */
    public static class e extends a {
        public e(Class<?> cls) {
            super("Failed to find source encoder for data class: " + cls);
        }
    }

    public g() {
        a.c cVar = new a.c(new b.j.i.f(20), new c.c.a.s.k.b(), new c.c.a.s.k.c());
        this.j = cVar;
        this.f3440a = new p(cVar);
        this.f3441b = new c.c.a.p.a();
        this.f3442c = new c.c.a.p.e();
        this.f3443d = new c.c.a.p.f();
        this.f3444e = new c.c.a.m.u.f();
        this.f3445f = new c.c.a.m.x.h.f();
        this.f3446g = new c.c.a.p.b();
        List<String> asList = Arrays.asList("Gif", "Bitmap", "BitmapDrawable");
        ArrayList arrayList = new ArrayList(asList.size());
        arrayList.add("legacy_prepend_all");
        for (String str : asList) {
            arrayList.add(str);
        }
        arrayList.add("legacy_append");
        c.c.a.p.e eVar = this.f3442c;
        synchronized (eVar) {
            ArrayList arrayList2 = new ArrayList(eVar.f4118a);
            eVar.f4118a.clear();
            Iterator it = arrayList.iterator();
            while (it.hasNext()) {
                eVar.f4118a.add((String) it.next());
            }
            Iterator it2 = arrayList2.iterator();
            while (it2.hasNext()) {
                String str2 = (String) it2.next();
                if (!arrayList.contains(str2)) {
                    eVar.f4118a.add(str2);
                }
            }
        }
    }

    public <Data> g a(Class<Data> cls, c.c.a.m.d<Data> dVar) {
        c.c.a.p.a aVar = this.f3441b;
        synchronized (aVar) {
            aVar.f4109a.add(new a.C0083a<>(cls, dVar));
        }
        return this;
    }

    public <TResource> g b(Class<TResource> cls, s<TResource> sVar) {
        c.c.a.p.f fVar = this.f3443d;
        synchronized (fVar) {
            fVar.f4123a.add(new f.a<>(cls, sVar));
        }
        return this;
    }

    public <Model, Data> g c(Class<Model> cls, Class<Data> cls2, o<Model, Data> oVar) {
        p pVar = this.f3440a;
        synchronized (pVar) {
            r rVar = pVar.f3866a;
            synchronized (rVar) {
                r.b<?, ?> bVar = new r.b<>(cls, cls2, oVar);
                List<r.b<?, ?>> list = rVar.f3881c;
                list.add(list.size(), bVar);
            }
            pVar.f3867b.f3868a.clear();
        }
        return this;
    }

    public <Data, TResource> g d(String str, Class<Data> cls, Class<TResource> cls2, c.c.a.m.r<Data, TResource> rVar) {
        c.c.a.p.e eVar = this.f3442c;
        synchronized (eVar) {
            eVar.a(str).add(new e.a<>(cls, cls2, rVar));
        }
        return this;
    }

    public List<ImageHeaderParser> e() {
        List<ImageHeaderParser> list;
        c.c.a.p.b bVar = this.f3446g;
        synchronized (bVar) {
            list = bVar.f4112a;
        }
        if (list.isEmpty()) {
            throw new b();
        }
        return list;
    }

    public <Model> List<n<Model, ?>> f(Model model) {
        List<n<?, ?>> list;
        p pVar = this.f3440a;
        Objects.requireNonNull(pVar);
        Class<?> cls = model.getClass();
        synchronized (pVar) {
            p.a.C0076a<?> c0076a = pVar.f3867b.f3868a.get(cls);
            list = c0076a == null ? null : c0076a.f3869a;
            if (list == null) {
                list = Collections.unmodifiableList(pVar.f3866a.c(cls));
                if (pVar.f3867b.f3868a.put(cls, new p.a.C0076a<>(list)) != null) {
                    throw new IllegalStateException("Already cached loaders for model: " + cls);
                }
            }
        }
        if (!list.isEmpty()) {
            int size = list.size();
            List<n<Model, ?>> emptyList = Collections.emptyList();
            boolean z = true;
            for (int i = 0; i < size; i++) {
                n<?, ?> nVar = list.get(i);
                if (nVar.a(model)) {
                    if (z) {
                        emptyList = new ArrayList<>(size - i);
                        z = false;
                    }
                    emptyList.add(nVar);
                }
            }
            if (emptyList.isEmpty()) {
                throw new c(model, (List<n<Model, ?>>) list);
            }
            return emptyList;
        }
        throw new c(model);
    }

    public g g(e.a<?> aVar) {
        c.c.a.m.u.f fVar = this.f3444e;
        synchronized (fVar) {
            fVar.f3556b.put(aVar.a(), aVar);
        }
        return this;
    }

    public <TResource, Transcode> g h(Class<TResource> cls, Class<Transcode> cls2, c.c.a.m.x.h.e<TResource, Transcode> eVar) {
        c.c.a.m.x.h.f fVar = this.f3445f;
        synchronized (fVar) {
            fVar.f4070a.add(new f.a<>(cls, cls2, eVar));
        }
        return this;
    }
}