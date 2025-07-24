package c.c.a.p;

import c.c.a.m.r;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* compiled from: ResourceDecoderRegistry.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final List<String> f4118a = new ArrayList();

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, List<a<?, ?>>> f4119b = new HashMap();

    /* compiled from: ResourceDecoderRegistry.java */
    /* loaded from: classes.dex */
    public static class a<T, R> {

        /* renamed from: a  reason: collision with root package name */
        public final Class<T> f4120a;

        /* renamed from: b  reason: collision with root package name */
        public final Class<R> f4121b;

        /* renamed from: c  reason: collision with root package name */
        public final r<T, R> f4122c;

        public a(Class<T> cls, Class<R> cls2, r<T, R> rVar) {
            this.f4120a = cls;
            this.f4121b = cls2;
            this.f4122c = rVar;
        }

        public boolean a(Class<?> cls, Class<?> cls2) {
            return this.f4120a.isAssignableFrom(cls) && cls2.isAssignableFrom(this.f4121b);
        }
    }

    public final synchronized List<a<?, ?>> a(String str) {
        List<a<?, ?>> list;
        if (!this.f4118a.contains(str)) {
            this.f4118a.add(str);
        }
        list = this.f4119b.get(str);
        if (list == null) {
            list = new ArrayList<>();
            this.f4119b.put(str, list);
        }
        return list;
    }

    public synchronized <T, R> List<Class<R>> b(Class<T> cls, Class<R> cls2) {
        ArrayList arrayList;
        arrayList = new ArrayList();
        for (String str : this.f4118a) {
            List<a<?, ?>> list = this.f4119b.get(str);
            if (list != null) {
                for (a<?, ?> aVar : list) {
                    if (aVar.a(cls, cls2) && !arrayList.contains(aVar.f4121b)) {
                        arrayList.add(aVar.f4121b);
                    }
                }
            }
        }
        return arrayList;
    }
}