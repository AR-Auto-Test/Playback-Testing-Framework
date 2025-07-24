package c.c.a.m.w;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* compiled from: ModelLoaderRegistry.java */
/* loaded from: classes.dex */
public class p {

    /* renamed from: a  reason: collision with root package name */
    public final r f3866a;

    /* renamed from: b  reason: collision with root package name */
    public final a f3867b;

    /* compiled from: ModelLoaderRegistry.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Map<Class<?>, C0076a<?>> f3868a = new HashMap();

        /* compiled from: ModelLoaderRegistry.java */
        /* renamed from: c.c.a.m.w.p$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public static class C0076a<Model> {

            /* renamed from: a  reason: collision with root package name */
            public final List<n<Model, ?>> f3869a;

            public C0076a(List<n<Model, ?>> list) {
                this.f3869a = list;
            }
        }
    }

    public p(b.j.i.d<List<Throwable>> dVar) {
        r rVar = new r(dVar);
        this.f3867b = new a();
        this.f3866a = rVar;
    }
}