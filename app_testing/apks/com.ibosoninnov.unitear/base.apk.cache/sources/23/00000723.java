package c.c.a.m.u;

import c.c.a.m.u.e;
import java.util.HashMap;
import java.util.Map;

/* compiled from: DataRewinderRegistry.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public static final e.a<?> f3555a = new a();

    /* renamed from: b  reason: collision with root package name */
    public final Map<Class<?>, e.a<?>> f3556b = new HashMap();

    /* compiled from: DataRewinderRegistry.java */
    /* loaded from: classes.dex */
    public class a implements e.a<Object> {
        @Override // c.c.a.m.u.e.a
        public Class<Object> a() {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override // c.c.a.m.u.e.a
        public e<Object> b(Object obj) {
            return new b(obj);
        }
    }

    /* compiled from: DataRewinderRegistry.java */
    /* loaded from: classes.dex */
    public static final class b implements e<Object> {

        /* renamed from: a  reason: collision with root package name */
        public final Object f3557a;

        public b(Object obj) {
            this.f3557a = obj;
        }

        @Override // c.c.a.m.u.e
        public Object a() {
            return this.f3557a;
        }

        @Override // c.c.a.m.u.e
        public void b() {
        }
    }
}