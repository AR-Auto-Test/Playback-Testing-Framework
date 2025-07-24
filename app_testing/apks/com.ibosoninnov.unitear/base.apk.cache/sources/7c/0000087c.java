package c.c.a.p;

import c.c.a.m.s;
import java.util.ArrayList;
import java.util.List;

/* compiled from: ResourceEncoderRegistry.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public final List<a<?>> f4123a = new ArrayList();

    /* compiled from: ResourceEncoderRegistry.java */
    /* loaded from: classes.dex */
    public static final class a<T> {

        /* renamed from: a  reason: collision with root package name */
        public final Class<T> f4124a;

        /* renamed from: b  reason: collision with root package name */
        public final s<T> f4125b;

        public a(Class<T> cls, s<T> sVar) {
            this.f4124a = cls;
            this.f4125b = sVar;
        }
    }

    public synchronized <Z> s<Z> a(Class<Z> cls) {
        int size = this.f4123a.size();
        for (int i = 0; i < size; i++) {
            a<?> aVar = this.f4123a.get(i);
            if (aVar.f4124a.isAssignableFrom(cls)) {
                return (s<Z>) aVar.f4125b;
            }
        }
        return null;
    }
}