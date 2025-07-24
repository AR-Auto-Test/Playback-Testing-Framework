package c.c.a.m.x.h;

import java.util.ArrayList;
import java.util.List;

/* compiled from: TranscoderRegistry.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public final List<a<?, ?>> f4070a = new ArrayList();

    /* compiled from: TranscoderRegistry.java */
    /* loaded from: classes.dex */
    public static final class a<Z, R> {

        /* renamed from: a  reason: collision with root package name */
        public final Class<Z> f4071a;

        /* renamed from: b  reason: collision with root package name */
        public final Class<R> f4072b;

        /* renamed from: c  reason: collision with root package name */
        public final e<Z, R> f4073c;

        public a(Class<Z> cls, Class<R> cls2, e<Z, R> eVar) {
            this.f4071a = cls;
            this.f4072b = cls2;
            this.f4073c = eVar;
        }

        public boolean a(Class<?> cls, Class<?> cls2) {
            return this.f4071a.isAssignableFrom(cls) && cls2.isAssignableFrom(this.f4072b);
        }
    }

    public synchronized <Z, R> List<Class<R>> a(Class<Z> cls, Class<R> cls2) {
        ArrayList arrayList = new ArrayList();
        if (cls2.isAssignableFrom(cls)) {
            arrayList.add(cls2);
            return arrayList;
        }
        for (a<?, ?> aVar : this.f4070a) {
            if (aVar.a(cls, cls2)) {
                arrayList.add(cls2);
            }
        }
        return arrayList;
    }
}