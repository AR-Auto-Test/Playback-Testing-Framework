package c.c.a.s.k;

import android.util.Log;
import b.j.i.f;
import c.c.a.s.k.d;

/* compiled from: FactoryPools.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static final e<Object> f4201a = new C0087a();

    /* compiled from: FactoryPools.java */
    /* renamed from: c.c.a.s.k.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0087a implements e<Object> {
        @Override // c.c.a.s.k.a.e
        public void a(Object obj) {
        }
    }

    /* compiled from: FactoryPools.java */
    /* loaded from: classes.dex */
    public interface b<T> {
        T a();
    }

    /* compiled from: FactoryPools.java */
    /* loaded from: classes.dex */
    public static final class c<T> implements b.j.i.d<T> {

        /* renamed from: a  reason: collision with root package name */
        public final b<T> f4202a;

        /* renamed from: b  reason: collision with root package name */
        public final e<T> f4203b;

        /* renamed from: c  reason: collision with root package name */
        public final b.j.i.d<T> f4204c;

        public c(b.j.i.d<T> dVar, b<T> bVar, e<T> eVar) {
            this.f4204c = dVar;
            this.f4202a = bVar;
            this.f4203b = eVar;
        }

        @Override // b.j.i.d
        public boolean a(T t) {
            if (t instanceof d) {
                ((d.b) ((d) t).b()).f4205a = true;
            }
            this.f4203b.a(t);
            return this.f4204c.a(t);
        }

        @Override // b.j.i.d
        public T b() {
            T b2 = this.f4204c.b();
            if (b2 == null) {
                b2 = this.f4202a.a();
                if (Log.isLoggable("FactoryPools", 2)) {
                    StringBuilder x = c.b.a.a.a.x("Created new ");
                    x.append(b2.getClass());
                    Log.v("FactoryPools", x.toString());
                }
            }
            if (b2 instanceof d) {
                ((d.b) ((d) b2).b()).f4205a = false;
            }
            return b2;
        }
    }

    /* compiled from: FactoryPools.java */
    /* loaded from: classes.dex */
    public interface d {
        c.c.a.s.k.d b();
    }

    /* compiled from: FactoryPools.java */
    /* loaded from: classes.dex */
    public interface e<T> {
        void a(T t);
    }

    public static <T extends d> b.j.i.d<T> a(int i, b<T> bVar) {
        return new c(new f(i), bVar, f4201a);
    }
}