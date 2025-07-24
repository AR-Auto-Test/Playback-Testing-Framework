package c.a.a.x.c;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: BaseKeyframeAnimation.java */
/* loaded from: classes.dex */
public abstract class a<K, A> {

    /* renamed from: c  reason: collision with root package name */
    public final d<K> f3225c;

    /* renamed from: e  reason: collision with root package name */
    public c.a.a.d0.c<A> f3227e;

    /* renamed from: a  reason: collision with root package name */
    public final List<b> f3223a = new ArrayList(1);

    /* renamed from: b  reason: collision with root package name */
    public boolean f3224b = false;

    /* renamed from: d  reason: collision with root package name */
    public float f3226d = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

    /* renamed from: f  reason: collision with root package name */
    public A f3228f = null;

    /* renamed from: g  reason: collision with root package name */
    public float f3229g = -1.0f;

    /* renamed from: h  reason: collision with root package name */
    public float f3230h = -1.0f;

    /* compiled from: BaseKeyframeAnimation.java */
    /* loaded from: classes.dex */
    public interface b {
        void a();
    }

    /* compiled from: BaseKeyframeAnimation.java */
    /* loaded from: classes.dex */
    public static final class c<T> implements d<T> {
        public c(C0061a c0061a) {
        }

        @Override // c.a.a.x.c.a.d
        public boolean a(float f2) {
            throw new IllegalStateException("not implemented");
        }

        @Override // c.a.a.x.c.a.d
        public c.a.a.d0.a<T> b() {
            throw new IllegalStateException("not implemented");
        }

        @Override // c.a.a.x.c.a.d
        public boolean c(float f2) {
            return false;
        }

        @Override // c.a.a.x.c.a.d
        public float d() {
            return 1.0f;
        }

        @Override // c.a.a.x.c.a.d
        public float e() {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // c.a.a.x.c.a.d
        public boolean isEmpty() {
            return true;
        }
    }

    /* compiled from: BaseKeyframeAnimation.java */
    /* loaded from: classes.dex */
    public interface d<T> {
        boolean a(float f2);

        c.a.a.d0.a<T> b();

        boolean c(float f2);

        float d();

        float e();

        boolean isEmpty();
    }

    /* compiled from: BaseKeyframeAnimation.java */
    /* loaded from: classes.dex */
    public static final class e<T> implements d<T> {

        /* renamed from: a  reason: collision with root package name */
        public final List<? extends c.a.a.d0.a<T>> f3231a;

        /* renamed from: c  reason: collision with root package name */
        public c.a.a.d0.a<T> f3233c = null;

        /* renamed from: d  reason: collision with root package name */
        public float f3234d = -1.0f;

        /* renamed from: b  reason: collision with root package name */
        public c.a.a.d0.a<T> f3232b = f(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);

        public e(List<? extends c.a.a.d0.a<T>> list) {
            this.f3231a = list;
        }

        @Override // c.a.a.x.c.a.d
        public boolean a(float f2) {
            c.a.a.d0.a<T> aVar = this.f3233c;
            c.a.a.d0.a<T> aVar2 = this.f3232b;
            if (aVar == aVar2 && this.f3234d == f2) {
                return true;
            }
            this.f3233c = aVar2;
            this.f3234d = f2;
            return false;
        }

        @Override // c.a.a.x.c.a.d
        public c.a.a.d0.a<T> b() {
            return this.f3232b;
        }

        @Override // c.a.a.x.c.a.d
        public boolean c(float f2) {
            if (this.f3232b.a(f2)) {
                return !this.f3232b.d();
            }
            this.f3232b = f(f2);
            return true;
        }

        @Override // c.a.a.x.c.a.d
        public float d() {
            List<? extends c.a.a.d0.a<T>> list = this.f3231a;
            return list.get(list.size() - 1).b();
        }

        @Override // c.a.a.x.c.a.d
        public float e() {
            return this.f3231a.get(0).c();
        }

        public final c.a.a.d0.a<T> f(float f2) {
            List<? extends c.a.a.d0.a<T>> list = this.f3231a;
            c.a.a.d0.a<T> aVar = list.get(list.size() - 1);
            if (f2 >= aVar.c()) {
                return aVar;
            }
            for (int size = this.f3231a.size() - 2; size >= 1; size--) {
                c.a.a.d0.a<T> aVar2 = this.f3231a.get(size);
                if (this.f3232b != aVar2 && aVar2.a(f2)) {
                    return aVar2;
                }
            }
            return this.f3231a.get(0);
        }

        @Override // c.a.a.x.c.a.d
        public boolean isEmpty() {
            return false;
        }
    }

    /* compiled from: BaseKeyframeAnimation.java */
    /* loaded from: classes.dex */
    public static final class f<T> implements d<T> {

        /* renamed from: a  reason: collision with root package name */
        public final c.a.a.d0.a<T> f3235a;

        /* renamed from: b  reason: collision with root package name */
        public float f3236b = -1.0f;

        public f(List<? extends c.a.a.d0.a<T>> list) {
            this.f3235a = list.get(0);
        }

        @Override // c.a.a.x.c.a.d
        public boolean a(float f2) {
            if (this.f3236b == f2) {
                return true;
            }
            this.f3236b = f2;
            return false;
        }

        @Override // c.a.a.x.c.a.d
        public c.a.a.d0.a<T> b() {
            return this.f3235a;
        }

        @Override // c.a.a.x.c.a.d
        public boolean c(float f2) {
            return !this.f3235a.d();
        }

        @Override // c.a.a.x.c.a.d
        public float d() {
            return this.f3235a.b();
        }

        @Override // c.a.a.x.c.a.d
        public float e() {
            return this.f3235a.c();
        }

        @Override // c.a.a.x.c.a.d
        public boolean isEmpty() {
            return false;
        }
    }

    public a(List<? extends c.a.a.d0.a<K>> list) {
        d eVar;
        d dVar;
        if (list.isEmpty()) {
            dVar = new c(null);
        } else {
            if (list.size() == 1) {
                eVar = new f(list);
            } else {
                eVar = new e(list);
            }
            dVar = eVar;
        }
        this.f3225c = dVar;
    }

    public c.a.a.d0.a<K> a() {
        c.a.a.d0.a<K> b2 = this.f3225c.b();
        c.a.a.c.a("BaseKeyframeAnimation#getCurrentKeyframe");
        return b2;
    }

    public float b() {
        if (this.f3230h == -1.0f) {
            this.f3230h = this.f3225c.d();
        }
        return this.f3230h;
    }

    public float c() {
        c.a.a.d0.a<K> a2 = a();
        return a2.d() ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : a2.f3048d.getInterpolation(d());
    }

    public float d() {
        if (this.f3224b) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        c.a.a.d0.a<K> a2 = a();
        return a2.d() ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : (this.f3226d - a2.c()) / (a2.b() - a2.c());
    }

    public A e() {
        float c2 = c();
        if (this.f3227e == null && this.f3225c.a(c2)) {
            return this.f3228f;
        }
        A f2 = f(a(), c2);
        this.f3228f = f2;
        return f2;
    }

    public abstract A f(c.a.a.d0.a<K> aVar, float f2);

    public void g() {
        for (int i = 0; i < this.f3223a.size(); i++) {
            this.f3223a.get(i).a();
        }
    }

    public void h(float f2) {
        if (this.f3225c.isEmpty()) {
            return;
        }
        if (this.f3229g == -1.0f) {
            this.f3229g = this.f3225c.e();
        }
        float f3 = this.f3229g;
        if (f2 < f3) {
            if (f3 == -1.0f) {
                this.f3229g = this.f3225c.e();
            }
            f2 = this.f3229g;
        } else if (f2 > b()) {
            f2 = b();
        }
        if (f2 == this.f3226d) {
            return;
        }
        this.f3226d = f2;
        if (this.f3225c.c(f2)) {
            g();
        }
    }

    public void i(c.a.a.d0.c<A> cVar) {
        c.a.a.d0.c<A> cVar2 = this.f3227e;
        this.f3227e = null;
    }
}