package b.o.a;

import android.os.Looper;
import android.util.AndroidRuntimeException;
import android.view.View;
import b.o.a.a;
import b.o.a.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: DynamicAnimation.java */
/* loaded from: classes.dex */
public abstract class b<T extends b<T>> implements a.b {

    /* renamed from: a  reason: collision with root package name */
    public static final r f2351a;

    /* renamed from: b  reason: collision with root package name */
    public static final r f2352b;

    /* renamed from: c  reason: collision with root package name */
    public static final r f2353c;

    /* renamed from: d  reason: collision with root package name */
    public static final r f2354d;

    /* renamed from: e  reason: collision with root package name */
    public static final r f2355e;

    /* renamed from: f  reason: collision with root package name */
    public static final r f2356f;
    public final Object j;
    public final b.o.a.c k;
    public float o;

    /* renamed from: g  reason: collision with root package name */
    public float f2357g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

    /* renamed from: h  reason: collision with root package name */
    public float f2358h = Float.MAX_VALUE;
    public boolean i = false;
    public boolean l = false;
    public float m = -3.4028235E38f;
    public long n = 0;
    public final ArrayList<p> p = new ArrayList<>();
    public final ArrayList<q> q = new ArrayList<>();

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class a extends r {
        public a(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getY();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setY(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* renamed from: b.o.a.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0045b extends r {
        public C0045b(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return view.getZ();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            view.setZ(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class c extends r {
        public c(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getAlpha();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setAlpha(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class d extends r {
        public d(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getScrollX();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setScrollX((int) f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class e extends r {
        public e(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getScrollY();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setScrollY((int) f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class f extends r {
        public f(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getTranslationX();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setTranslationX(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class g extends r {
        public g(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getTranslationY();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setTranslationY(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class h extends r {
        public h(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return view.getTranslationZ();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            view.setTranslationZ(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class i extends r {
        public i(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getScaleX();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setScaleX(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class j extends r {
        public j(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getScaleY();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setScaleY(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class k extends r {
        public k(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getRotation();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setRotation(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class l extends r {
        public l(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getRotationX();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setRotationX(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class m extends r {
        public m(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getRotationY();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setRotationY(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class n extends r {
        public n(String str) {
            super(str, null);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.o.a.c
        public float getValue(View view) {
            return view.getX();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, float] */
        @Override // b.o.a.c
        public void setValue(View view, float f2) {
            view.setX(f2);
        }
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static class o {

        /* renamed from: a  reason: collision with root package name */
        public float f2359a;

        /* renamed from: b  reason: collision with root package name */
        public float f2360b;
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public interface p {
        void a(b bVar, boolean z, float f2, float f3);
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public interface q {
        void a(b bVar, float f2, float f3);
    }

    /* compiled from: DynamicAnimation.java */
    /* loaded from: classes.dex */
    public static abstract class r extends b.o.a.c<View> {
        public r(String str, f fVar) {
            super(str);
        }
    }

    static {
        new f("translationX");
        new g("translationY");
        new h("translationZ");
        f2351a = new i("scaleX");
        f2352b = new j("scaleY");
        f2353c = new k("rotation");
        f2354d = new l("rotationX");
        f2355e = new m("rotationY");
        new n("x");
        new a("y");
        new C0045b("z");
        f2356f = new c("alpha");
        new d("scrollX");
        new e("scrollY");
    }

    public <K> b(K k2, b.o.a.c<K> cVar) {
        this.j = k2;
        this.k = cVar;
        if (cVar != f2353c && cVar != f2354d && cVar != f2355e) {
            if (cVar == f2356f) {
                this.o = 0.00390625f;
                return;
            } else if (cVar != f2351a && cVar != f2352b) {
                this.o = 1.0f;
                return;
            } else {
                this.o = 0.00390625f;
                return;
            }
        }
        this.o = 0.1f;
    }

    public static <T> void d(ArrayList<T> arrayList) {
        for (int size = arrayList.size() - 1; size >= 0; size--) {
            if (arrayList.get(size) == null) {
                arrayList.remove(size);
            }
        }
    }

    @Override // b.o.a.a.b
    public boolean a(long j2) {
        long j3 = this.n;
        if (j3 == 0) {
            this.n = j2;
            e(this.f2358h);
            return false;
        }
        long j4 = j2 - j3;
        this.n = j2;
        b.o.a.d dVar = (b.o.a.d) this;
        boolean z = true;
        if (dVar.s != Float.MAX_VALUE) {
            b.o.a.e eVar = dVar.r;
            double d2 = eVar.i;
            long j5 = j4 / 2;
            o b2 = eVar.b(dVar.f2358h, dVar.f2357g, j5);
            b.o.a.e eVar2 = dVar.r;
            eVar2.i = dVar.s;
            dVar.s = Float.MAX_VALUE;
            o b3 = eVar2.b(b2.f2359a, b2.f2360b, j5);
            dVar.f2358h = b3.f2359a;
            dVar.f2357g = b3.f2360b;
        } else {
            o b4 = dVar.r.b(dVar.f2358h, dVar.f2357g, j4);
            dVar.f2358h = b4.f2359a;
            dVar.f2357g = b4.f2360b;
        }
        float max = Math.max(dVar.f2358h, dVar.m);
        dVar.f2358h = max;
        float min = Math.min(max, Float.MAX_VALUE);
        dVar.f2358h = min;
        float f2 = dVar.f2357g;
        b.o.a.e eVar3 = dVar.r;
        Objects.requireNonNull(eVar3);
        if (((double) Math.abs(f2)) < eVar3.f2366e && ((double) Math.abs(min - ((float) eVar3.i))) < eVar3.f2365d) {
            dVar.f2358h = (float) dVar.r.i;
            dVar.f2357g = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        } else {
            z = false;
        }
        float min2 = Math.min(this.f2358h, Float.MAX_VALUE);
        this.f2358h = min2;
        float max2 = Math.max(min2, this.m);
        this.f2358h = max2;
        e(max2);
        if (z) {
            c(false);
        }
        return z;
    }

    public void b() {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            if (this.l) {
                c(true);
                return;
            }
            return;
        }
        throw new AndroidRuntimeException("Animations may only be canceled on the main thread");
    }

    public final void c(boolean z) {
        this.l = false;
        b.o.a.a a2 = b.o.a.a.a();
        a2.f2340b.remove(this);
        int indexOf = a2.f2341c.indexOf(this);
        if (indexOf >= 0) {
            a2.f2341c.set(indexOf, null);
            a2.f2345g = true;
        }
        this.n = 0L;
        this.i = false;
        for (int i2 = 0; i2 < this.p.size(); i2++) {
            if (this.p.get(i2) != null) {
                this.p.get(i2).a(this, z, this.f2358h, this.f2357g);
            }
        }
        d(this.p);
    }

    public void e(float f2) {
        this.k.setValue(this.j, f2);
        for (int i2 = 0; i2 < this.q.size(); i2++) {
            if (this.q.get(i2) != null) {
                this.q.get(i2).a(this, this.f2358h, this.f2357g);
            }
        }
        d(this.q);
    }
}