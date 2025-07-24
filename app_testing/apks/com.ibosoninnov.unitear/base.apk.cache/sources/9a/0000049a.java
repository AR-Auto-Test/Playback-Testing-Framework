package b.j.j;

import android.annotation.SuppressLint;
import android.graphics.Insets;
import android.graphics.Rect;
import android.os.Build;
import android.util.Log;
import android.view.DisplayCutout;
import android.view.View;
import android.view.WindowInsets;
import b.j.j.q;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: WindowInsetsCompat.java */
/* loaded from: classes.dex */
public class w {

    /* renamed from: a  reason: collision with root package name */
    public static final w f2237a;

    /* renamed from: b  reason: collision with root package name */
    public final j f2238b;

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class c extends b {
        public c() {
        }

        public c(w wVar) {
            super(wVar);
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class d {

        /* renamed from: a  reason: collision with root package name */
        public final w f2246a;

        public d() {
            this(new w((w) null));
        }

        public final void a() {
        }

        public w b() {
            throw null;
        }

        public void c(b.j.d.b bVar) {
            throw null;
        }

        public void d(b.j.d.b bVar) {
            throw null;
        }

        public d(w wVar) {
            this.f2246a = wVar;
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class e extends j {

        /* renamed from: c  reason: collision with root package name */
        public static boolean f2247c = false;

        /* renamed from: d  reason: collision with root package name */
        public static Method f2248d;

        /* renamed from: e  reason: collision with root package name */
        public static Class<?> f2249e;

        /* renamed from: f  reason: collision with root package name */
        public static Class<?> f2250f;

        /* renamed from: g  reason: collision with root package name */
        public static Field f2251g;

        /* renamed from: h  reason: collision with root package name */
        public static Field f2252h;
        public final WindowInsets i;
        public b.j.d.b j;
        public w k;
        public b.j.d.b l;

        public e(w wVar, WindowInsets windowInsets) {
            super(wVar);
            this.j = null;
            this.i = windowInsets;
        }

        @SuppressLint({"PrivateApi"})
        public static void q() {
            try {
                f2248d = View.class.getDeclaredMethod("getViewRootImpl", new Class[0]);
                f2249e = Class.forName("android.view.ViewRootImpl");
                Class<?> cls = Class.forName("android.view.View$AttachInfo");
                f2250f = cls;
                f2251g = cls.getDeclaredField("mVisibleInsets");
                f2252h = f2249e.getDeclaredField("mAttachInfo");
                f2251g.setAccessible(true);
                f2252h.setAccessible(true);
            } catch (ReflectiveOperationException e2) {
                StringBuilder x = c.b.a.a.a.x("Failed to get visible insets. (Reflection error). ");
                x.append(e2.getMessage());
                Log.e("WindowInsetsCompat", x.toString(), e2);
            }
            f2247c = true;
        }

        @Override // b.j.j.w.j
        public void d(View view) {
            b.j.d.b p = p(view);
            if (p == null) {
                p = b.j.d.b.f2095a;
            }
            r(p);
        }

        @Override // b.j.j.w.j
        public boolean equals(Object obj) {
            if (super.equals(obj)) {
                return Objects.equals(this.l, ((e) obj).l);
            }
            return false;
        }

        @Override // b.j.j.w.j
        public final b.j.d.b i() {
            if (this.j == null) {
                this.j = b.j.d.b.a(this.i.getSystemWindowInsetLeft(), this.i.getSystemWindowInsetTop(), this.i.getSystemWindowInsetRight(), this.i.getSystemWindowInsetBottom());
            }
            return this.j;
        }

        @Override // b.j.j.w.j
        public w j(int i, int i2, int i3, int i4) {
            d aVar;
            w j = w.j(this.i);
            int i5 = Build.VERSION.SDK_INT;
            if (i5 >= 30) {
                aVar = new c(j);
            } else if (i5 >= 29) {
                aVar = new b(j);
            } else {
                aVar = new a(j);
            }
            aVar.d(w.f(i(), i, i2, i3, i4));
            aVar.c(w.f(g(), i, i2, i3, i4));
            return aVar.b();
        }

        @Override // b.j.j.w.j
        public boolean l() {
            return this.i.isRound();
        }

        @Override // b.j.j.w.j
        public void m(b.j.d.b[] bVarArr) {
        }

        @Override // b.j.j.w.j
        public void n(w wVar) {
            this.k = wVar;
        }

        public final b.j.d.b p(View view) {
            if (Build.VERSION.SDK_INT < 30) {
                if (!f2247c) {
                    q();
                }
                Method method = f2248d;
                if (method != null && f2250f != null && f2251g != null) {
                    try {
                        Object invoke = method.invoke(view, new Object[0]);
                        if (invoke == null) {
                            Log.w("WindowInsetsCompat", "Failed to get visible insets. getViewRootImpl() returned null from the provided view. This means that the view is either not attached or the method has been overridden", new NullPointerException());
                            return null;
                        }
                        Rect rect = (Rect) f2251g.get(f2252h.get(invoke));
                        if (rect != null) {
                            return b.j.d.b.a(rect.left, rect.top, rect.right, rect.bottom);
                        }
                        return null;
                    } catch (ReflectiveOperationException e2) {
                        StringBuilder x = c.b.a.a.a.x("Failed to get visible insets. (Reflection error). ");
                        x.append(e2.getMessage());
                        Log.e("WindowInsetsCompat", x.toString(), e2);
                    }
                }
                return null;
            }
            throw new UnsupportedOperationException("getVisibleInsets() should not be called on API >= 30. Use WindowInsets.isVisible() instead.");
        }

        public void r(b.j.d.b bVar) {
            this.l = bVar;
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class f extends e {
        public b.j.d.b m;

        public f(w wVar, WindowInsets windowInsets) {
            super(wVar, windowInsets);
            this.m = null;
        }

        @Override // b.j.j.w.j
        public w b() {
            return w.j(this.i.consumeStableInsets());
        }

        @Override // b.j.j.w.j
        public w c() {
            return w.j(this.i.consumeSystemWindowInsets());
        }

        @Override // b.j.j.w.j
        public final b.j.d.b g() {
            if (this.m == null) {
                this.m = b.j.d.b.a(this.i.getStableInsetLeft(), this.i.getStableInsetTop(), this.i.getStableInsetRight(), this.i.getStableInsetBottom());
            }
            return this.m;
        }

        @Override // b.j.j.w.j
        public boolean k() {
            return this.i.isConsumed();
        }

        @Override // b.j.j.w.j
        public void o(b.j.d.b bVar) {
            this.m = bVar;
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class g extends f {
        public g(w wVar, WindowInsets windowInsets) {
            super(wVar, windowInsets);
        }

        @Override // b.j.j.w.j
        public w a() {
            return w.j(this.i.consumeDisplayCutout());
        }

        @Override // b.j.j.w.j
        public b.j.j.c e() {
            DisplayCutout displayCutout = this.i.getDisplayCutout();
            if (displayCutout == null) {
                return null;
            }
            return new b.j.j.c(displayCutout);
        }

        @Override // b.j.j.w.e, b.j.j.w.j
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj instanceof g) {
                g gVar = (g) obj;
                return Objects.equals(this.i, gVar.i) && Objects.equals(this.l, gVar.l);
            }
            return false;
        }

        @Override // b.j.j.w.j
        public int hashCode() {
            return this.i.hashCode();
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class h extends g {
        public b.j.d.b n;
        public b.j.d.b o;

        public h(w wVar, WindowInsets windowInsets) {
            super(wVar, windowInsets);
            this.n = null;
            this.o = null;
        }

        @Override // b.j.j.w.j
        public b.j.d.b f() {
            if (this.o == null) {
                Insets mandatorySystemGestureInsets = this.i.getMandatorySystemGestureInsets();
                this.o = b.j.d.b.a(mandatorySystemGestureInsets.left, mandatorySystemGestureInsets.top, mandatorySystemGestureInsets.right, mandatorySystemGestureInsets.bottom);
            }
            return this.o;
        }

        @Override // b.j.j.w.j
        public b.j.d.b h() {
            if (this.n == null) {
                Insets systemGestureInsets = this.i.getSystemGestureInsets();
                this.n = b.j.d.b.a(systemGestureInsets.left, systemGestureInsets.top, systemGestureInsets.right, systemGestureInsets.bottom);
            }
            return this.n;
        }

        @Override // b.j.j.w.e, b.j.j.w.j
        public w j(int i, int i2, int i3, int i4) {
            return w.j(this.i.inset(i, i2, i3, i4));
        }

        @Override // b.j.j.w.f, b.j.j.w.j
        public void o(b.j.d.b bVar) {
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class i extends h {
        public static final w p = w.j(WindowInsets.CONSUMED);

        public i(w wVar, WindowInsets windowInsets) {
            super(wVar, windowInsets);
        }

        @Override // b.j.j.w.e, b.j.j.w.j
        public final void d(View view) {
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class j {

        /* renamed from: a  reason: collision with root package name */
        public static final w f2253a;

        /* renamed from: b  reason: collision with root package name */
        public final w f2254b;

        static {
            d aVar;
            int i = Build.VERSION.SDK_INT;
            if (i >= 30) {
                aVar = new c();
            } else if (i >= 29) {
                aVar = new b();
            } else {
                aVar = new a();
            }
            f2253a = aVar.b().f2238b.a().f2238b.b().a();
        }

        public j(w wVar) {
            this.f2254b = wVar;
        }

        public w a() {
            return this.f2254b;
        }

        public w b() {
            return this.f2254b;
        }

        public w c() {
            return this.f2254b;
        }

        public void d(View view) {
        }

        public b.j.j.c e() {
            return null;
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj instanceof j) {
                j jVar = (j) obj;
                return l() == jVar.l() && k() == jVar.k() && Objects.equals(i(), jVar.i()) && Objects.equals(g(), jVar.g()) && Objects.equals(e(), jVar.e());
            }
            return false;
        }

        public b.j.d.b f() {
            return i();
        }

        public b.j.d.b g() {
            return b.j.d.b.f2095a;
        }

        public b.j.d.b h() {
            return i();
        }

        public int hashCode() {
            return Objects.hash(Boolean.valueOf(l()), Boolean.valueOf(k()), i(), g(), e());
        }

        public b.j.d.b i() {
            return b.j.d.b.f2095a;
        }

        public w j(int i, int i2, int i3, int i4) {
            return f2253a;
        }

        public boolean k() {
            return false;
        }

        public boolean l() {
            return false;
        }

        public void m(b.j.d.b[] bVarArr) {
        }

        public void n(w wVar) {
        }

        public void o(b.j.d.b bVar) {
        }
    }

    static {
        if (Build.VERSION.SDK_INT >= 30) {
            f2237a = i.p;
        } else {
            f2237a = j.f2253a;
        }
    }

    public w(WindowInsets windowInsets) {
        int i2 = Build.VERSION.SDK_INT;
        if (i2 >= 30) {
            this.f2238b = new i(this, windowInsets);
        } else if (i2 >= 29) {
            this.f2238b = new h(this, windowInsets);
        } else if (i2 >= 28) {
            this.f2238b = new g(this, windowInsets);
        } else {
            this.f2238b = new f(this, windowInsets);
        }
    }

    public static b.j.d.b f(b.j.d.b bVar, int i2, int i3, int i4, int i5) {
        int max = Math.max(0, bVar.f2096b - i2);
        int max2 = Math.max(0, bVar.f2097c - i3);
        int max3 = Math.max(0, bVar.f2098d - i4);
        int max4 = Math.max(0, bVar.f2099e - i5);
        return (max == i2 && max2 == i3 && max3 == i4 && max4 == i5) ? bVar : b.j.d.b.a(max, max2, max3, max4);
    }

    public static w j(WindowInsets windowInsets) {
        return k(windowInsets, null);
    }

    public static w k(WindowInsets windowInsets, View view) {
        Objects.requireNonNull(windowInsets);
        w wVar = new w(windowInsets);
        if (view != null && view.isAttachedToWindow()) {
            AtomicInteger atomicInteger = q.f2214a;
            wVar.f2238b.n(q.c.a(view));
            wVar.f2238b.d(view.getRootView());
        }
        return wVar;
    }

    @Deprecated
    public w a() {
        return this.f2238b.c();
    }

    @Deprecated
    public int b() {
        return this.f2238b.i().f2099e;
    }

    @Deprecated
    public int c() {
        return this.f2238b.i().f2096b;
    }

    @Deprecated
    public int d() {
        return this.f2238b.i().f2098d;
    }

    @Deprecated
    public int e() {
        return this.f2238b.i().f2097c;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof w) {
            return Objects.equals(this.f2238b, ((w) obj).f2238b);
        }
        return false;
    }

    public boolean g() {
        return this.f2238b.k();
    }

    @Deprecated
    public w h(int i2, int i3, int i4, int i5) {
        d aVar;
        int i6 = Build.VERSION.SDK_INT;
        if (i6 >= 30) {
            aVar = new c(this);
        } else if (i6 >= 29) {
            aVar = new b(this);
        } else {
            aVar = new a(this);
        }
        aVar.d(b.j.d.b.a(i2, i3, i4, i5));
        return aVar.b();
    }

    public int hashCode() {
        j jVar = this.f2238b;
        if (jVar == null) {
            return 0;
        }
        return jVar.hashCode();
    }

    public WindowInsets i() {
        j jVar = this.f2238b;
        if (jVar instanceof e) {
            return ((e) jVar).i;
        }
        return null;
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class a extends d {

        /* renamed from: b  reason: collision with root package name */
        public static Field f2239b = null;

        /* renamed from: c  reason: collision with root package name */
        public static boolean f2240c = false;

        /* renamed from: d  reason: collision with root package name */
        public static Constructor<WindowInsets> f2241d = null;

        /* renamed from: e  reason: collision with root package name */
        public static boolean f2242e = false;

        /* renamed from: f  reason: collision with root package name */
        public WindowInsets f2243f;

        /* renamed from: g  reason: collision with root package name */
        public b.j.d.b f2244g;

        public a() {
            this.f2243f = e();
        }

        public static WindowInsets e() {
            if (!f2240c) {
                try {
                    f2239b = WindowInsets.class.getDeclaredField("CONSUMED");
                } catch (ReflectiveOperationException e2) {
                    Log.i("WindowInsetsCompat", "Could not retrieve WindowInsets.CONSUMED field", e2);
                }
                f2240c = true;
            }
            Field field = f2239b;
            if (field != null) {
                try {
                    WindowInsets windowInsets = (WindowInsets) field.get(null);
                    if (windowInsets != null) {
                        return new WindowInsets(windowInsets);
                    }
                } catch (ReflectiveOperationException e3) {
                    Log.i("WindowInsetsCompat", "Could not get value from WindowInsets.CONSUMED field", e3);
                }
            }
            if (!f2242e) {
                try {
                    f2241d = WindowInsets.class.getConstructor(Rect.class);
                } catch (ReflectiveOperationException e4) {
                    Log.i("WindowInsetsCompat", "Could not retrieve WindowInsets(Rect) constructor", e4);
                }
                f2242e = true;
            }
            Constructor<WindowInsets> constructor = f2241d;
            if (constructor != null) {
                try {
                    return constructor.newInstance(new Rect());
                } catch (ReflectiveOperationException e5) {
                    Log.i("WindowInsetsCompat", "Could not invoke WindowInsets(Rect) constructor", e5);
                }
            }
            return null;
        }

        @Override // b.j.j.w.d
        public w b() {
            a();
            w j = w.j(this.f2243f);
            j.f2238b.m(null);
            j.f2238b.o(this.f2244g);
            return j;
        }

        @Override // b.j.j.w.d
        public void c(b.j.d.b bVar) {
            this.f2244g = bVar;
        }

        @Override // b.j.j.w.d
        public void d(b.j.d.b bVar) {
            WindowInsets windowInsets = this.f2243f;
            if (windowInsets != null) {
                this.f2243f = windowInsets.replaceSystemWindowInsets(bVar.f2096b, bVar.f2097c, bVar.f2098d, bVar.f2099e);
            }
        }

        public a(w wVar) {
            this.f2243f = wVar.i();
        }
    }

    /* compiled from: WindowInsetsCompat.java */
    /* loaded from: classes.dex */
    public static class b extends d {

        /* renamed from: b  reason: collision with root package name */
        public final WindowInsets.Builder f2245b;

        public b() {
            this.f2245b = new WindowInsets.Builder();
        }

        @Override // b.j.j.w.d
        public w b() {
            a();
            w j = w.j(this.f2245b.build());
            j.f2238b.m(null);
            return j;
        }

        @Override // b.j.j.w.d
        public void c(b.j.d.b bVar) {
            this.f2245b.setStableInsets(bVar.b());
        }

        @Override // b.j.j.w.d
        public void d(b.j.d.b bVar) {
            this.f2245b.setSystemWindowInsets(bVar.b());
        }

        public b(w wVar) {
            WindowInsets.Builder builder;
            WindowInsets i = wVar.i();
            if (i != null) {
                builder = new WindowInsets.Builder(i);
            } else {
                builder = new WindowInsets.Builder();
            }
            this.f2245b = builder;
        }
    }

    public w(w wVar) {
        this.f2238b = new j(this);
    }
}