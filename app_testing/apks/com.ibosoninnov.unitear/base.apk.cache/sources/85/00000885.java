package c.c.a.q;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.os.SystemClock;
import android.util.Log;
import c.c.a.c;
import c.c.a.m.v.l;
import c.c.a.m.v.r;
import c.c.a.m.v.w;
import c.c.a.s.j;
import c.c.a.s.k.d;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: SingleRequest.java */
/* loaded from: classes.dex */
public final class h<R> implements c, c.c.a.q.j.g, g {

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f4139a = Log.isLoggable("Request", 2);
    public int A;
    public int B;
    public boolean C;
    public RuntimeException D;

    /* renamed from: b  reason: collision with root package name */
    public final String f4140b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.s.k.d f4141c;

    /* renamed from: d  reason: collision with root package name */
    public final Object f4142d;

    /* renamed from: e  reason: collision with root package name */
    public final e<R> f4143e;

    /* renamed from: f  reason: collision with root package name */
    public final d f4144f;

    /* renamed from: g  reason: collision with root package name */
    public final Context f4145g;

    /* renamed from: h  reason: collision with root package name */
    public final c.c.a.d f4146h;
    public final Object i;
    public final Class<R> j;
    public final a<?> k;
    public final int l;
    public final int m;
    public final c.c.a.f n;
    public final c.c.a.q.j.h<R> o;
    public final List<e<R>> p;
    public final c.c.a.q.k.c<? super R> q;
    public final Executor r;
    public w<R> s;
    public l.d t;
    public long u;
    public volatile l v;
    public int w;
    public Drawable x;
    public Drawable y;
    public Drawable z;

    public h(Context context, c.c.a.d dVar, Object obj, Object obj2, Class<R> cls, a<?> aVar, int i, int i2, c.c.a.f fVar, c.c.a.q.j.h<R> hVar, e<R> eVar, List<e<R>> list, d dVar2, l lVar, c.c.a.q.k.c<? super R> cVar, Executor executor) {
        this.f4140b = f4139a ? String.valueOf(hashCode()) : null;
        this.f4141c = new d.b();
        this.f4142d = obj;
        this.f4145g = context;
        this.f4146h = dVar;
        this.i = obj2;
        this.j = cls;
        this.k = aVar;
        this.l = i;
        this.m = i2;
        this.n = fVar;
        this.o = hVar;
        this.f4143e = eVar;
        this.p = list;
        this.f4144f = dVar2;
        this.v = lVar;
        this.q = cVar;
        this.r = executor;
        this.w = 1;
        if (this.D == null && dVar.i.f3433a.containsKey(c.C0062c.class)) {
            this.D = new RuntimeException("Glide request origin trace");
        }
    }

    @Override // c.c.a.q.c
    public boolean a() {
        boolean z;
        synchronized (this.f4142d) {
            z = this.w == 4;
        }
        return z;
    }

    @Override // c.c.a.q.j.g
    public void b(int i, int i2) {
        Object obj;
        int i3 = i;
        this.f4141c.a();
        Object obj2 = this.f4142d;
        synchronized (obj2) {
            try {
                boolean z = f4139a;
                if (z) {
                    m("Got onSizeReady in " + c.c.a.s.f.a(this.u));
                }
                if (this.w == 3) {
                    this.w = 2;
                    float f2 = this.k.f4127c;
                    if (i3 != Integer.MIN_VALUE) {
                        i3 = Math.round(i3 * f2);
                    }
                    this.A = i3;
                    this.B = i2 == Integer.MIN_VALUE ? i2 : Math.round(f2 * i2);
                    if (z) {
                        m("finished setup for calling load in " + c.c.a.s.f.a(this.u));
                    }
                    l lVar = this.v;
                    c.c.a.d dVar = this.f4146h;
                    Object obj3 = this.i;
                    a<?> aVar = this.k;
                    try {
                        obj = obj2;
                        try {
                        } catch (Throwable th) {
                            th = th;
                        }
                    } catch (Throwable th2) {
                        th = th2;
                        obj = obj2;
                    }
                    try {
                        this.t = lVar.b(dVar, obj3, aVar.m, this.A, this.B, aVar.t, this.j, this.n, aVar.f4128d, aVar.s, aVar.n, aVar.z, aVar.r, aVar.j, aVar.x, aVar.A, aVar.y, this, this.r);
                        if (this.w != 2) {
                            this.t = null;
                        }
                        if (z) {
                            m("finished onSizeReady in " + c.c.a.s.f.a(this.u));
                        }
                    } catch (Throwable th3) {
                        th = th3;
                        while (true) {
                            try {
                                break;
                            } catch (Throwable th4) {
                                th = th4;
                            }
                        }
                        throw th;
                    }
                }
            } catch (Throwable th5) {
                th = th5;
                obj = obj2;
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean c(c cVar) {
        int i;
        int i2;
        Object obj;
        Class<R> cls;
        a<?> aVar;
        c.c.a.f fVar;
        int size;
        int i3;
        int i4;
        Object obj2;
        Class<R> cls2;
        a<?> aVar2;
        c.c.a.f fVar2;
        int size2;
        boolean equals;
        if (cVar instanceof h) {
            synchronized (this.f4142d) {
                i = this.l;
                i2 = this.m;
                obj = this.i;
                cls = this.j;
                aVar = this.k;
                fVar = this.n;
                List<e<R>> list = this.p;
                size = list != null ? list.size() : 0;
            }
            h hVar = (h) cVar;
            synchronized (hVar.f4142d) {
                i3 = hVar.l;
                i4 = hVar.m;
                obj2 = hVar.i;
                cls2 = hVar.j;
                aVar2 = hVar.k;
                fVar2 = hVar.n;
                List<e<R>> list2 = hVar.p;
                size2 = list2 != null ? list2.size() : 0;
            }
            if (i == i3 && i2 == i4) {
                char[] cArr = j.f4197a;
                if (obj == null) {
                    equals = obj2 == null;
                } else if (obj instanceof c.c.a.m.w.l) {
                    equals = ((c.c.a.m.w.l) obj).a(obj2);
                } else {
                    equals = obj.equals(obj2);
                }
                if (equals && cls.equals(cls2) && aVar.equals(aVar2) && fVar == fVar2 && size == size2) {
                    return true;
                }
            }
            return false;
        }
        return false;
    }

    /* JADX WARN: Removed duplicated region for block: B:20:0x002e A[Catch: all -> 0x0042, TryCatch #0 {, blocks: (B:4:0x0003, B:6:0x0010, B:8:0x0012, B:10:0x001a, B:12:0x001e, B:14:0x0022, B:20:0x002e, B:21:0x0037, B:22:0x0039), top: B:29:0x0003 }] */
    /* JADX WARN: Removed duplicated region for block: B:24:0x003c  */
    /* JADX WARN: Removed duplicated region for block: B:30:? A[RETURN, SYNTHETIC] */
    @Override // c.c.a.q.c
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void clear() {
        boolean z;
        synchronized (this.f4142d) {
            e();
            this.f4141c.a();
            if (this.w == 6) {
                return;
            }
            f();
            w<R> wVar = this.s;
            if (wVar != null) {
                this.s = null;
            } else {
                wVar = null;
            }
            d dVar = this.f4144f;
            if (dVar != null && !dVar.j(this)) {
                z = false;
                if (z) {
                    this.o.g(j());
                }
                this.w = 6;
                if (wVar == null) {
                    this.v.f(wVar);
                    return;
                }
                return;
            }
            z = true;
            if (z) {
            }
            this.w = 6;
            if (wVar == null) {
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean d() {
        boolean z;
        synchronized (this.f4142d) {
            z = this.w == 6;
        }
        return z;
    }

    public final void e() {
        if (this.C) {
            throw new IllegalStateException("You can't start or clear loads in RequestListener or Target callbacks. If you're trying to start a fallback request when a load fails, use RequestBuilder#error(RequestBuilder). Otherwise consider posting your into() or clear() calls to the main thread using a Handler instead.");
        }
    }

    public final void f() {
        e();
        this.f4141c.a();
        this.o.a(this);
        l.d dVar = this.t;
        if (dVar != null) {
            synchronized (l.this) {
                dVar.f3756a.h(dVar.f3757b);
            }
            this.t = null;
        }
    }

    @Override // c.c.a.q.c
    public void g() {
        synchronized (this.f4142d) {
            e();
            this.f4141c.a();
            int i = c.c.a.s.f.f4187b;
            this.u = SystemClock.elapsedRealtimeNanos();
            if (this.i == null) {
                if (j.j(this.l, this.m)) {
                    this.A = this.l;
                    this.B = this.m;
                }
                n(new r("Received null model"), h() == null ? 5 : 3);
                return;
            }
            int i2 = this.w;
            if (i2 == 2) {
                throw new IllegalArgumentException("Cannot restart a running request");
            }
            boolean z = false;
            if (i2 == 4) {
                o(this.s, c.c.a.m.a.MEMORY_CACHE, false);
                return;
            }
            this.w = 3;
            if (j.j(this.l, this.m)) {
                b(this.l, this.m);
            } else {
                this.o.h(this);
            }
            int i3 = this.w;
            if (i3 == 2 || i3 == 3) {
                d dVar = this.f4144f;
                if ((dVar == null || dVar.e(this)) ? true : true) {
                    this.o.e(j());
                }
            }
            if (f4139a) {
                m("finished run method in " + c.c.a.s.f.a(this.u));
            }
        }
    }

    public final Drawable h() {
        int i;
        if (this.z == null) {
            a<?> aVar = this.k;
            Drawable drawable = aVar.p;
            this.z = drawable;
            if (drawable == null && (i = aVar.q) > 0) {
                this.z = l(i);
            }
        }
        return this.z;
    }

    @Override // c.c.a.q.c
    public boolean i() {
        boolean z;
        synchronized (this.f4142d) {
            z = this.w == 4;
        }
        return z;
    }

    @Override // c.c.a.q.c
    public boolean isRunning() {
        boolean z;
        synchronized (this.f4142d) {
            int i = this.w;
            z = i == 2 || i == 3;
        }
        return z;
    }

    public final Drawable j() {
        int i;
        if (this.y == null) {
            a<?> aVar = this.k;
            Drawable drawable = aVar.f4132h;
            this.y = drawable;
            if (drawable == null && (i = aVar.i) > 0) {
                this.y = l(i);
            }
        }
        return this.y;
    }

    public final boolean k() {
        d dVar = this.f4144f;
        return dVar == null || !dVar.getRoot().a();
    }

    public final Drawable l(int i) {
        Resources.Theme theme = this.k.v;
        if (theme == null) {
            theme = this.f4145g.getTheme();
        }
        c.c.a.d dVar = this.f4146h;
        return c.c.a.m.x.e.a.a(dVar, dVar, i, theme);
    }

    public final void m(String str) {
        StringBuilder A = c.b.a.a.a.A(str, " this: ");
        A.append(this.f4140b);
        Log.v("Request", A.toString());
    }

    public final void n(r rVar, int i) {
        boolean z;
        this.f4141c.a();
        synchronized (this.f4142d) {
            Objects.requireNonNull(rVar);
            int i2 = this.f4146h.j;
            if (i2 <= i) {
                Log.w("Glide", "Load failed for " + this.i + " with size [" + this.A + "x" + this.B + "]", rVar);
                if (i2 <= 4) {
                    rVar.e("Glide");
                }
            }
            this.t = null;
            this.w = 5;
            boolean z2 = true;
            this.C = true;
            List<e<R>> list = this.p;
            if (list != null) {
                z = false;
                for (e<R> eVar : list) {
                    z |= eVar.a(rVar, this.i, this.o, k());
                }
            } else {
                z = false;
            }
            e<R> eVar2 = this.f4143e;
            if (eVar2 == null || !eVar2.a(rVar, this.i, this.o, k())) {
                z2 = false;
            }
            if (!(z | z2)) {
                q();
            }
            this.C = false;
            d dVar = this.f4144f;
            if (dVar != null) {
                dVar.b(this);
            }
        }
    }

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:45:0x00b9 -> B:57:0x00bc). Please submit an issue!!! */
    public void o(w<?> wVar, c.c.a.m.a aVar, boolean z) {
        h<R> hVar;
        h<R> hVar2;
        Throwable th;
        this.f4141c.a();
        w<?> wVar2 = null;
        try {
            synchronized (this.f4142d) {
                try {
                    this.t = null;
                    if (wVar == null) {
                        n(new r("Expected to receive a Resource<R> with an object of " + this.j + " inside, but instead got null."), 5);
                        return;
                    }
                    Object obj = wVar.get();
                    try {
                        if (obj != null && this.j.isAssignableFrom(obj.getClass())) {
                            d dVar = this.f4144f;
                            if (!(dVar == null || dVar.f(this))) {
                                this.s = null;
                                this.w = 4;
                                this.v.f(wVar);
                            }
                            p(wVar, obj, aVar);
                            return;
                        }
                        this.s = null;
                        StringBuilder sb = new StringBuilder();
                        sb.append("Expected to receive an object of ");
                        sb.append(this.j);
                        sb.append(" but instead got ");
                        sb.append(obj != null ? obj.getClass() : "");
                        sb.append("{");
                        sb.append(obj);
                        sb.append("} inside Resource{");
                        sb.append(wVar);
                        sb.append("}.");
                        sb.append(obj != null ? "" : " To indicate failure return a null Resource object, rather than a Resource object containing null data.");
                        n(new r(sb.toString()), 5);
                        this.v.f(wVar);
                    } catch (Throwable th2) {
                        th = th2;
                        wVar2 = wVar;
                        hVar = this;
                        try {
                            try {
                            } catch (Throwable th3) {
                                hVar2 = hVar;
                                th = th3;
                                h<R> hVar3 = hVar2;
                                th = th;
                                hVar = hVar3;
                                throw th;
                            }
                            throw th;
                        } catch (Throwable th4) {
                            th = th4;
                            if (wVar2 != null) {
                                hVar.v.f(wVar2);
                            }
                            throw th;
                        }
                    }
                } catch (Throwable th5) {
                    th = th5;
                    hVar2 = this;
                    h<R> hVar32 = hVar2;
                    th = th;
                    hVar = hVar32;
                    throw th;
                }
            }
        } catch (Throwable th6) {
            th = th6;
            hVar = this;
        }
    }

    /* JADX DEBUG: Incorrect args count in method signature: (Lc/c/a/m/v/w<TR;>;TR;Lc/c/a/m/a;Z)V */
    public final void p(w wVar, Object obj, c.c.a.m.a aVar) {
        boolean z;
        boolean k = k();
        this.w = 4;
        this.s = wVar;
        if (this.f4146h.j <= 3) {
            StringBuilder x = c.b.a.a.a.x("Finished loading ");
            x.append(obj.getClass().getSimpleName());
            x.append(" from ");
            x.append(aVar);
            x.append(" for ");
            x.append(this.i);
            x.append(" with size [");
            x.append(this.A);
            x.append("x");
            x.append(this.B);
            x.append("] in ");
            x.append(c.c.a.s.f.a(this.u));
            x.append(" ms");
            Log.d("Glide", x.toString());
        }
        boolean z2 = true;
        this.C = true;
        try {
            List<e<R>> list = this.p;
            if (list != null) {
                z = false;
                for (e<R> eVar : list) {
                    z |= eVar.b(obj, this.i, this.o, aVar, k);
                }
            } else {
                z = false;
            }
            e<R> eVar2 = this.f4143e;
            if (eVar2 == null || !eVar2.b(obj, this.i, this.o, aVar, k)) {
                z2 = false;
            }
            if (!(z2 | z)) {
                Objects.requireNonNull(this.q);
                this.o.b(obj, c.c.a.q.k.a.f4165a);
            }
            this.C = false;
            d dVar = this.f4144f;
            if (dVar != null) {
                dVar.h(this);
            }
        } catch (Throwable th) {
            this.C = false;
            throw th;
        }
    }

    @Override // c.c.a.q.c
    public void pause() {
        synchronized (this.f4142d) {
            if (isRunning()) {
                clear();
            }
        }
    }

    public final void q() {
        int i;
        d dVar = this.f4144f;
        if (dVar == null || dVar.e(this)) {
            Drawable h2 = this.i == null ? h() : null;
            if (h2 == null) {
                if (this.x == null) {
                    a<?> aVar = this.k;
                    Drawable drawable = aVar.f4130f;
                    this.x = drawable;
                    if (drawable == null && (i = aVar.f4131g) > 0) {
                        this.x = l(i);
                    }
                }
                h2 = this.x;
            }
            if (h2 == null) {
                h2 = j();
            }
            this.o.d(h2);
        }
    }
}