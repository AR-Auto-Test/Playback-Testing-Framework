package b.d.b;

import android.media.ImageReader;
import android.util.LongSparseArray;
import android.view.Surface;
import b.d.b.d1.o0;
import b.d.b.p0;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: MetadataImageReader.java */
/* loaded from: classes.dex */
public class v0 implements b.d.b.d1.o0, p0.a {

    /* renamed from: a  reason: collision with root package name */
    public final Object f1674a;

    /* renamed from: b  reason: collision with root package name */
    public b.d.b.d1.q f1675b;

    /* renamed from: c  reason: collision with root package name */
    public o0.a f1676c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f1677d;

    /* renamed from: e  reason: collision with root package name */
    public final b.d.b.d1.o0 f1678e;

    /* renamed from: f  reason: collision with root package name */
    public o0.a f1679f;

    /* renamed from: g  reason: collision with root package name */
    public Executor f1680g;

    /* renamed from: h  reason: collision with root package name */
    public final LongSparseArray<q0> f1681h;
    public final LongSparseArray<r0> i;
    public int j;
    public final List<r0> k;
    public final List<r0> l;

    /* compiled from: MetadataImageReader.java */
    /* loaded from: classes.dex */
    public class a extends b.d.b.d1.q {
        public a() {
        }

        @Override // b.d.b.d1.q
        public void b(b.d.b.d1.t tVar) {
            v0 v0Var = v0.this;
            synchronized (v0Var.f1674a) {
                if (v0Var.f1677d) {
                    return;
                }
                b.d.a.e.n0 n0Var = (b.d.a.e.n0) tVar;
                v0Var.f1681h.put(n0Var.a(), new b.d.b.e1.b(n0Var));
                v0Var.g();
            }
        }
    }

    public v0(int i, int i2, int i3, int i4) {
        a0 a0Var = new a0(ImageReader.newInstance(i, i2, i3, i4));
        this.f1674a = new Object();
        this.f1675b = new a();
        this.f1676c = new o0.a() { // from class: b.d.b.n
            @Override // b.d.b.d1.o0.a
            public final void a(b.d.b.d1.o0 o0Var) {
                v0 v0Var = v0.this;
                synchronized (v0Var.f1674a) {
                    if (v0Var.f1677d) {
                        return;
                    }
                    int i5 = 0;
                    do {
                        r0 r0Var = null;
                        try {
                            r0Var = o0Var.d();
                            if (r0Var != null) {
                                i5++;
                                v0Var.i.put(r0Var.n().getTimestamp(), r0Var);
                                v0Var.g();
                            }
                        } catch (IllegalStateException e2) {
                            u0.a("MetadataImageReader", "Failed to acquire next image.", e2);
                        }
                        if (r0Var == null) {
                            break;
                        }
                    } while (i5 < o0Var.c());
                }
            }
        };
        this.f1677d = false;
        this.f1681h = new LongSparseArray<>();
        this.i = new LongSparseArray<>();
        this.l = new ArrayList();
        this.f1678e = a0Var;
        this.j = 0;
        this.k = new ArrayList(c());
    }

    @Override // b.d.b.d1.o0
    public Surface a() {
        Surface a2;
        synchronized (this.f1674a) {
            a2 = this.f1678e.a();
        }
        return a2;
    }

    @Override // b.d.b.p0.a
    public void b(r0 r0Var) {
        synchronized (this.f1674a) {
            synchronized (this.f1674a) {
                int indexOf = this.k.indexOf(r0Var);
                if (indexOf >= 0) {
                    this.k.remove(indexOf);
                    int i = this.j;
                    if (indexOf <= i) {
                        this.j = i - 1;
                    }
                }
                this.l.remove(r0Var);
            }
        }
    }

    @Override // b.d.b.d1.o0
    public int c() {
        int c2;
        synchronized (this.f1674a) {
            c2 = this.f1678e.c();
        }
        return c2;
    }

    @Override // b.d.b.d1.o0
    public void close() {
        synchronized (this.f1674a) {
            if (this.f1677d) {
                return;
            }
            Iterator it = new ArrayList(this.k).iterator();
            while (it.hasNext()) {
                ((r0) it.next()).close();
            }
            this.k.clear();
            this.f1678e.close();
            this.f1677d = true;
        }
    }

    @Override // b.d.b.d1.o0
    public r0 d() {
        synchronized (this.f1674a) {
            if (this.k.isEmpty()) {
                return null;
            }
            if (this.j < this.k.size()) {
                List<r0> list = this.k;
                int i = this.j;
                this.j = i + 1;
                r0 r0Var = list.get(i);
                this.l.add(r0Var);
                return r0Var;
            }
            throw new IllegalStateException("Maximum image number reached.");
        }
    }

    @Override // b.d.b.d1.o0
    public void e(o0.a aVar, Executor executor) {
        synchronized (this.f1674a) {
            Objects.requireNonNull(aVar);
            this.f1679f = aVar;
            this.f1680g = executor;
            this.f1678e.e(this.f1676c, executor);
        }
    }

    public final void f(y0 y0Var) {
        final o0.a aVar;
        Executor executor;
        synchronized (this.f1674a) {
            aVar = null;
            if (this.k.size() < c()) {
                synchronized (y0Var) {
                    y0Var.f1663c.add(this);
                }
                this.k.add(y0Var);
                aVar = this.f1679f;
                executor = this.f1680g;
            } else {
                u0.a("TAG", "Maximum image number reached.", null);
                y0Var.close();
                executor = null;
            }
        }
        if (aVar != null) {
            if (executor != null) {
                executor.execute(new Runnable() { // from class: b.d.b.o
                    @Override // java.lang.Runnable
                    public final void run() {
                        v0 v0Var = v0.this;
                        o0.a aVar2 = aVar;
                        Objects.requireNonNull(v0Var);
                        aVar2.a(v0Var);
                    }
                });
            } else {
                aVar.a(this);
            }
        }
    }

    public final void g() {
        synchronized (this.f1674a) {
            for (int size = this.f1681h.size() - 1; size >= 0; size--) {
                q0 valueAt = this.f1681h.valueAt(size);
                long timestamp = valueAt.getTimestamp();
                r0 r0Var = this.i.get(timestamp);
                if (r0Var != null) {
                    this.i.remove(timestamp);
                    this.f1681h.removeAt(size);
                    f(new y0(r0Var, valueAt));
                }
            }
            h();
        }
    }

    public final void h() {
        synchronized (this.f1674a) {
            if (this.i.size() != 0 && this.f1681h.size() != 0) {
                Long valueOf = Long.valueOf(this.i.keyAt(0));
                Long valueOf2 = Long.valueOf(this.f1681h.keyAt(0));
                b.j.b.d.d(valueOf2.equals(valueOf) ? false : true);
                if (valueOf2.longValue() > valueOf.longValue()) {
                    for (int size = this.i.size() - 1; size >= 0; size--) {
                        if (this.i.keyAt(size) < valueOf2.longValue()) {
                            this.i.valueAt(size).close();
                            this.i.removeAt(size);
                        }
                    }
                } else {
                    for (int size2 = this.f1681h.size() - 1; size2 >= 0; size2--) {
                        if (this.f1681h.keyAt(size2) < valueOf.longValue()) {
                            this.f1681h.removeAt(size2);
                        }
                    }
                }
            }
        }
    }
}