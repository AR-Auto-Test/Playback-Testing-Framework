package b.d.b;

import android.annotation.SuppressLint;
import android.graphics.Rect;
import android.util.Size;
import b.d.b.d1.i0;
import b.d.b.d1.i1;
import b.d.b.d1.j1;
import b.d.b.w0;
import java.util.HashSet;
import java.util.Set;

/* compiled from: UseCase.java */
/* loaded from: classes.dex */
public abstract class a1 {

    /* renamed from: d  reason: collision with root package name */
    public i1<?> f1382d;

    /* renamed from: e  reason: collision with root package name */
    public i1<?> f1383e;

    /* renamed from: f  reason: collision with root package name */
    public i1<?> f1384f;

    /* renamed from: g  reason: collision with root package name */
    public Size f1385g;

    /* renamed from: h  reason: collision with root package name */
    public i1<?> f1386h;
    public Rect i;
    public b.d.b.d1.a0 j;

    /* renamed from: a  reason: collision with root package name */
    public final Set<b> f1379a = new HashSet();

    /* renamed from: b  reason: collision with root package name */
    public final Object f1380b = new Object();

    /* renamed from: c  reason: collision with root package name */
    public int f1381c = 2;
    public b.d.b.d1.b1 k = b.d.b.d1.b1.a();

    /* compiled from: UseCase.java */
    /* loaded from: classes.dex */
    public interface a {
        void a();

        void b(i0 i0Var);
    }

    /* compiled from: UseCase.java */
    /* loaded from: classes.dex */
    public interface b {
        void c(a1 a1Var);

        void d(a1 a1Var);

        void e(a1 a1Var);

        void f(a1 a1Var);
    }

    public a1(i1<?> i1Var) {
        this.f1383e = i1Var;
        this.f1384f = i1Var;
    }

    public b.d.b.d1.a0 a() {
        b.d.b.d1.a0 a0Var;
        synchronized (this.f1380b) {
            a0Var = this.j;
        }
        return a0Var;
    }

    public String b() {
        b.d.b.d1.a0 a2 = a();
        b.j.b.d.h(a2, "No camera attached to use case: " + this);
        return a2.j().b();
    }

    public abstract i1<?> c(boolean z, j1 j1Var);

    public String d() {
        i1<?> i1Var = this.f1384f;
        StringBuilder x = c.b.a.a.a.x("<UnknownUseCase-");
        x.append(hashCode());
        x.append(">");
        return i1Var.p(x.toString());
    }

    public abstract i1.a<?, ?, ?> e(b.d.b.d1.i0 i0Var);

    public i1<?> f(b.d.b.d1.z zVar, i1<?> i1Var, i1<?> i1Var2) {
        b.d.b.d1.u0 y;
        if (i1Var2 != null) {
            y = b.d.b.d1.u0.z(i1Var2);
            y.r.remove(b.d.b.e1.e.n);
        } else {
            y = b.d.b.d1.u0.y();
        }
        for (i0.a<?> aVar : this.f1383e.e()) {
            y.A(aVar, this.f1383e.g(aVar), this.f1383e.a(aVar));
        }
        if (i1Var != null) {
            for (i0.a<?> aVar2 : i1Var.e()) {
                if (!aVar2.a().equals(b.d.b.e1.e.n.a())) {
                    y.A(aVar2, i1Var.g(aVar2), i1Var.a(aVar2));
                }
            }
        }
        if (y.b(b.d.b.d1.n0.f1576d)) {
            i0.a<Integer> aVar3 = b.d.b.d1.n0.f1574b;
            if (y.b(aVar3)) {
                y.r.remove(aVar3);
            }
        }
        return l(zVar, e(y));
    }

    public final void g() {
        for (b bVar : this.f1379a) {
            bVar.d(this);
        }
    }

    public final void h() {
        int f2 = m0.f(this.f1381c);
        if (f2 == 0) {
            for (b bVar : this.f1379a) {
                bVar.c(this);
            }
        } else if (f2 == 1) {
            for (b bVar2 : this.f1379a) {
                bVar2.f(this);
            }
        }
    }

    @SuppressLint({"WrongConstant"})
    public void i(b.d.b.d1.a0 a0Var, i1<?> i1Var, i1<?> i1Var2) {
        synchronized (this.f1380b) {
            this.j = a0Var;
            this.f1379a.add(a0Var);
        }
        this.f1382d = i1Var;
        this.f1386h = i1Var2;
        i1<?> f2 = f(a0Var.j(), this.f1382d, this.f1386h);
        this.f1384f = f2;
        a u = f2.u(null);
        if (u != null) {
            u.b(a0Var.j());
        }
    }

    public void j(b.d.b.d1.a0 a0Var) {
        k();
        a u = this.f1384f.u(null);
        if (u != null) {
            u.a();
        }
        synchronized (this.f1380b) {
            b.j.b.d.d(a0Var == this.j);
            this.f1379a.remove(this.j);
            this.j = null;
        }
        this.f1385g = null;
        this.i = null;
        this.f1384f = this.f1383e;
        this.f1382d = null;
        this.f1386h = null;
    }

    public void k() {
    }

    public i1<?> l(b.d.b.d1.z zVar, i1.a<?, ?, ?> aVar) {
        return ((w0.b) aVar).b();
    }

    public abstract Size m(Size size);
}