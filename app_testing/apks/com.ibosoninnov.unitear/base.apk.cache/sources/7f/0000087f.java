package c.c.a.q;

/* compiled from: ErrorRequestCoordinator.java */
/* loaded from: classes.dex */
public final class b implements d, c {

    /* renamed from: a  reason: collision with root package name */
    public final Object f4133a;

    /* renamed from: b  reason: collision with root package name */
    public final d f4134b;

    /* renamed from: c  reason: collision with root package name */
    public volatile c f4135c;

    /* renamed from: d  reason: collision with root package name */
    public volatile c f4136d;

    /* renamed from: e  reason: collision with root package name */
    public int f4137e = 3;

    /* renamed from: f  reason: collision with root package name */
    public int f4138f = 3;

    public b(Object obj, d dVar) {
        this.f4133a = obj;
        this.f4134b = dVar;
    }

    @Override // c.c.a.q.d, c.c.a.q.c
    public boolean a() {
        boolean z;
        synchronized (this.f4133a) {
            z = this.f4135c.a() || this.f4136d.a();
        }
        return z;
    }

    @Override // c.c.a.q.d
    public void b(c cVar) {
        synchronized (this.f4133a) {
            if (!cVar.equals(this.f4136d)) {
                this.f4137e = 5;
                if (this.f4138f != 1) {
                    this.f4138f = 1;
                    this.f4136d.g();
                }
                return;
            }
            this.f4138f = 5;
            d dVar = this.f4134b;
            if (dVar != null) {
                dVar.b(this);
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean c(c cVar) {
        if (cVar instanceof b) {
            b bVar = (b) cVar;
            return this.f4135c.c(bVar.f4135c) && this.f4136d.c(bVar.f4136d);
        }
        return false;
    }

    @Override // c.c.a.q.c
    public void clear() {
        synchronized (this.f4133a) {
            this.f4137e = 3;
            this.f4135c.clear();
            if (this.f4138f != 3) {
                this.f4138f = 3;
                this.f4136d.clear();
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean d() {
        boolean z;
        synchronized (this.f4133a) {
            z = this.f4137e == 3 && this.f4138f == 3;
        }
        return z;
    }

    @Override // c.c.a.q.d
    public boolean e(c cVar) {
        boolean z;
        boolean z2;
        synchronized (this.f4133a) {
            d dVar = this.f4134b;
            z = false;
            if (dVar != null && !dVar.e(this)) {
                z2 = false;
                if (z2 && k(cVar)) {
                    z = true;
                }
            }
            z2 = true;
            if (z2) {
                z = true;
            }
        }
        return z;
    }

    @Override // c.c.a.q.d
    public boolean f(c cVar) {
        boolean z;
        boolean z2;
        synchronized (this.f4133a) {
            d dVar = this.f4134b;
            z = false;
            if (dVar != null && !dVar.f(this)) {
                z2 = false;
                if (z2 && k(cVar)) {
                    z = true;
                }
            }
            z2 = true;
            if (z2) {
                z = true;
            }
        }
        return z;
    }

    @Override // c.c.a.q.c
    public void g() {
        synchronized (this.f4133a) {
            if (this.f4137e != 1) {
                this.f4137e = 1;
                this.f4135c.g();
            }
        }
    }

    @Override // c.c.a.q.d
    public d getRoot() {
        d root;
        synchronized (this.f4133a) {
            d dVar = this.f4134b;
            root = dVar != null ? dVar.getRoot() : this;
        }
        return root;
    }

    @Override // c.c.a.q.d
    public void h(c cVar) {
        synchronized (this.f4133a) {
            if (cVar.equals(this.f4135c)) {
                this.f4137e = 4;
            } else if (cVar.equals(this.f4136d)) {
                this.f4138f = 4;
            }
            d dVar = this.f4134b;
            if (dVar != null) {
                dVar.h(this);
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean i() {
        boolean z;
        synchronized (this.f4133a) {
            z = this.f4137e == 4 || this.f4138f == 4;
        }
        return z;
    }

    @Override // c.c.a.q.c
    public boolean isRunning() {
        boolean z;
        synchronized (this.f4133a) {
            z = true;
            if (this.f4137e != 1 && this.f4138f != 1) {
                z = false;
            }
        }
        return z;
    }

    @Override // c.c.a.q.d
    public boolean j(c cVar) {
        boolean z;
        boolean z2;
        synchronized (this.f4133a) {
            d dVar = this.f4134b;
            z = false;
            if (dVar != null && !dVar.j(this)) {
                z2 = false;
                if (z2 && k(cVar)) {
                    z = true;
                }
            }
            z2 = true;
            if (z2) {
                z = true;
            }
        }
        return z;
    }

    public final boolean k(c cVar) {
        return cVar.equals(this.f4135c) || (this.f4137e == 5 && cVar.equals(this.f4136d));
    }

    @Override // c.c.a.q.c
    public void pause() {
        synchronized (this.f4133a) {
            if (this.f4137e == 1) {
                this.f4137e = 2;
                this.f4135c.pause();
            }
            if (this.f4138f == 1) {
                this.f4138f = 2;
                this.f4136d.pause();
            }
        }
    }
}