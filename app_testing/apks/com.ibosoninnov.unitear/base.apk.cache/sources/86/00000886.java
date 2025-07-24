package c.c.a.q;

import b.d.b.m0;

/* compiled from: ThumbnailRequestCoordinator.java */
/* loaded from: classes.dex */
public class i implements d, c {

    /* renamed from: a  reason: collision with root package name */
    public final d f4147a;

    /* renamed from: b  reason: collision with root package name */
    public final Object f4148b;

    /* renamed from: c  reason: collision with root package name */
    public volatile c f4149c;

    /* renamed from: d  reason: collision with root package name */
    public volatile c f4150d;

    /* renamed from: e  reason: collision with root package name */
    public int f4151e = 3;

    /* renamed from: f  reason: collision with root package name */
    public int f4152f = 3;

    /* renamed from: g  reason: collision with root package name */
    public boolean f4153g;

    public i(Object obj, d dVar) {
        this.f4148b = obj;
        this.f4147a = dVar;
    }

    @Override // c.c.a.q.d, c.c.a.q.c
    public boolean a() {
        boolean z;
        synchronized (this.f4148b) {
            z = this.f4150d.a() || this.f4149c.a();
        }
        return z;
    }

    @Override // c.c.a.q.d
    public void b(c cVar) {
        synchronized (this.f4148b) {
            if (!cVar.equals(this.f4149c)) {
                this.f4152f = 5;
                return;
            }
            this.f4151e = 5;
            d dVar = this.f4147a;
            if (dVar != null) {
                dVar.b(this);
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean c(c cVar) {
        if (cVar instanceof i) {
            i iVar = (i) cVar;
            if (this.f4149c == null) {
                if (iVar.f4149c != null) {
                    return false;
                }
            } else if (!this.f4149c.c(iVar.f4149c)) {
                return false;
            }
            if (this.f4150d == null) {
                if (iVar.f4150d != null) {
                    return false;
                }
            } else if (!this.f4150d.c(iVar.f4150d)) {
                return false;
            }
            return true;
        }
        return false;
    }

    @Override // c.c.a.q.c
    public void clear() {
        synchronized (this.f4148b) {
            this.f4153g = false;
            this.f4151e = 3;
            this.f4152f = 3;
            this.f4150d.clear();
            this.f4149c.clear();
        }
    }

    @Override // c.c.a.q.c
    public boolean d() {
        boolean z;
        synchronized (this.f4148b) {
            z = this.f4151e == 3;
        }
        return z;
    }

    @Override // c.c.a.q.d
    public boolean e(c cVar) {
        boolean z;
        boolean z2;
        synchronized (this.f4148b) {
            d dVar = this.f4147a;
            z = false;
            if (dVar != null && !dVar.e(this)) {
                z2 = false;
                if (z2 && cVar.equals(this.f4149c) && !a()) {
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
        synchronized (this.f4148b) {
            d dVar = this.f4147a;
            z = false;
            if (dVar != null && !dVar.f(this)) {
                z2 = false;
                if (z2 && (cVar.equals(this.f4149c) || this.f4151e != 4)) {
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
        synchronized (this.f4148b) {
            this.f4153g = true;
            if (this.f4151e != 4 && this.f4152f != 1) {
                this.f4152f = 1;
                this.f4150d.g();
            }
            if (this.f4153g && this.f4151e != 1) {
                this.f4151e = 1;
                this.f4149c.g();
            }
            this.f4153g = false;
        }
    }

    @Override // c.c.a.q.d
    public d getRoot() {
        d root;
        synchronized (this.f4148b) {
            d dVar = this.f4147a;
            root = dVar != null ? dVar.getRoot() : this;
        }
        return root;
    }

    @Override // c.c.a.q.d
    public void h(c cVar) {
        synchronized (this.f4148b) {
            if (cVar.equals(this.f4150d)) {
                this.f4152f = 4;
                return;
            }
            this.f4151e = 4;
            d dVar = this.f4147a;
            if (dVar != null) {
                dVar.h(this);
            }
            if (!m0.j(this.f4152f)) {
                this.f4150d.clear();
            }
        }
    }

    @Override // c.c.a.q.c
    public boolean i() {
        boolean z;
        synchronized (this.f4148b) {
            z = this.f4151e == 4;
        }
        return z;
    }

    @Override // c.c.a.q.c
    public boolean isRunning() {
        boolean z;
        synchronized (this.f4148b) {
            z = true;
            if (this.f4151e != 1) {
                z = false;
            }
        }
        return z;
    }

    @Override // c.c.a.q.d
    public boolean j(c cVar) {
        boolean z;
        boolean z2;
        synchronized (this.f4148b) {
            d dVar = this.f4147a;
            z = false;
            if (dVar != null && !dVar.j(this)) {
                z2 = false;
                if (z2 && cVar.equals(this.f4149c) && this.f4151e != 2) {
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
    public void pause() {
        synchronized (this.f4148b) {
            if (!m0.j(this.f4152f)) {
                this.f4152f = 2;
                this.f4150d.pause();
            }
            if (!m0.j(this.f4151e)) {
                this.f4151e = 2;
                this.f4149c.pause();
            }
        }
    }
}