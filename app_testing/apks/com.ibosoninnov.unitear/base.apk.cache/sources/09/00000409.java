package b.h.b.i.l;

/* compiled from: WidgetRun.java */
/* loaded from: classes.dex */
public abstract class o implements d {

    /* renamed from: a  reason: collision with root package name */
    public int f1928a;

    /* renamed from: b  reason: collision with root package name */
    public b.h.b.i.d f1929b;

    /* renamed from: c  reason: collision with root package name */
    public l f1930c;

    /* renamed from: d  reason: collision with root package name */
    public int f1931d;

    /* renamed from: e  reason: collision with root package name */
    public g f1932e = new g(this);

    /* renamed from: f  reason: collision with root package name */
    public int f1933f = 0;

    /* renamed from: g  reason: collision with root package name */
    public boolean f1934g = false;

    /* renamed from: h  reason: collision with root package name */
    public f f1935h = new f(this);
    public f i = new f(this);
    public int j = 1;

    public o(b.h.b.i.d dVar) {
        this.f1929b = dVar;
    }

    @Override // b.h.b.i.l.d
    public void a(d dVar) {
    }

    public final void b(f fVar, f fVar2, int i) {
        fVar.l.add(fVar2);
        fVar.f1908f = i;
        fVar2.k.add(fVar);
    }

    public final void c(f fVar, f fVar2, int i, g gVar) {
        fVar.l.add(fVar2);
        fVar.l.add(this.f1932e);
        fVar.f1910h = i;
        fVar.i = gVar;
        fVar2.k.add(fVar);
        gVar.k.add(fVar);
    }

    public abstract void d();

    public abstract void e();

    public abstract void f();

    public final int g(int i, int i2) {
        int max;
        if (i2 == 0) {
            b.h.b.i.d dVar = this.f1929b;
            int i3 = dVar.p;
            max = Math.max(dVar.o, i);
            if (i3 > 0) {
                max = Math.min(i3, i);
            }
            if (max == i) {
                return i;
            }
        } else {
            b.h.b.i.d dVar2 = this.f1929b;
            int i4 = dVar2.s;
            max = Math.max(dVar2.r, i);
            if (i4 > 0) {
                max = Math.min(i4, i);
            }
            if (max == i) {
                return i;
            }
        }
        return max;
    }

    public final f h(b.h.b.i.c cVar) {
        b.h.b.i.c cVar2 = cVar.f1865f;
        if (cVar2 == null) {
            return null;
        }
        b.h.b.i.d dVar = cVar2.f1863d;
        int ordinal = cVar2.f1864e.ordinal();
        if (ordinal != 1) {
            if (ordinal != 2) {
                if (ordinal != 3) {
                    if (ordinal != 4) {
                        if (ordinal != 5) {
                            return null;
                        }
                        return dVar.f1879e.k;
                    }
                    return dVar.f1879e.i;
                }
                return dVar.f1878d.i;
            }
            return dVar.f1879e.f1935h;
        }
        return dVar.f1878d.f1935h;
    }

    public final f i(b.h.b.i.c cVar, int i) {
        b.h.b.i.c cVar2 = cVar.f1865f;
        if (cVar2 == null) {
            return null;
        }
        b.h.b.i.d dVar = cVar2.f1863d;
        o oVar = i == 0 ? dVar.f1878d : dVar.f1879e;
        int ordinal = cVar2.f1864e.ordinal();
        if (ordinal == 1 || ordinal == 2) {
            return oVar.f1935h;
        }
        if (ordinal == 3 || ordinal == 4) {
            return oVar.i;
        }
        return null;
    }

    public long j() {
        g gVar = this.f1932e;
        if (gVar.j) {
            return gVar.f1909g;
        }
        return 0L;
    }

    public abstract boolean k();

    /* JADX WARN: Code restructure failed: missing block: B:26:0x0051, code lost:
        if (r9.f1928a == 3) goto L46;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void l(b.h.b.i.c cVar, b.h.b.i.c cVar2, int i) {
        int i2;
        float f2;
        g gVar;
        g gVar2;
        int i3;
        f h2 = h(cVar);
        f h3 = h(cVar2);
        if (h2.j && h3.j) {
            int d2 = cVar.d() + h2.f1909g;
            int d3 = h3.f1909g - cVar2.d();
            int i4 = d3 - d2;
            g gVar3 = this.f1932e;
            if (!gVar3.j && this.f1931d == 3) {
                int i5 = this.f1928a;
                if (i5 == 0) {
                    gVar3.c(g(i4, i));
                } else if (i5 == 1) {
                    this.f1932e.c(Math.min(g(gVar3.m, i), i4));
                } else if (i5 == 2) {
                    b.h.b.i.d dVar = this.f1929b;
                    b.h.b.i.d dVar2 = dVar.P;
                    if (dVar2 != null) {
                        if ((i == 0 ? dVar2.f1878d : dVar2.f1879e).f1932e.j) {
                            gVar3.c(g((int) ((gVar.f1909g * (i == 0 ? dVar.q : dVar.t)) + 0.5f), i));
                        }
                    }
                } else if (i5 == 3) {
                    b.h.b.i.d dVar3 = this.f1929b;
                    o oVar = dVar3.f1878d;
                    if (oVar.f1931d == 3 && oVar.f1928a == 3) {
                        m mVar = dVar3.f1879e;
                        if (mVar.f1931d == 3) {
                        }
                    }
                    if (i == 0) {
                        oVar = dVar3.f1879e;
                    }
                    if (oVar.f1932e.j) {
                        float f3 = dVar3.S;
                        if (i == 1) {
                            i3 = (int) ((gVar2.f1909g / f3) + 0.5f);
                        } else {
                            i3 = (int) ((f3 * gVar2.f1909g) + 0.5f);
                        }
                        gVar3.c(i3);
                    }
                }
            }
            g gVar4 = this.f1932e;
            if (gVar4.j) {
                if (gVar4.f1909g == i4) {
                    this.f1935h.c(d2);
                    this.i.c(d3);
                    return;
                }
                b.h.b.i.d dVar4 = this.f1929b;
                if (i == 0) {
                    f2 = dVar4.Z;
                } else {
                    f2 = dVar4.a0;
                }
                if (h2 == h3) {
                    d2 = h2.f1909g;
                    d3 = h3.f1909g;
                    f2 = 0.5f;
                }
                this.f1935h.c((int) ((((d3 - d2) - i2) * f2) + d2 + 0.5f));
                this.i.c(this.f1935h.f1909g + this.f1932e.f1909g);
            }
        }
    }
}