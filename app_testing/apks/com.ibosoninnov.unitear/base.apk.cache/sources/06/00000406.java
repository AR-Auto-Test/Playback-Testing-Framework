package b.h.b.i.l;

import b.d.b.m0;
import b.h.b.i.c;
import b.h.b.i.l.f;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: VerticalWidgetRun.java */
/* loaded from: classes.dex */
public class m extends o {
    public f k;
    public g l;

    public m(b.h.b.i.d dVar) {
        super(dVar);
        f fVar = new f(this);
        this.k = fVar;
        this.l = null;
        this.f1935h.f1907e = f.a.TOP;
        this.i.f1907e = f.a.BOTTOM;
        fVar.f1907e = f.a.BASELINE;
        this.f1933f = 1;
    }

    @Override // b.h.b.i.l.o, b.h.b.i.l.d
    public void a(d dVar) {
        g gVar;
        float f2;
        float f3;
        float f4;
        int i;
        int f5 = m0.f(this.j);
        if (f5 != 1 && f5 != 2 && f5 == 3) {
            b.h.b.i.d dVar2 = this.f1929b;
            l(dVar2.E, dVar2.G, 1);
            return;
        }
        g gVar2 = this.f1932e;
        if (gVar2.f1905c && !gVar2.j && this.f1931d == 3) {
            b.h.b.i.d dVar3 = this.f1929b;
            int i2 = dVar3.m;
            if (i2 == 2) {
                b.h.b.i.d dVar4 = dVar3.P;
                if (dVar4 != null) {
                    if (dVar4.f1879e.f1932e.j) {
                        gVar2.c((int) ((gVar.f1909g * dVar3.t) + 0.5f));
                    }
                }
            } else if (i2 == 3) {
                g gVar3 = dVar3.f1878d.f1932e;
                if (gVar3.j) {
                    int i3 = dVar3.T;
                    if (i3 == -1) {
                        f2 = gVar3.f1909g;
                        f3 = dVar3.S;
                    } else if (i3 == 0) {
                        f4 = gVar3.f1909g * dVar3.S;
                        i = (int) (f4 + 0.5f);
                        gVar2.c(i);
                    } else if (i3 == 1) {
                        f2 = gVar3.f1909g;
                        f3 = dVar3.S;
                    } else {
                        i = 0;
                        gVar2.c(i);
                    }
                    f4 = f2 / f3;
                    i = (int) (f4 + 0.5f);
                    gVar2.c(i);
                }
            }
        }
        f fVar = this.f1935h;
        if (fVar.f1905c) {
            f fVar2 = this.i;
            if (fVar2.f1905c) {
                if (fVar.j && fVar2.j && this.f1932e.j) {
                    return;
                }
                if (!this.f1932e.j && this.f1931d == 3) {
                    b.h.b.i.d dVar5 = this.f1929b;
                    if (dVar5.l == 0 && !dVar5.x()) {
                        int i4 = this.f1935h.l.get(0).f1909g;
                        f fVar3 = this.f1935h;
                        int i5 = i4 + fVar3.f1908f;
                        int i6 = this.i.l.get(0).f1909g + this.i.f1908f;
                        fVar3.c(i5);
                        this.i.c(i6);
                        this.f1932e.c(i6 - i5);
                        return;
                    }
                }
                if (!this.f1932e.j && this.f1931d == 3 && this.f1928a == 1 && this.f1935h.l.size() > 0 && this.i.l.size() > 0) {
                    int i7 = (this.i.l.get(0).f1909g + this.i.f1908f) - (this.f1935h.l.get(0).f1909g + this.f1935h.f1908f);
                    g gVar4 = this.f1932e;
                    int i8 = gVar4.m;
                    if (i7 < i8) {
                        gVar4.c(i7);
                    } else {
                        gVar4.c(i8);
                    }
                }
                if (this.f1932e.j && this.f1935h.l.size() > 0 && this.i.l.size() > 0) {
                    f fVar4 = this.f1935h.l.get(0);
                    f fVar5 = this.i.l.get(0);
                    int i9 = fVar4.f1909g;
                    f fVar6 = this.f1935h;
                    int i10 = fVar6.f1908f + i9;
                    int i11 = fVar5.f1909g;
                    int i12 = this.i.f1908f + i11;
                    float f6 = this.f1929b.a0;
                    if (fVar4 == fVar5) {
                        f6 = 0.5f;
                    } else {
                        i9 = i10;
                        i11 = i12;
                    }
                    fVar6.c((int) ((((i11 - i9) - this.f1932e.f1909g) * f6) + i9 + 0.5f));
                    this.i.c(this.f1935h.f1909g + this.f1932e.f1909g);
                }
            }
        }
    }

    @Override // b.h.b.i.l.o
    public void d() {
        b.h.b.i.d dVar;
        b.h.b.i.d dVar2;
        b.h.b.i.d dVar3;
        b.h.b.i.d dVar4 = this.f1929b;
        if (dVar4.f1875a) {
            this.f1932e.c(dVar4.l());
        }
        if (!this.f1932e.j) {
            this.f1931d = this.f1929b.q();
            if (this.f1929b.y) {
                this.l = new a(this);
            }
            int i = this.f1931d;
            if (i != 3) {
                if (i == 4 && (dVar3 = this.f1929b.P) != null && dVar3.q() == 1) {
                    int l = (dVar3.l() - this.f1929b.E.d()) - this.f1929b.G.d();
                    b(this.f1935h, dVar3.f1879e.f1935h, this.f1929b.E.d());
                    b(this.i, dVar3.f1879e.i, -this.f1929b.G.d());
                    this.f1932e.c(l);
                    return;
                } else if (this.f1931d == 1) {
                    this.f1932e.c(this.f1929b.l());
                }
            }
        } else if (this.f1931d == 4 && (dVar = this.f1929b.P) != null && dVar.q() == 1) {
            b(this.f1935h, dVar.f1879e.f1935h, this.f1929b.E.d());
            b(this.i, dVar.f1879e.i, -this.f1929b.G.d());
            return;
        }
        g gVar = this.f1932e;
        boolean z = gVar.j;
        if (z) {
            b.h.b.i.d dVar5 = this.f1929b;
            if (dVar5.f1875a) {
                b.h.b.i.c[] cVarArr = dVar5.L;
                if (cVarArr[2].f1865f != null && cVarArr[3].f1865f != null) {
                    if (dVar5.x()) {
                        this.f1935h.f1908f = this.f1929b.L[2].d();
                        this.i.f1908f = -this.f1929b.L[3].d();
                    } else {
                        f h2 = h(this.f1929b.L[2]);
                        if (h2 != null) {
                            f fVar = this.f1935h;
                            int d2 = this.f1929b.L[2].d();
                            fVar.l.add(h2);
                            fVar.f1908f = d2;
                            h2.k.add(fVar);
                        }
                        f h3 = h(this.f1929b.L[3]);
                        if (h3 != null) {
                            f fVar2 = this.i;
                            fVar2.l.add(h3);
                            fVar2.f1908f = -this.f1929b.L[3].d();
                            h3.k.add(fVar2);
                        }
                        this.f1935h.f1904b = true;
                        this.i.f1904b = true;
                    }
                    b.h.b.i.d dVar6 = this.f1929b;
                    if (dVar6.y) {
                        b(this.k, this.f1935h, dVar6.W);
                        return;
                    }
                    return;
                } else if (cVarArr[2].f1865f != null) {
                    f h4 = h(cVarArr[2]);
                    if (h4 != null) {
                        f fVar3 = this.f1935h;
                        int d3 = this.f1929b.L[2].d();
                        fVar3.l.add(h4);
                        fVar3.f1908f = d3;
                        h4.k.add(fVar3);
                        b(this.i, this.f1935h, this.f1932e.f1909g);
                        b.h.b.i.d dVar7 = this.f1929b;
                        if (dVar7.y) {
                            b(this.k, this.f1935h, dVar7.W);
                            return;
                        }
                        return;
                    }
                    return;
                } else if (cVarArr[3].f1865f != null) {
                    f h5 = h(cVarArr[3]);
                    if (h5 != null) {
                        f fVar4 = this.i;
                        fVar4.l.add(h5);
                        fVar4.f1908f = -this.f1929b.L[3].d();
                        h5.k.add(fVar4);
                        b(this.f1935h, this.i, -this.f1932e.f1909g);
                    }
                    b.h.b.i.d dVar8 = this.f1929b;
                    if (dVar8.y) {
                        b(this.k, this.f1935h, dVar8.W);
                        return;
                    }
                    return;
                } else if (cVarArr[4].f1865f != null) {
                    f h6 = h(cVarArr[4]);
                    if (h6 != null) {
                        f fVar5 = this.k;
                        fVar5.l.add(h6);
                        fVar5.f1908f = 0;
                        h6.k.add(fVar5);
                        b(this.f1935h, this.k, -this.f1929b.W);
                        b(this.i, this.f1935h, this.f1932e.f1909g);
                        return;
                    }
                    return;
                } else if ((dVar5 instanceof b.h.b.i.g) || dVar5.P == null || dVar5.i(c.a.CENTER).f1865f != null) {
                    return;
                } else {
                    b.h.b.i.d dVar9 = this.f1929b;
                    b(this.f1935h, dVar9.P.f1879e.f1935h, dVar9.t());
                    b(this.i, this.f1935h, this.f1932e.f1909g);
                    b.h.b.i.d dVar10 = this.f1929b;
                    if (dVar10.y) {
                        b(this.k, this.f1935h, dVar10.W);
                        return;
                    }
                    return;
                }
            }
        }
        if (!z && this.f1931d == 3) {
            b.h.b.i.d dVar11 = this.f1929b;
            int i2 = dVar11.m;
            if (i2 != 2) {
                if (i2 == 3 && !dVar11.x()) {
                    b.h.b.i.d dVar12 = this.f1929b;
                    if (dVar12.l != 3) {
                        g gVar2 = dVar12.f1878d.f1932e;
                        this.f1932e.l.add(gVar2);
                        gVar2.k.add(this.f1932e);
                        g gVar3 = this.f1932e;
                        gVar3.f1904b = true;
                        gVar3.k.add(this.f1935h);
                        this.f1932e.k.add(this.i);
                    }
                }
            } else {
                b.h.b.i.d dVar13 = dVar11.P;
                if (dVar13 != null) {
                    g gVar4 = dVar13.f1879e.f1932e;
                    gVar.l.add(gVar4);
                    gVar4.k.add(this.f1932e);
                    g gVar5 = this.f1932e;
                    gVar5.f1904b = true;
                    gVar5.k.add(this.f1935h);
                    this.f1932e.k.add(this.i);
                }
            }
        } else {
            gVar.k.add(this);
            if (gVar.j) {
                a(this);
            }
        }
        b.h.b.i.d dVar14 = this.f1929b;
        b.h.b.i.c[] cVarArr2 = dVar14.L;
        if (cVarArr2[2].f1865f != null && cVarArr2[3].f1865f != null) {
            if (dVar14.x()) {
                this.f1935h.f1908f = this.f1929b.L[2].d();
                this.i.f1908f = -this.f1929b.L[3].d();
            } else {
                f h7 = h(this.f1929b.L[2]);
                f h8 = h(this.f1929b.L[3]);
                h7.k.add(this);
                if (h7.j) {
                    a(this);
                }
                h8.k.add(this);
                if (h8.j) {
                    a(this);
                }
                this.j = 4;
            }
            if (this.f1929b.y) {
                c(this.k, this.f1935h, 1, this.l);
            }
        } else if (cVarArr2[2].f1865f != null) {
            f h9 = h(cVarArr2[2]);
            if (h9 != null) {
                f fVar6 = this.f1935h;
                int d4 = this.f1929b.L[2].d();
                fVar6.l.add(h9);
                fVar6.f1908f = d4;
                h9.k.add(fVar6);
                c(this.i, this.f1935h, 1, this.f1932e);
                if (this.f1929b.y) {
                    c(this.k, this.f1935h, 1, this.l);
                }
                if (this.f1931d == 3) {
                    b.h.b.i.d dVar15 = this.f1929b;
                    if (dVar15.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        k kVar = dVar15.f1878d;
                        if (kVar.f1931d == 3) {
                            kVar.f1932e.k.add(this.f1932e);
                            this.f1932e.l.add(this.f1929b.f1878d.f1932e);
                            this.f1932e.f1903a = this;
                        }
                    }
                }
            }
        } else if (cVarArr2[3].f1865f != null) {
            f h10 = h(cVarArr2[3]);
            if (h10 != null) {
                f fVar7 = this.i;
                fVar7.l.add(h10);
                fVar7.f1908f = -this.f1929b.L[3].d();
                h10.k.add(fVar7);
                c(this.f1935h, this.i, -1, this.f1932e);
                if (this.f1929b.y) {
                    c(this.k, this.f1935h, 1, this.l);
                }
            }
        } else if (cVarArr2[4].f1865f != null) {
            f h11 = h(cVarArr2[4]);
            if (h11 != null) {
                f fVar8 = this.k;
                fVar8.l.add(h11);
                fVar8.f1908f = 0;
                h11.k.add(fVar8);
                c(this.f1935h, this.k, -1, this.l);
                c(this.i, this.f1935h, 1, this.f1932e);
            }
        } else if (!(dVar14 instanceof b.h.b.i.g) && (dVar2 = dVar14.P) != null) {
            b(this.f1935h, dVar2.f1879e.f1935h, dVar14.t());
            c(this.i, this.f1935h, 1, this.f1932e);
            if (this.f1929b.y) {
                c(this.k, this.f1935h, 1, this.l);
            }
            if (this.f1931d == 3) {
                b.h.b.i.d dVar16 = this.f1929b;
                if (dVar16.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    k kVar2 = dVar16.f1878d;
                    if (kVar2.f1931d == 3) {
                        kVar2.f1932e.k.add(this.f1932e);
                        this.f1932e.l.add(this.f1929b.f1878d.f1932e);
                        this.f1932e.f1903a = this;
                    }
                }
            }
        }
        if (this.f1932e.l.size() == 0) {
            this.f1932e.f1905c = true;
        }
    }

    @Override // b.h.b.i.l.o
    public void e() {
        f fVar = this.f1935h;
        if (fVar.j) {
            this.f1929b.V = fVar.f1909g;
        }
    }

    @Override // b.h.b.i.l.o
    public void f() {
        this.f1930c = null;
        this.f1935h.b();
        this.i.b();
        this.k.b();
        this.f1932e.b();
        this.f1934g = false;
    }

    @Override // b.h.b.i.l.o
    public boolean k() {
        return this.f1931d != 3 || this.f1929b.m == 0;
    }

    public void m() {
        this.f1934g = false;
        this.f1935h.b();
        this.f1935h.j = false;
        this.i.b();
        this.i.j = false;
        this.k.b();
        this.k.j = false;
        this.f1932e.j = false;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("VerticalRun ");
        x.append(this.f1929b.d0);
        return x.toString();
    }
}