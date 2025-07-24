package b.h.b.i.l;

import b.d.b.m0;
import b.h.b.i.c;
import b.h.b.i.l.f;

/* compiled from: HorizontalWidgetRun.java */
/* loaded from: classes.dex */
public class k extends o {
    public static int[] k = new int[2];

    public k(b.h.b.i.d dVar) {
        super(dVar);
        this.f1935h.f1907e = f.a.LEFT;
        this.i.f1907e = f.a.RIGHT;
        this.f1933f = 0;
    }

    /* JADX WARN: Code restructure failed: missing block: B:118:0x028a, code lost:
        if (r15 != 1) goto L132;
     */
    @Override // b.h.b.i.l.o, b.h.b.i.l.d
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void a(d dVar) {
        g gVar;
        float f2;
        f fVar;
        float f3;
        float f4;
        float f5;
        int i;
        int f6 = m0.f(this.j);
        if (f6 != 1 && f6 != 2 && f6 == 3) {
            b.h.b.i.d dVar2 = this.f1929b;
            l(dVar2.D, dVar2.F, 0);
            return;
        }
        g gVar2 = this.f1932e;
        if (!gVar2.j && this.f1931d == 3) {
            b.h.b.i.d dVar3 = this.f1929b;
            int i2 = dVar3.l;
            if (i2 == 2) {
                b.h.b.i.d dVar4 = dVar3.P;
                if (dVar4 != null) {
                    if (dVar4.f1878d.f1932e.j) {
                        gVar2.c((int) ((gVar.f1909g * dVar3.q) + 0.5f));
                    }
                }
            } else if (i2 == 3) {
                int i3 = dVar3.m;
                if (i3 != 0 && i3 != 3) {
                    int i4 = dVar3.T;
                    if (i4 == -1) {
                        f3 = dVar3.f1879e.f1932e.f1909g;
                        f4 = dVar3.S;
                    } else if (i4 == 0) {
                        f5 = dVar3.f1879e.f1932e.f1909g / dVar3.S;
                        i = (int) (f5 + 0.5f);
                        gVar2.c(i);
                    } else if (i4 == 1) {
                        f3 = dVar3.f1879e.f1932e.f1909g;
                        f4 = dVar3.S;
                    } else {
                        i = 0;
                        gVar2.c(i);
                    }
                    f5 = f3 * f4;
                    i = (int) (f5 + 0.5f);
                    gVar2.c(i);
                } else {
                    m mVar = dVar3.f1879e;
                    f fVar2 = mVar.f1935h;
                    f fVar3 = mVar.i;
                    boolean z = dVar3.D.f1865f != null;
                    boolean z2 = dVar3.E.f1865f != null;
                    boolean z3 = dVar3.F.f1865f != null;
                    boolean z4 = dVar3.G.f1865f != null;
                    int i5 = dVar3.T;
                    if (z && z2 && z3 && z4) {
                        float f7 = dVar3.S;
                        if (fVar2.j && fVar3.j) {
                            f fVar4 = this.f1935h;
                            if (fVar4.f1905c && this.i.f1905c) {
                                m(k, fVar4.l.get(0).f1909g + this.f1935h.f1908f, this.i.l.get(0).f1909g - this.i.f1908f, fVar2.f1909g + fVar2.f1908f, fVar3.f1909g - fVar3.f1908f, f7, i5);
                                this.f1932e.c(k[0]);
                                this.f1929b.f1879e.f1932e.c(k[1]);
                                return;
                            }
                            return;
                        }
                        f fVar5 = this.f1935h;
                        if (fVar5.j) {
                            f fVar6 = this.i;
                            if (fVar6.j) {
                                if (!fVar2.f1905c || !fVar3.f1905c) {
                                    return;
                                }
                                f2 = f7;
                                m(k, fVar5.f1909g + fVar5.f1908f, fVar6.f1909g - fVar6.f1908f, fVar2.l.get(0).f1909g + fVar2.f1908f, fVar3.l.get(0).f1909g - fVar3.f1908f, f7, i5);
                                this.f1932e.c(k[0]);
                                this.f1929b.f1879e.f1932e.c(k[1]);
                                fVar = this.f1935h;
                                if (fVar.f1905c || !this.i.f1905c || !fVar2.f1905c || !fVar3.f1905c) {
                                    return;
                                }
                                m(k, fVar.l.get(0).f1909g + this.f1935h.f1908f, this.i.l.get(0).f1909g - this.i.f1908f, fVar2.l.get(0).f1909g + fVar2.f1908f, fVar3.l.get(0).f1909g - fVar3.f1908f, f2, i5);
                                this.f1932e.c(k[0]);
                                this.f1929b.f1879e.f1932e.c(k[1]);
                            }
                        }
                        f2 = f7;
                        fVar = this.f1935h;
                        if (fVar.f1905c) {
                            return;
                        }
                        return;
                    } else if (z && z3) {
                        f fVar7 = this.f1935h;
                        if (!fVar7.f1905c || !this.i.f1905c) {
                            return;
                        }
                        float f8 = dVar3.S;
                        int i6 = fVar7.l.get(0).f1909g + this.f1935h.f1908f;
                        int i7 = this.i.l.get(0).f1909g - this.i.f1908f;
                        if (i5 == -1 || i5 == 0) {
                            int g2 = g(i7 - i6, 0);
                            int i8 = (int) ((g2 * f8) + 0.5f);
                            int g3 = g(i8, 1);
                            if (i8 != g3) {
                                g2 = (int) ((g3 / f8) + 0.5f);
                            }
                            this.f1932e.c(g2);
                            this.f1929b.f1879e.f1932e.c(g3);
                        } else if (i5 == 1) {
                            int g4 = g(i7 - i6, 0);
                            int i9 = (int) ((g4 / f8) + 0.5f);
                            int g5 = g(i9, 1);
                            if (i9 != g5) {
                                g4 = (int) ((g5 * f8) + 0.5f);
                            }
                            this.f1932e.c(g4);
                            this.f1929b.f1879e.f1932e.c(g5);
                        }
                    } else if (z2 && z4) {
                        if (!fVar2.f1905c || !fVar3.f1905c) {
                            return;
                        }
                        float f9 = dVar3.S;
                        int i10 = fVar2.l.get(0).f1909g + fVar2.f1908f;
                        int i11 = fVar3.l.get(0).f1909g - fVar3.f1908f;
                        if (i5 != -1) {
                            if (i5 == 0) {
                                int g6 = g(i11 - i10, 1);
                                int i12 = (int) ((g6 * f9) + 0.5f);
                                int g7 = g(i12, 0);
                                if (i12 != g7) {
                                    g6 = (int) ((g7 / f9) + 0.5f);
                                }
                                this.f1932e.c(g7);
                                this.f1929b.f1879e.f1932e.c(g6);
                            }
                        }
                        int g8 = g(i11 - i10, 1);
                        int i13 = (int) ((g8 / f9) + 0.5f);
                        int g9 = g(i13, 0);
                        if (i13 != g9) {
                            g8 = (int) ((g9 * f9) + 0.5f);
                        }
                        this.f1932e.c(g9);
                        this.f1929b.f1879e.f1932e.c(g8);
                    }
                }
            }
        }
        f fVar8 = this.f1935h;
        if (fVar8.f1905c) {
            f fVar9 = this.i;
            if (fVar9.f1905c) {
                if (fVar8.j && fVar9.j && this.f1932e.j) {
                    return;
                }
                if (!this.f1932e.j && this.f1931d == 3) {
                    b.h.b.i.d dVar5 = this.f1929b;
                    if (dVar5.l == 0 && !dVar5.w()) {
                        int i14 = this.f1935h.l.get(0).f1909g;
                        f fVar10 = this.f1935h;
                        int i15 = i14 + fVar10.f1908f;
                        int i16 = this.i.l.get(0).f1909g + this.i.f1908f;
                        fVar10.c(i15);
                        this.i.c(i16);
                        this.f1932e.c(i16 - i15);
                        return;
                    }
                }
                if (!this.f1932e.j && this.f1931d == 3 && this.f1928a == 1 && this.f1935h.l.size() > 0 && this.i.l.size() > 0) {
                    int min = Math.min((this.i.l.get(0).f1909g + this.i.f1908f) - (this.f1935h.l.get(0).f1909g + this.f1935h.f1908f), this.f1932e.m);
                    b.h.b.i.d dVar6 = this.f1929b;
                    int i17 = dVar6.p;
                    int max = Math.max(dVar6.o, min);
                    if (i17 > 0) {
                        max = Math.min(i17, max);
                    }
                    this.f1932e.c(max);
                }
                if (this.f1932e.j) {
                    f fVar11 = this.f1935h.l.get(0);
                    f fVar12 = this.i.l.get(0);
                    int i18 = fVar11.f1909g;
                    f fVar13 = this.f1935h;
                    int i19 = fVar13.f1908f + i18;
                    int i20 = fVar12.f1909g;
                    int i21 = this.i.f1908f + i20;
                    float f10 = this.f1929b.Z;
                    if (fVar11 == fVar12) {
                        f10 = 0.5f;
                    } else {
                        i18 = i19;
                        i20 = i21;
                    }
                    fVar13.c((int) ((((i20 - i18) - this.f1932e.f1909g) * f10) + i18 + 0.5f));
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
            this.f1932e.c(dVar4.r());
        }
        if (!this.f1932e.j) {
            int m = this.f1929b.m();
            this.f1931d = m;
            if (m != 3) {
                if (m == 4 && (((dVar3 = this.f1929b.P) != null && dVar3.m() == 1) || dVar3.m() == 4)) {
                    int r = (dVar3.r() - this.f1929b.D.d()) - this.f1929b.F.d();
                    b(this.f1935h, dVar3.f1878d.f1935h, this.f1929b.D.d());
                    b(this.i, dVar3.f1878d.i, -this.f1929b.F.d());
                    this.f1932e.c(r);
                    return;
                } else if (this.f1931d == 1) {
                    this.f1932e.c(this.f1929b.r());
                }
            }
        } else if (this.f1931d == 4 && (((dVar = this.f1929b.P) != null && dVar.m() == 1) || dVar.m() == 4)) {
            b(this.f1935h, dVar.f1878d.f1935h, this.f1929b.D.d());
            b(this.i, dVar.f1878d.i, -this.f1929b.F.d());
            return;
        }
        g gVar = this.f1932e;
        if (gVar.j) {
            b.h.b.i.d dVar5 = this.f1929b;
            if (dVar5.f1875a) {
                b.h.b.i.c[] cVarArr = dVar5.L;
                if (cVarArr[0].f1865f != null && cVarArr[1].f1865f != null) {
                    if (dVar5.w()) {
                        this.f1935h.f1908f = this.f1929b.L[0].d();
                        this.i.f1908f = -this.f1929b.L[1].d();
                        return;
                    }
                    f h2 = h(this.f1929b.L[0]);
                    if (h2 != null) {
                        f fVar = this.f1935h;
                        int d2 = this.f1929b.L[0].d();
                        fVar.l.add(h2);
                        fVar.f1908f = d2;
                        h2.k.add(fVar);
                    }
                    f h3 = h(this.f1929b.L[1]);
                    if (h3 != null) {
                        f fVar2 = this.i;
                        fVar2.l.add(h3);
                        fVar2.f1908f = -this.f1929b.L[1].d();
                        h3.k.add(fVar2);
                    }
                    this.f1935h.f1904b = true;
                    this.i.f1904b = true;
                    return;
                } else if (cVarArr[0].f1865f != null) {
                    f h4 = h(cVarArr[0]);
                    if (h4 != null) {
                        f fVar3 = this.f1935h;
                        int d3 = this.f1929b.L[0].d();
                        fVar3.l.add(h4);
                        fVar3.f1908f = d3;
                        h4.k.add(fVar3);
                        b(this.i, this.f1935h, this.f1932e.f1909g);
                        return;
                    }
                    return;
                } else if (cVarArr[1].f1865f != null) {
                    f h5 = h(cVarArr[1]);
                    if (h5 != null) {
                        f fVar4 = this.i;
                        fVar4.l.add(h5);
                        fVar4.f1908f = -this.f1929b.L[1].d();
                        h5.k.add(fVar4);
                        b(this.f1935h, this.i, -this.f1932e.f1909g);
                        return;
                    }
                    return;
                } else if ((dVar5 instanceof b.h.b.i.g) || dVar5.P == null || dVar5.i(c.a.CENTER).f1865f != null) {
                    return;
                } else {
                    b.h.b.i.d dVar6 = this.f1929b;
                    b(this.f1935h, dVar6.P.f1878d.f1935h, dVar6.s());
                    b(this.i, this.f1935h, this.f1932e.f1909g);
                    return;
                }
            }
        }
        if (this.f1931d == 3) {
            b.h.b.i.d dVar7 = this.f1929b;
            int i = dVar7.l;
            if (i == 2) {
                b.h.b.i.d dVar8 = dVar7.P;
                if (dVar8 != null) {
                    g gVar2 = dVar8.f1879e.f1932e;
                    gVar.l.add(gVar2);
                    gVar2.k.add(this.f1932e);
                    g gVar3 = this.f1932e;
                    gVar3.f1904b = true;
                    gVar3.k.add(this.f1935h);
                    this.f1932e.k.add(this.i);
                }
            } else if (i == 3) {
                if (dVar7.m == 3) {
                    this.f1935h.f1903a = this;
                    this.i.f1903a = this;
                    m mVar = dVar7.f1879e;
                    mVar.f1935h.f1903a = this;
                    mVar.i.f1903a = this;
                    gVar.f1903a = this;
                    if (dVar7.x()) {
                        this.f1932e.l.add(this.f1929b.f1879e.f1932e);
                        this.f1929b.f1879e.f1932e.k.add(this.f1932e);
                        m mVar2 = this.f1929b.f1879e;
                        mVar2.f1932e.f1903a = this;
                        this.f1932e.l.add(mVar2.f1935h);
                        this.f1932e.l.add(this.f1929b.f1879e.i);
                        this.f1929b.f1879e.f1935h.k.add(this.f1932e);
                        this.f1929b.f1879e.i.k.add(this.f1932e);
                    } else if (this.f1929b.w()) {
                        this.f1929b.f1879e.f1932e.l.add(this.f1932e);
                        this.f1932e.k.add(this.f1929b.f1879e.f1932e);
                    } else {
                        this.f1929b.f1879e.f1932e.l.add(this.f1932e);
                    }
                } else {
                    g gVar4 = dVar7.f1879e.f1932e;
                    gVar.l.add(gVar4);
                    gVar4.k.add(this.f1932e);
                    this.f1929b.f1879e.f1935h.k.add(this.f1932e);
                    this.f1929b.f1879e.i.k.add(this.f1932e);
                    g gVar5 = this.f1932e;
                    gVar5.f1904b = true;
                    gVar5.k.add(this.f1935h);
                    this.f1932e.k.add(this.i);
                    this.f1935h.l.add(this.f1932e);
                    this.i.l.add(this.f1932e);
                }
            }
        }
        b.h.b.i.d dVar9 = this.f1929b;
        b.h.b.i.c[] cVarArr2 = dVar9.L;
        if (cVarArr2[0].f1865f != null && cVarArr2[1].f1865f != null) {
            if (dVar9.w()) {
                this.f1935h.f1908f = this.f1929b.L[0].d();
                this.i.f1908f = -this.f1929b.L[1].d();
                return;
            }
            f h6 = h(this.f1929b.L[0]);
            f h7 = h(this.f1929b.L[1]);
            h6.k.add(this);
            if (h6.j) {
                a(this);
            }
            h7.k.add(this);
            if (h7.j) {
                a(this);
            }
            this.j = 4;
        } else if (cVarArr2[0].f1865f != null) {
            f h8 = h(cVarArr2[0]);
            if (h8 != null) {
                f fVar5 = this.f1935h;
                int d4 = this.f1929b.L[0].d();
                fVar5.l.add(h8);
                fVar5.f1908f = d4;
                h8.k.add(fVar5);
                c(this.i, this.f1935h, 1, this.f1932e);
            }
        } else if (cVarArr2[1].f1865f != null) {
            f h9 = h(cVarArr2[1]);
            if (h9 != null) {
                f fVar6 = this.i;
                fVar6.l.add(h9);
                fVar6.f1908f = -this.f1929b.L[1].d();
                h9.k.add(fVar6);
                c(this.f1935h, this.i, -1, this.f1932e);
            }
        } else if ((dVar9 instanceof b.h.b.i.g) || (dVar2 = dVar9.P) == null) {
        } else {
            b(this.f1935h, dVar2.f1878d.f1935h, dVar9.s());
            c(this.i, this.f1935h, 1, this.f1932e);
        }
    }

    @Override // b.h.b.i.l.o
    public void e() {
        f fVar = this.f1935h;
        if (fVar.j) {
            this.f1929b.U = fVar.f1909g;
        }
    }

    @Override // b.h.b.i.l.o
    public void f() {
        this.f1930c = null;
        this.f1935h.b();
        this.i.b();
        this.f1932e.b();
        this.f1934g = false;
    }

    @Override // b.h.b.i.l.o
    public boolean k() {
        return this.f1931d != 3 || this.f1929b.l == 0;
    }

    public final void m(int[] iArr, int i, int i2, int i3, int i4, float f2, int i5) {
        int i6 = i2 - i;
        int i7 = i4 - i3;
        if (i5 != -1) {
            if (i5 == 0) {
                iArr[0] = (int) ((i7 * f2) + 0.5f);
                iArr[1] = i7;
                return;
            } else if (i5 != 1) {
                return;
            } else {
                iArr[0] = i6;
                iArr[1] = (int) ((i6 * f2) + 0.5f);
                return;
            }
        }
        int i8 = (int) ((i7 * f2) + 0.5f);
        int i9 = (int) ((i6 / f2) + 0.5f);
        if (i8 <= i6 && i7 <= i7) {
            iArr[0] = i8;
            iArr[1] = i7;
        } else if (i6 > i6 || i9 > i7) {
        } else {
            iArr[0] = i6;
            iArr[1] = i9;
        }
    }

    public void n() {
        this.f1934g = false;
        this.f1935h.b();
        this.f1935h.j = false;
        this.i.b();
        this.i.j = false;
        this.f1932e.j = false;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("HorizontalRun ");
        x.append(this.f1929b.d0);
        return x.toString();
    }
}