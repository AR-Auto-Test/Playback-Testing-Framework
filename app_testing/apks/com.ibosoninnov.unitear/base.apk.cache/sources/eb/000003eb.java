package b.h.b.i;

import b.h.b.i.c;

/* compiled from: Barrier.java */
/* loaded from: classes.dex */
public class a extends h {
    public int n0 = 0;
    public boolean o0 = true;
    public int p0 = 0;
    public boolean q0 = false;

    @Override // b.h.b.i.d
    public boolean A() {
        return this.q0;
    }

    public boolean Q() {
        int i;
        int i2;
        int i3;
        c.a aVar = c.a.BOTTOM;
        c.a aVar2 = c.a.TOP;
        c.a aVar3 = c.a.RIGHT;
        c.a aVar4 = c.a.LEFT;
        int i4 = 0;
        boolean z = true;
        while (true) {
            i = this.m0;
            if (i4 >= i) {
                break;
            }
            d dVar = this.l0[i4];
            if ((this.o0 || dVar.e()) && ((((i2 = this.n0) == 0 || i2 == 1) && !dVar.z()) || (((i3 = this.n0) == 2 || i3 == 3) && !dVar.A()))) {
                z = false;
            }
            i4++;
        }
        if (!z || i <= 0) {
            return false;
        }
        int i5 = 0;
        boolean z2 = false;
        for (int i6 = 0; i6 < this.m0; i6++) {
            d dVar2 = this.l0[i6];
            if (this.o0 || dVar2.e()) {
                if (!z2) {
                    int i7 = this.n0;
                    if (i7 == 0) {
                        i5 = dVar2.i(aVar4).c();
                    } else if (i7 == 1) {
                        i5 = dVar2.i(aVar3).c();
                    } else if (i7 == 2) {
                        i5 = dVar2.i(aVar2).c();
                    } else if (i7 == 3) {
                        i5 = dVar2.i(aVar).c();
                    }
                    z2 = true;
                }
                int i8 = this.n0;
                if (i8 == 0) {
                    i5 = Math.min(i5, dVar2.i(aVar4).c());
                } else if (i8 == 1) {
                    i5 = Math.max(i5, dVar2.i(aVar3).c());
                } else if (i8 == 2) {
                    i5 = Math.min(i5, dVar2.i(aVar2).c());
                } else if (i8 == 3) {
                    i5 = Math.max(i5, dVar2.i(aVar).c());
                }
            }
        }
        int i9 = i5 + this.p0;
        int i10 = this.n0;
        if (i10 != 0 && i10 != 1) {
            G(i9, i9);
        } else {
            F(i9, i9);
        }
        this.q0 = true;
        return true;
    }

    public int R() {
        int i = this.n0;
        if (i == 0 || i == 1) {
            return 0;
        }
        return (i == 2 || i == 3) ? 1 : -1;
    }

    @Override // b.h.b.i.d
    public void d(b.h.b.d dVar, boolean z) {
        Object[] objArr;
        boolean z2;
        int i;
        int i2;
        int i3;
        c[] cVarArr = this.L;
        cVarArr[0] = this.D;
        cVarArr[2] = this.E;
        cVarArr[1] = this.F;
        cVarArr[3] = this.G;
        int i4 = 0;
        while (true) {
            objArr = this.L;
            if (i4 >= objArr.length) {
                break;
            }
            objArr[i4].i = dVar.l(objArr[i4]);
            i4++;
        }
        int i5 = this.n0;
        if (i5 < 0 || i5 >= 4) {
            return;
        }
        c cVar = objArr[i5];
        if (!this.q0) {
            Q();
        }
        if (this.q0) {
            this.q0 = false;
            int i6 = this.n0;
            if (i6 == 0 || i6 == 1) {
                dVar.e(this.D.i, this.U);
                dVar.e(this.F.i, this.U);
                return;
            } else if (i6 == 2 || i6 == 3) {
                dVar.e(this.E.i, this.V);
                dVar.e(this.G.i, this.V);
                return;
            } else {
                return;
            }
        }
        for (int i7 = 0; i7 < this.m0; i7++) {
            d dVar2 = this.l0[i7];
            if ((this.o0 || dVar2.e()) && ((((i2 = this.n0) == 0 || i2 == 1) && dVar2.m() == 3 && dVar2.D.f1865f != null && dVar2.F.f1865f != null) || (((i3 = this.n0) == 2 || i3 == 3) && dVar2.q() == 3 && dVar2.E.f1865f != null && dVar2.G.f1865f != null))) {
                z2 = true;
                break;
            }
        }
        z2 = false;
        boolean z3 = this.D.e() || this.F.e();
        boolean z4 = this.E.e() || this.G.e();
        int i8 = !z2 && (((i = this.n0) == 0 && z3) || ((i == 2 && z4) || ((i == 1 && z3) || (i == 3 && z4)))) ? 5 : 4;
        for (int i9 = 0; i9 < this.m0; i9++) {
            d dVar3 = this.l0[i9];
            if (this.o0 || dVar3.e()) {
                b.h.b.h l = dVar.l(dVar3.L[this.n0]);
                c[] cVarArr2 = dVar3.L;
                int i10 = this.n0;
                cVarArr2[i10].i = l;
                int i11 = (cVarArr2[i10].f1865f == null || cVarArr2[i10].f1865f.f1863d != this) ? 0 : cVarArr2[i10].f1866g + 0;
                if (i10 != 0 && i10 != 2) {
                    b.h.b.b m = dVar.m();
                    b.h.b.h n = dVar.n();
                    n.f1848e = 0;
                    m.e(cVar.i, l, n, this.p0 + i11);
                    dVar.c(m);
                } else {
                    b.h.b.b m2 = dVar.m();
                    b.h.b.h n2 = dVar.n();
                    n2.f1848e = 0;
                    m2.f(cVar.i, l, n2, this.p0 - i11);
                    dVar.c(m2);
                }
                dVar.d(cVar.i, l, this.p0 + i11, i8);
            }
        }
        int i12 = this.n0;
        if (i12 == 0) {
            dVar.d(this.F.i, this.D.i, 0, 8);
            dVar.d(this.D.i, this.P.F.i, 0, 4);
            dVar.d(this.D.i, this.P.D.i, 0, 0);
        } else if (i12 == 1) {
            dVar.d(this.D.i, this.F.i, 0, 8);
            dVar.d(this.D.i, this.P.D.i, 0, 4);
            dVar.d(this.D.i, this.P.F.i, 0, 0);
        } else if (i12 == 2) {
            dVar.d(this.G.i, this.E.i, 0, 8);
            dVar.d(this.E.i, this.P.G.i, 0, 4);
            dVar.d(this.E.i, this.P.E.i, 0, 0);
        } else if (i12 == 3) {
            dVar.d(this.E.i, this.G.i, 0, 8);
            dVar.d(this.E.i, this.P.E.i, 0, 4);
            dVar.d(this.E.i, this.P.G.i, 0, 0);
        }
    }

    @Override // b.h.b.i.d
    public boolean e() {
        return true;
    }

    @Override // b.h.b.i.d
    public String toString() {
        String v = c.b.a.a.a.v(c.b.a.a.a.x("[Barrier] "), this.d0, " {");
        for (int i = 0; i < this.m0; i++) {
            d dVar = this.l0[i];
            if (i > 0) {
                v = c.b.a.a.a.q(v, ", ");
            }
            StringBuilder x = c.b.a.a.a.x(v);
            x.append(dVar.d0);
            v = x.toString();
        }
        return c.b.a.a.a.q(v, "}");
    }

    @Override // b.h.b.i.d
    public boolean z() {
        return this.q0;
    }
}