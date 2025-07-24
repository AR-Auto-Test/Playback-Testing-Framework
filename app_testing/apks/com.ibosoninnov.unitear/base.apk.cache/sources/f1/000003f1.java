package b.h.b.i;

import b.h.b.i.c;

/* compiled from: Guideline.java */
/* loaded from: classes.dex */
public class f extends d {
    public float l0 = -1.0f;
    public int m0 = -1;
    public int n0 = -1;
    public c o0 = this.E;
    public int p0 = 0;
    public boolean q0;

    public f() {
        this.M.clear();
        this.M.add(this.o0);
        int length = this.L.length;
        for (int i = 0; i < length; i++) {
            this.L[i] = this.o0;
        }
    }

    @Override // b.h.b.i.d
    public boolean A() {
        return this.q0;
    }

    @Override // b.h.b.i.d
    public void O(b.h.b.d dVar, boolean z) {
        if (this.P == null) {
            return;
        }
        int o = dVar.o(this.o0);
        if (this.p0 == 1) {
            this.U = o;
            this.V = 0;
            H(this.P.l());
            M(0);
            return;
        }
        this.U = 0;
        this.V = o;
        M(this.P.r());
        H(0);
    }

    public void P(int i) {
        c cVar = this.o0;
        cVar.f1861b = i;
        cVar.f1862c = true;
        this.q0 = true;
    }

    public void Q(int i) {
        if (this.p0 == i) {
            return;
        }
        this.p0 = i;
        this.M.clear();
        if (this.p0 == 1) {
            this.o0 = this.D;
        } else {
            this.o0 = this.E;
        }
        this.M.add(this.o0);
        int length = this.L.length;
        for (int i2 = 0; i2 < length; i2++) {
            this.L[i2] = this.o0;
        }
    }

    @Override // b.h.b.i.d
    public void d(b.h.b.d dVar, boolean z) {
        e eVar = (e) this.P;
        if (eVar == null) {
            return;
        }
        Object i = eVar.i(c.a.LEFT);
        Object i2 = eVar.i(c.a.RIGHT);
        d dVar2 = this.P;
        boolean z2 = true;
        boolean z3 = dVar2 != null && dVar2.O[0] == 2;
        if (this.p0 == 0) {
            i = eVar.i(c.a.TOP);
            i2 = eVar.i(c.a.BOTTOM);
            d dVar3 = this.P;
            if (dVar3 == null || dVar3.O[1] != 2) {
                z2 = false;
            }
            z3 = z2;
        }
        if (this.q0) {
            c cVar = this.o0;
            if (cVar.f1862c) {
                b.h.b.h l = dVar.l(cVar);
                dVar.e(l, this.o0.c());
                if (this.m0 != -1) {
                    if (z3) {
                        dVar.f(dVar.l(i2), l, 0, 5);
                    }
                } else if (this.n0 != -1 && z3) {
                    b.h.b.h l2 = dVar.l(i2);
                    dVar.f(l, dVar.l(i), 0, 5);
                    dVar.f(l2, l, 0, 5);
                }
                this.q0 = false;
                return;
            }
        }
        if (this.m0 != -1) {
            b.h.b.h l3 = dVar.l(this.o0);
            dVar.d(l3, dVar.l(i), this.m0, 8);
            if (z3) {
                dVar.f(dVar.l(i2), l3, 0, 5);
            }
        } else if (this.n0 != -1) {
            b.h.b.h l4 = dVar.l(this.o0);
            b.h.b.h l5 = dVar.l(i2);
            dVar.d(l4, l5, -this.n0, 8);
            if (z3) {
                dVar.f(l4, dVar.l(i), 0, 5);
                dVar.f(l5, l4, 0, 5);
            }
        } else if (this.l0 != -1.0f) {
            b.h.b.h l6 = dVar.l(this.o0);
            b.h.b.h l7 = dVar.l(i2);
            float f2 = this.l0;
            b.h.b.b m = dVar.m();
            m.f1823d.i(l6, -1.0f);
            m.f1823d.i(l7, f2);
            dVar.c(m);
        }
    }

    @Override // b.h.b.i.d
    public boolean e() {
        return true;
    }

    @Override // b.h.b.i.d
    public c i(c.a aVar) {
        switch (aVar.ordinal()) {
            case 0:
            case 5:
            case 6:
            case 7:
            case 8:
                return null;
            case 1:
            case 3:
                if (this.p0 == 1) {
                    return this.o0;
                }
                break;
            case 2:
            case 4:
                if (this.p0 == 0) {
                    return this.o0;
                }
                break;
        }
        throw new AssertionError(aVar.name());
    }

    @Override // b.h.b.i.d
    public boolean z() {
        return this.q0;
    }
}