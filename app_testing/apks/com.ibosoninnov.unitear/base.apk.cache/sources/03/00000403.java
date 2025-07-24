package b.h.b.i.l;

import b.h.b.i.l.f;

/* compiled from: HelperReferences.java */
/* loaded from: classes.dex */
public class j extends o {
    public j(b.h.b.i.d dVar) {
        super(dVar);
    }

    @Override // b.h.b.i.l.o, b.h.b.i.l.d
    public void a(d dVar) {
        b.h.b.i.a aVar = (b.h.b.i.a) this.f1929b;
        int i = aVar.n0;
        int i2 = 0;
        int i3 = -1;
        for (f fVar : this.f1935h.l) {
            int i4 = fVar.f1909g;
            if (i3 == -1 || i4 < i3) {
                i3 = i4;
            }
            if (i2 < i4) {
                i2 = i4;
            }
        }
        if (i != 0 && i != 2) {
            this.f1935h.c(i2 + aVar.p0);
        } else {
            this.f1935h.c(i3 + aVar.p0);
        }
    }

    @Override // b.h.b.i.l.o
    public void d() {
        b.h.b.i.d dVar = this.f1929b;
        if (dVar instanceof b.h.b.i.a) {
            f fVar = this.f1935h;
            fVar.f1904b = true;
            b.h.b.i.a aVar = (b.h.b.i.a) dVar;
            int i = aVar.n0;
            boolean z = aVar.o0;
            int i2 = 0;
            if (i == 0) {
                fVar.f1907e = f.a.LEFT;
                while (i2 < aVar.m0) {
                    b.h.b.i.d dVar2 = aVar.l0[i2];
                    if (z || dVar2.c0 != 8) {
                        f fVar2 = dVar2.f1878d.f1935h;
                        fVar2.k.add(this.f1935h);
                        this.f1935h.l.add(fVar2);
                    }
                    i2++;
                }
                m(this.f1929b.f1878d.f1935h);
                m(this.f1929b.f1878d.i);
            } else if (i == 1) {
                fVar.f1907e = f.a.RIGHT;
                while (i2 < aVar.m0) {
                    b.h.b.i.d dVar3 = aVar.l0[i2];
                    if (z || dVar3.c0 != 8) {
                        f fVar3 = dVar3.f1878d.i;
                        fVar3.k.add(this.f1935h);
                        this.f1935h.l.add(fVar3);
                    }
                    i2++;
                }
                m(this.f1929b.f1878d.f1935h);
                m(this.f1929b.f1878d.i);
            } else if (i == 2) {
                fVar.f1907e = f.a.TOP;
                while (i2 < aVar.m0) {
                    b.h.b.i.d dVar4 = aVar.l0[i2];
                    if (z || dVar4.c0 != 8) {
                        f fVar4 = dVar4.f1879e.f1935h;
                        fVar4.k.add(this.f1935h);
                        this.f1935h.l.add(fVar4);
                    }
                    i2++;
                }
                m(this.f1929b.f1879e.f1935h);
                m(this.f1929b.f1879e.i);
            } else if (i != 3) {
            } else {
                fVar.f1907e = f.a.BOTTOM;
                while (i2 < aVar.m0) {
                    b.h.b.i.d dVar5 = aVar.l0[i2];
                    if (z || dVar5.c0 != 8) {
                        f fVar5 = dVar5.f1879e.i;
                        fVar5.k.add(this.f1935h);
                        this.f1935h.l.add(fVar5);
                    }
                    i2++;
                }
                m(this.f1929b.f1879e.f1935h);
                m(this.f1929b.f1879e.i);
            }
        }
    }

    @Override // b.h.b.i.l.o
    public void e() {
        b.h.b.i.d dVar = this.f1929b;
        if (dVar instanceof b.h.b.i.a) {
            int i = ((b.h.b.i.a) dVar).n0;
            if (i != 0 && i != 1) {
                dVar.V = this.f1935h.f1909g;
            } else {
                dVar.U = this.f1935h.f1909g;
            }
        }
    }

    @Override // b.h.b.i.l.o
    public void f() {
        this.f1930c = null;
        this.f1935h.b();
    }

    @Override // b.h.b.i.l.o
    public boolean k() {
        return false;
    }

    public final void m(f fVar) {
        this.f1935h.k.add(fVar);
        fVar.l.add(this.f1935h);
    }
}