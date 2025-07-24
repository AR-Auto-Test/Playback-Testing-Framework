package b.h.b.i.l;

/* compiled from: GuidelineReference.java */
/* loaded from: classes.dex */
public class i extends o {
    public i(b.h.b.i.d dVar) {
        super(dVar);
        dVar.f1878d.f();
        dVar.f1879e.f();
        this.f1933f = ((b.h.b.i.f) dVar).p0;
    }

    @Override // b.h.b.i.l.o, b.h.b.i.l.d
    public void a(d dVar) {
        f fVar = this.f1935h;
        if (fVar.f1905c && !fVar.j) {
            this.f1935h.c((int) ((fVar.l.get(0).f1909g * ((b.h.b.i.f) this.f1929b).l0) + 0.5f));
        }
    }

    @Override // b.h.b.i.l.o
    public void d() {
        b.h.b.i.d dVar = this.f1929b;
        b.h.b.i.f fVar = (b.h.b.i.f) dVar;
        int i = fVar.m0;
        int i2 = fVar.n0;
        if (fVar.p0 == 1) {
            if (i != -1) {
                this.f1935h.l.add(dVar.P.f1878d.f1935h);
                this.f1929b.P.f1878d.f1935h.k.add(this.f1935h);
                this.f1935h.f1908f = i;
            } else if (i2 != -1) {
                this.f1935h.l.add(dVar.P.f1878d.i);
                this.f1929b.P.f1878d.i.k.add(this.f1935h);
                this.f1935h.f1908f = -i2;
            } else {
                f fVar2 = this.f1935h;
                fVar2.f1904b = true;
                fVar2.l.add(dVar.P.f1878d.i);
                this.f1929b.P.f1878d.i.k.add(this.f1935h);
            }
            m(this.f1929b.f1878d.f1935h);
            m(this.f1929b.f1878d.i);
            return;
        }
        if (i != -1) {
            this.f1935h.l.add(dVar.P.f1879e.f1935h);
            this.f1929b.P.f1879e.f1935h.k.add(this.f1935h);
            this.f1935h.f1908f = i;
        } else if (i2 != -1) {
            this.f1935h.l.add(dVar.P.f1879e.i);
            this.f1929b.P.f1879e.i.k.add(this.f1935h);
            this.f1935h.f1908f = -i2;
        } else {
            f fVar3 = this.f1935h;
            fVar3.f1904b = true;
            fVar3.l.add(dVar.P.f1879e.i);
            this.f1929b.P.f1879e.i.k.add(this.f1935h);
        }
        m(this.f1929b.f1879e.f1935h);
        m(this.f1929b.f1879e.i);
    }

    @Override // b.h.b.i.l.o
    public void e() {
        b.h.b.i.d dVar = this.f1929b;
        if (((b.h.b.i.f) dVar).p0 == 1) {
            dVar.U = this.f1935h.f1909g;
        } else {
            dVar.V = this.f1935h.f1909g;
        }
    }

    @Override // b.h.b.i.l.o
    public void f() {
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