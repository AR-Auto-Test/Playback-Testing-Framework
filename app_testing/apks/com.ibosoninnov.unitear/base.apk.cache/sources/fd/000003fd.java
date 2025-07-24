package b.h.b.i.l;

import androidx.constraintlayout.widget.ConstraintLayout;
import b.h.b.i.l.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Objects;

/* compiled from: DependencyGraph.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public b.h.b.i.e f1895a;

    /* renamed from: d  reason: collision with root package name */
    public b.h.b.i.e f1898d;

    /* renamed from: f  reason: collision with root package name */
    public b.InterfaceC0029b f1900f;

    /* renamed from: g  reason: collision with root package name */
    public b.a f1901g;

    /* renamed from: h  reason: collision with root package name */
    public ArrayList<l> f1902h;

    /* renamed from: b  reason: collision with root package name */
    public boolean f1896b = true;

    /* renamed from: c  reason: collision with root package name */
    public boolean f1897c = true;

    /* renamed from: e  reason: collision with root package name */
    public ArrayList<o> f1899e = new ArrayList<>();

    public e(b.h.b.i.e eVar) {
        new ArrayList();
        this.f1900f = null;
        this.f1901g = new b.a();
        this.f1902h = new ArrayList<>();
        this.f1895a = eVar;
        this.f1898d = eVar;
    }

    public final void a(f fVar, int i, int i2, f fVar2, ArrayList<l> arrayList, l lVar) {
        o oVar = fVar.f1906d;
        if (oVar.f1930c == null) {
            b.h.b.i.e eVar = this.f1895a;
            if (oVar == eVar.f1878d || oVar == eVar.f1879e) {
                return;
            }
            if (lVar == null) {
                lVar = new l(oVar, i2);
                arrayList.add(lVar);
            }
            oVar.f1930c = lVar;
            lVar.f1921c.add(oVar);
            for (d dVar : oVar.f1935h.k) {
                if (dVar instanceof f) {
                    a((f) dVar, i, 0, fVar2, arrayList, lVar);
                }
            }
            for (d dVar2 : oVar.i.k) {
                if (dVar2 instanceof f) {
                    a((f) dVar2, i, 1, fVar2, arrayList, lVar);
                }
            }
            if (i == 1 && (oVar instanceof m)) {
                for (d dVar3 : ((m) oVar).k.k) {
                    if (dVar3 instanceof f) {
                        a((f) dVar3, i, 2, fVar2, arrayList, lVar);
                    }
                }
            }
            for (f fVar3 : oVar.f1935h.l) {
                a(fVar3, i, 0, fVar2, arrayList, lVar);
            }
            for (f fVar4 : oVar.i.l) {
                a(fVar4, i, 1, fVar2, arrayList, lVar);
            }
            if (i == 1 && (oVar instanceof m)) {
                for (f fVar5 : ((m) oVar).k.l) {
                    a(fVar5, i, 2, fVar2, arrayList, lVar);
                }
            }
        }
    }

    public final boolean b(b.h.b.i.e eVar) {
        int i;
        int i2;
        int i3;
        int i4;
        Iterator<b.h.b.i.d> it = eVar.l0.iterator();
        while (it.hasNext()) {
            b.h.b.i.d next = it.next();
            int[] iArr = next.O;
            int i5 = iArr[0];
            int i6 = iArr[1];
            if (next.c0 == 8) {
                next.f1875a = true;
            } else {
                float f2 = next.q;
                if (f2 < 1.0f && i5 == 3) {
                    next.l = 2;
                }
                float f3 = next.t;
                if (f3 < 1.0f && i6 == 3) {
                    next.m = 2;
                }
                if (next.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    if (i5 == 3 && (i6 == 2 || i6 == 1)) {
                        next.l = 3;
                    } else if (i6 == 3 && (i5 == 2 || i5 == 1)) {
                        next.m = 3;
                    } else if (i5 == 3 && i6 == 3) {
                        if (next.l == 0) {
                            next.l = 3;
                        }
                        if (next.m == 0) {
                            next.m = 3;
                        }
                    }
                }
                int i7 = (i5 == 3 && next.l == 1 && (next.D.f1865f == null || next.F.f1865f == null)) ? 2 : i5;
                int i8 = (i6 == 3 && next.m == 1 && (next.E.f1865f == null || next.G.f1865f == null)) ? 2 : i6;
                k kVar = next.f1878d;
                kVar.f1931d = i7;
                int i9 = next.l;
                kVar.f1928a = i9;
                m mVar = next.f1879e;
                mVar.f1931d = i8;
                int i10 = next.m;
                mVar.f1928a = i10;
                if ((i7 == 4 || i7 == 1 || i7 == 2) && (i8 == 4 || i8 == 1 || i8 == 2)) {
                    int r = next.r();
                    if (i7 == 4) {
                        i = (eVar.r() - next.D.f1866g) - next.F.f1866g;
                        i2 = 1;
                    } else {
                        i = r;
                        i2 = i7;
                    }
                    int l = next.l();
                    if (i8 == 4) {
                        i3 = (eVar.l() - next.E.f1866g) - next.G.f1866g;
                        i4 = 1;
                    } else {
                        i3 = l;
                        i4 = i8;
                    }
                    f(next, i2, i, i4, i3);
                    next.f1878d.f1932e.c(next.r());
                    next.f1879e.f1932e.c(next.l());
                    next.f1875a = true;
                } else {
                    if (i7 == 3 && (i8 == 2 || i8 == 1)) {
                        if (i9 == 3) {
                            if (i8 == 2) {
                                f(next, 2, 0, 2, 0);
                            }
                            int l2 = next.l();
                            f(next, 1, (int) ((l2 * next.S) + 0.5f), 1, l2);
                            next.f1878d.f1932e.c(next.r());
                            next.f1879e.f1932e.c(next.l());
                            next.f1875a = true;
                        } else if (i9 == 1) {
                            f(next, 2, 0, i8, 0);
                            next.f1878d.f1932e.m = next.r();
                        } else if (i9 == 2) {
                            int[] iArr2 = eVar.O;
                            if (iArr2[0] == 1 || iArr2[0] == 4) {
                                f(next, 1, (int) ((f2 * eVar.r()) + 0.5f), i8, next.l());
                                next.f1878d.f1932e.c(next.r());
                                next.f1879e.f1932e.c(next.l());
                                next.f1875a = true;
                            }
                        } else {
                            b.h.b.i.c[] cVarArr = next.L;
                            if (cVarArr[0].f1865f == null || cVarArr[1].f1865f == null) {
                                f(next, 2, 0, i8, 0);
                                next.f1878d.f1932e.c(next.r());
                                next.f1879e.f1932e.c(next.l());
                                next.f1875a = true;
                            }
                        }
                    }
                    if (i8 == 3 && (i7 == 2 || i7 == 1)) {
                        if (i10 == 3) {
                            if (i7 == 2) {
                                f(next, 2, 0, 2, 0);
                            }
                            int r2 = next.r();
                            float f4 = next.S;
                            if (next.T == -1) {
                                f4 = 1.0f / f4;
                            }
                            f(next, 1, r2, 1, (int) ((r2 * f4) + 0.5f));
                            next.f1878d.f1932e.c(next.r());
                            next.f1879e.f1932e.c(next.l());
                            next.f1875a = true;
                        } else if (i10 == 1) {
                            f(next, i7, 0, 2, 0);
                            next.f1879e.f1932e.m = next.l();
                        } else if (i10 == 2) {
                            int[] iArr3 = eVar.O;
                            if (iArr3[1] == 1 || iArr3[1] == 4) {
                                f(next, i7, next.r(), 1, (int) ((f3 * eVar.l()) + 0.5f));
                                next.f1878d.f1932e.c(next.r());
                                next.f1879e.f1932e.c(next.l());
                                next.f1875a = true;
                            }
                        } else {
                            b.h.b.i.c[] cVarArr2 = next.L;
                            if (cVarArr2[2].f1865f == null || cVarArr2[3].f1865f == null) {
                                f(next, 2, 0, i8, 0);
                                next.f1878d.f1932e.c(next.r());
                                next.f1879e.f1932e.c(next.l());
                                next.f1875a = true;
                            }
                        }
                    }
                    if (i7 == 3 && i8 == 3) {
                        if (i9 == 1 || i10 == 1) {
                            f(next, 2, 0, 2, 0);
                            next.f1878d.f1932e.m = next.r();
                            next.f1879e.f1932e.m = next.l();
                        } else if (i10 == 2 && i9 == 2) {
                            int[] iArr4 = eVar.O;
                            if (iArr4[0] == 1 || iArr4[0] == 1) {
                                if (iArr4[1] == 1 || iArr4[1] == 1) {
                                    f(next, 1, (int) ((f2 * eVar.r()) + 0.5f), 1, (int) ((f3 * eVar.l()) + 0.5f));
                                    next.f1878d.f1932e.c(next.r());
                                    next.f1879e.f1932e.c(next.l());
                                    next.f1875a = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    public void c() {
        ArrayList<o> arrayList = this.f1899e;
        arrayList.clear();
        this.f1898d.f1878d.f();
        this.f1898d.f1879e.f();
        arrayList.add(this.f1898d.f1878d);
        arrayList.add(this.f1898d.f1879e);
        Iterator<b.h.b.i.d> it = this.f1898d.l0.iterator();
        HashSet hashSet = null;
        while (it.hasNext()) {
            b.h.b.i.d next = it.next();
            if (next instanceof b.h.b.i.f) {
                arrayList.add(new i(next));
            } else {
                if (next.w()) {
                    if (next.f1876b == null) {
                        next.f1876b = new c(next, 0);
                    }
                    if (hashSet == null) {
                        hashSet = new HashSet();
                    }
                    hashSet.add(next.f1876b);
                } else {
                    arrayList.add(next.f1878d);
                }
                if (next.x()) {
                    if (next.f1877c == null) {
                        next.f1877c = new c(next, 1);
                    }
                    if (hashSet == null) {
                        hashSet = new HashSet();
                    }
                    hashSet.add(next.f1877c);
                } else {
                    arrayList.add(next.f1879e);
                }
                if (next instanceof b.h.b.i.h) {
                    arrayList.add(new j(next));
                }
            }
        }
        if (hashSet != null) {
            arrayList.addAll(hashSet);
        }
        Iterator<o> it2 = arrayList.iterator();
        while (it2.hasNext()) {
            it2.next().f();
        }
        Iterator<o> it3 = arrayList.iterator();
        while (it3.hasNext()) {
            o next2 = it3.next();
            if (next2.f1929b != this.f1898d) {
                next2.d();
            }
        }
        this.f1902h.clear();
        l.f1919a = 0;
        e(this.f1895a.f1878d, 0, this.f1902h);
        e(this.f1895a.f1879e, 1, this.f1902h);
        this.f1896b = false;
    }

    /* JADX WARN: Removed duplicated region for block: B:17:0x003a  */
    /* JADX WARN: Removed duplicated region for block: B:18:0x003d  */
    /* JADX WARN: Removed duplicated region for block: B:21:0x0043  */
    /* JADX WARN: Removed duplicated region for block: B:22:0x0046  */
    /* JADX WARN: Removed duplicated region for block: B:25:0x0064 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:45:0x00d5  */
    /* JADX WARN: Removed duplicated region for block: B:46:0x00ed  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final int d(b.h.b.i.e eVar, int i) {
        boolean contains;
        o oVar;
        f fVar;
        f fVar2;
        float f2;
        e eVar2 = this;
        b.h.b.i.e eVar3 = eVar;
        int size = eVar2.f1902h.size();
        long j = 0;
        int i2 = 0;
        long j2 = 0;
        while (i2 < size) {
            l lVar = eVar2.f1902h.get(i2);
            o oVar2 = lVar.f1920b;
            if (oVar2 instanceof c) {
                if (((c) oVar2).f1933f != i) {
                    j = Math.max(j, j2);
                    i2++;
                    j2 = 0;
                    eVar2 = this;
                    eVar3 = eVar;
                }
                f fVar3 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).f1935h;
                f fVar4 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).i;
                contains = oVar2.f1935h.l.contains(fVar3);
                boolean contains2 = lVar.f1920b.i.l.contains(fVar4);
                long j3 = lVar.f1920b.j();
                if (!contains && contains2) {
                    long b2 = lVar.b(lVar.f1920b.f1935h, j2);
                    long a2 = lVar.a(lVar.f1920b.i, j2);
                    long j4 = b2 - j3;
                    o oVar3 = lVar.f1920b;
                    int i3 = oVar3.i.f1908f;
                    if (j4 >= (-i3)) {
                        j4 += i3;
                    }
                    long j5 = oVar3.f1935h.f1908f;
                    long j6 = ((-a2) - j3) - j5;
                    if (j6 >= j5) {
                        j6 -= j5;
                    }
                    b.h.b.i.d dVar = oVar3.f1929b;
                    Objects.requireNonNull(dVar);
                    if (i == 0) {
                        f2 = dVar.Z;
                    } else {
                        f2 = i == 1 ? dVar.a0 : -1.0f;
                    }
                    float f3 = (float) (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? (((float) j4) / (1.0f - f2)) + (((float) j6) / f2) : 0L);
                    long a3 = (f3 * f2) + 0.5f + j3 + c.b.a.a.a.a(1.0f, f2, f3, 0.5f);
                    o oVar4 = lVar.f1920b;
                    j2 = (oVar4.f1935h.f1908f + a3) - oVar4.i.f1908f;
                } else if (contains) {
                    j2 = Math.max(lVar.b(lVar.f1920b.f1935h, fVar2.f1908f), lVar.f1920b.f1935h.f1908f + j3);
                } else if (contains2) {
                    j2 = Math.max(-lVar.a(lVar.f1920b.i, fVar.f1908f), (-lVar.f1920b.i.f1908f) + j3);
                } else {
                    j2 = (lVar.f1920b.j() + oVar.f1935h.f1908f) - lVar.f1920b.i.f1908f;
                }
                j = Math.max(j, j2);
                i2++;
                j2 = 0;
                eVar2 = this;
                eVar3 = eVar;
            } else if (i == 0) {
                if (!(oVar2 instanceof k)) {
                    j = Math.max(j, j2);
                    i2++;
                    j2 = 0;
                    eVar2 = this;
                    eVar3 = eVar;
                }
                f fVar32 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).f1935h;
                f fVar42 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).i;
                contains = oVar2.f1935h.l.contains(fVar32);
                boolean contains22 = lVar.f1920b.i.l.contains(fVar42);
                long j32 = lVar.f1920b.j();
                if (!contains) {
                }
                if (contains) {
                }
                j = Math.max(j, j2);
                i2++;
                j2 = 0;
                eVar2 = this;
                eVar3 = eVar;
            } else {
                if (!(oVar2 instanceof m)) {
                    j = Math.max(j, j2);
                    i2++;
                    j2 = 0;
                    eVar2 = this;
                    eVar3 = eVar;
                }
                f fVar322 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).f1935h;
                f fVar422 = (i == 0 ? eVar3.f1878d : eVar3.f1879e).i;
                contains = oVar2.f1935h.l.contains(fVar322);
                boolean contains222 = lVar.f1920b.i.l.contains(fVar422);
                long j322 = lVar.f1920b.j();
                if (!contains) {
                }
                if (contains) {
                }
                j = Math.max(j, j2);
                i2++;
                j2 = 0;
                eVar2 = this;
                eVar3 = eVar;
            }
        }
        return (int) j;
    }

    public final void e(o oVar, int i, ArrayList<l> arrayList) {
        for (d dVar : oVar.f1935h.k) {
            if (dVar instanceof f) {
                a((f) dVar, i, 0, oVar.i, arrayList, null);
            } else if (dVar instanceof o) {
                a(((o) dVar).f1935h, i, 0, oVar.i, arrayList, null);
            }
        }
        for (d dVar2 : oVar.i.k) {
            if (dVar2 instanceof f) {
                a((f) dVar2, i, 1, oVar.f1935h, arrayList, null);
            } else if (dVar2 instanceof o) {
                a(((o) dVar2).i, i, 1, oVar.f1935h, arrayList, null);
            }
        }
        if (i == 1) {
            for (d dVar3 : ((m) oVar).k.k) {
                if (dVar3 instanceof f) {
                    a((f) dVar3, i, 2, null, arrayList, null);
                }
            }
        }
    }

    public final void f(b.h.b.i.d dVar, int i, int i2, int i3, int i4) {
        b.a aVar = this.f1901g;
        aVar.f1887a = i;
        aVar.f1888b = i3;
        aVar.f1889c = i2;
        aVar.f1890d = i4;
        ((ConstraintLayout.b) this.f1900f).b(dVar, aVar);
        dVar.M(this.f1901g.f1891e);
        dVar.H(this.f1901g.f1892f);
        b.a aVar2 = this.f1901g;
        dVar.y = aVar2.f1894h;
        dVar.E(aVar2.f1893g);
    }

    public void g() {
        g gVar;
        Iterator<b.h.b.i.d> it = this.f1895a.l0.iterator();
        while (it.hasNext()) {
            b.h.b.i.d next = it.next();
            if (!next.f1875a) {
                int[] iArr = next.O;
                boolean z = false;
                int i = iArr[0];
                int i2 = iArr[1];
                int i3 = next.l;
                int i4 = next.m;
                boolean z2 = i == 2 || (i == 3 && i3 == 1);
                if (i2 == 2 || (i2 == 3 && i4 == 1)) {
                    z = true;
                }
                g gVar2 = next.f1878d.f1932e;
                boolean z3 = gVar2.j;
                g gVar3 = next.f1879e.f1932e;
                boolean z4 = gVar3.j;
                if (z3 && z4) {
                    f(next, 1, gVar2.f1909g, 1, gVar3.f1909g);
                    next.f1875a = true;
                } else if (z3 && z) {
                    f(next, 1, gVar2.f1909g, 2, gVar3.f1909g);
                    if (i2 == 3) {
                        next.f1879e.f1932e.m = next.l();
                    } else {
                        next.f1879e.f1932e.c(next.l());
                        next.f1875a = true;
                    }
                } else if (z4 && z2) {
                    f(next, 2, gVar2.f1909g, 1, gVar3.f1909g);
                    if (i == 3) {
                        next.f1878d.f1932e.m = next.r();
                    } else {
                        next.f1878d.f1932e.c(next.r());
                        next.f1875a = true;
                    }
                }
                if (next.f1875a && (gVar = next.f1879e.l) != null) {
                    gVar.c(next.W);
                }
            }
        }
    }
}