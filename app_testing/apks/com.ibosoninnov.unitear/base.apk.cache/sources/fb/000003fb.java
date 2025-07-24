package b.h.b.i.l;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Iterator;

/* compiled from: ChainRun.java */
/* loaded from: classes.dex */
public class c extends o {
    public ArrayList<o> k;
    public int l;

    public c(b.h.b.i.d dVar, int i) {
        super(dVar);
        b.h.b.i.d dVar2;
        o oVar;
        int i2;
        o oVar2;
        this.k = new ArrayList<>();
        this.f1933f = i;
        b.h.b.i.d dVar3 = this.f1929b;
        b.h.b.i.d o = dVar3.o(i);
        while (true) {
            b.h.b.i.d dVar4 = o;
            dVar2 = dVar3;
            dVar3 = dVar4;
            if (dVar3 == null) {
                break;
            }
            o = dVar3.o(this.f1933f);
        }
        this.f1929b = dVar2;
        ArrayList<o> arrayList = this.k;
        int i3 = this.f1933f;
        if (i3 == 0) {
            oVar = dVar2.f1878d;
        } else {
            oVar = i3 == 1 ? dVar2.f1879e : null;
        }
        arrayList.add(oVar);
        b.h.b.i.d n = dVar2.n(this.f1933f);
        while (n != null) {
            ArrayList<o> arrayList2 = this.k;
            int i4 = this.f1933f;
            if (i4 == 0) {
                oVar2 = n.f1878d;
            } else {
                oVar2 = i4 == 1 ? n.f1879e : null;
            }
            arrayList2.add(oVar2);
            n = n.n(this.f1933f);
        }
        Iterator<o> it = this.k.iterator();
        while (it.hasNext()) {
            o next = it.next();
            int i5 = this.f1933f;
            if (i5 == 0) {
                next.f1929b.f1876b = this;
            } else if (i5 == 1) {
                next.f1929b.f1877c = this;
            }
        }
        if ((this.f1933f == 0 && ((b.h.b.i.e) this.f1929b.P).p0) && this.k.size() > 1) {
            ArrayList<o> arrayList3 = this.k;
            this.f1929b = arrayList3.get(arrayList3.size() - 1).f1929b;
        }
        if (this.f1933f == 0) {
            i2 = this.f1929b.e0;
        } else {
            i2 = this.f1929b.f0;
        }
        this.l = i2;
    }

    /* JADX WARN: Code restructure failed: missing block: B:280:0x03e9, code lost:
        r9 = r9 - r10;
     */
    /* JADX WARN: Removed duplicated region for block: B:64:0x00c7  */
    /* JADX WARN: Removed duplicated region for block: B:67:0x00d7  */
    @Override // b.h.b.i.l.o, b.h.b.i.l.d
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void a(d dVar) {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        float f2;
        boolean z;
        int i6;
        int i7;
        int i8;
        int i9;
        float f3;
        int i10;
        boolean z2;
        int i11;
        float f4;
        int i12;
        int i13;
        int i14;
        int i15;
        int i16;
        f fVar = this.f1935h;
        if (fVar.j) {
            f fVar2 = this.i;
            if (fVar2.j) {
                b.h.b.i.d dVar2 = this.f1929b.P;
                boolean z3 = (dVar2 == null || !(dVar2 instanceof b.h.b.i.e)) ? false : ((b.h.b.i.e) dVar2).p0;
                int i17 = fVar2.f1909g - fVar.f1909g;
                int size = this.k.size();
                int i18 = 0;
                while (true) {
                    i = -1;
                    i2 = 8;
                    if (i18 >= size) {
                        i18 = -1;
                        break;
                    } else if (this.k.get(i18).f1929b.c0 != 8) {
                        break;
                    } else {
                        i18++;
                    }
                }
                int i19 = size - 1;
                int i20 = i19;
                while (true) {
                    if (i20 < 0) {
                        break;
                    }
                    if (this.k.get(i20).f1929b.c0 != 8) {
                        i = i20;
                        break;
                    }
                    i20--;
                }
                int i21 = 0;
                while (true) {
                    int i22 = 3;
                    if (i21 >= 2) {
                        i3 = 0;
                        i4 = 0;
                        i5 = 0;
                        f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        break;
                    }
                    int i23 = 0;
                    i5 = 0;
                    i13 = 0;
                    i14 = 0;
                    f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    while (i23 < size) {
                        o oVar = this.k.get(i23);
                        b.h.b.i.d dVar3 = oVar.f1929b;
                        if (dVar3.c0 != i2) {
                            i14++;
                            if (i23 > 0 && i23 >= i18) {
                                i5 += oVar.f1935h.f1908f;
                            }
                            g gVar = oVar.f1932e;
                            int i24 = gVar.f1909g;
                            boolean z4 = oVar.f1931d != i22;
                            if (z4) {
                                int i25 = this.f1933f;
                                if (i25 == 0 && !dVar3.f1878d.f1932e.j) {
                                    return;
                                }
                                if (i25 == 1 && !dVar3.f1879e.f1932e.j) {
                                    return;
                                }
                                i15 = i24;
                            } else {
                                i15 = i24;
                                if (oVar.f1928a == 1 && i21 == 0) {
                                    i16 = gVar.m;
                                    i13++;
                                } else if (gVar.j) {
                                    i16 = i15;
                                }
                                z4 = true;
                                if (z4) {
                                    i13++;
                                    float f5 = dVar3.g0[this.f1933f];
                                    if (f5 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                        f2 += f5;
                                    }
                                } else {
                                    i5 += i16;
                                }
                                if (i23 < i19 && i23 < i) {
                                    i5 += -oVar.i.f1908f;
                                }
                            }
                            i16 = i15;
                            if (z4) {
                            }
                            if (i23 < i19) {
                                i5 += -oVar.i.f1908f;
                            }
                        }
                        i23++;
                        i2 = 8;
                        i22 = 3;
                    }
                    if (i5 < i17 || i13 == 0) {
                        break;
                    }
                    i21++;
                    i2 = 8;
                }
                i3 = i13;
                i4 = i14;
                int i26 = this.f1935h.f1909g;
                if (z3) {
                    i26 = this.i.f1909g;
                }
                if (i5 > i17) {
                    int i27 = (int) (((i5 - i17) / 2.0f) + 0.5f);
                    i26 = z3 ? i26 + i27 : i26 - i27;
                }
                if (i3 > 0) {
                    float f6 = i17 - i5;
                    int i28 = (int) ((f6 / i3) + 0.5f);
                    int i29 = 0;
                    int i30 = 0;
                    while (i29 < size) {
                        o oVar2 = this.k.get(i29);
                        int i31 = i28;
                        b.h.b.i.d dVar4 = oVar2.f1929b;
                        int i32 = i5;
                        int i33 = i26;
                        if (dVar4.c0 != 8 && oVar2.f1931d == 3) {
                            g gVar2 = oVar2.f1932e;
                            if (!gVar2.j) {
                                if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    z2 = z3;
                                    i12 = (int) (((dVar4.g0[this.f1933f] * f6) / f2) + 0.5f);
                                } else {
                                    z2 = z3;
                                    i12 = i31;
                                }
                                if (this.f1933f == 0) {
                                    int i34 = dVar4.p;
                                    f4 = f6;
                                    i11 = i4;
                                    int max = Math.max(dVar4.o, oVar2.f1928a == 1 ? Math.min(i12, gVar2.m) : i12);
                                    if (i34 > 0) {
                                        max = Math.min(i34, max);
                                    }
                                    if (max != i12) {
                                        i30++;
                                        i12 = max;
                                    }
                                } else {
                                    i11 = i4;
                                    f4 = f6;
                                    int i35 = dVar4.s;
                                    int max2 = Math.max(dVar4.r, oVar2.f1928a == 1 ? Math.min(i12, gVar2.m) : i12);
                                    if (i35 > 0) {
                                        max2 = Math.min(i35, max2);
                                    }
                                    if (max2 != i12) {
                                        i30++;
                                        i12 = max2;
                                    }
                                }
                                oVar2.f1932e.c(i12);
                                i29++;
                                i28 = i31;
                                i5 = i32;
                                i26 = i33;
                                z3 = z2;
                                f6 = f4;
                                i4 = i11;
                            }
                        }
                        z2 = z3;
                        i11 = i4;
                        f4 = f6;
                        i29++;
                        i28 = i31;
                        i5 = i32;
                        i26 = i33;
                        z3 = z2;
                        f6 = f4;
                        i4 = i11;
                    }
                    z = z3;
                    i6 = i4;
                    i7 = i26;
                    int i36 = i5;
                    if (i30 > 0) {
                        i3 -= i30;
                        int i37 = 0;
                        for (int i38 = 0; i38 < size; i38++) {
                            o oVar3 = this.k.get(i38);
                            if (oVar3.f1929b.c0 != 8) {
                                if (i38 > 0 && i38 >= i18) {
                                    i37 += oVar3.f1935h.f1908f;
                                }
                                i37 += oVar3.f1932e.f1909g;
                                if (i38 < i19 && i38 < i) {
                                    i37 += -oVar3.i.f1908f;
                                }
                            }
                        }
                        i5 = i37;
                    } else {
                        i5 = i36;
                    }
                    i9 = 2;
                    if (this.l == 2 && i30 == 0) {
                        i8 = 0;
                        this.l = 0;
                    } else {
                        i8 = 0;
                    }
                } else {
                    z = z3;
                    i6 = i4;
                    i7 = i26;
                    i8 = 0;
                    i9 = 2;
                }
                if (i5 > i17) {
                    this.l = i9;
                }
                if (i6 > 0 && i3 == 0 && i18 == i) {
                    this.l = i9;
                }
                int i39 = this.l;
                if (i39 == 1) {
                    int i40 = i6;
                    if (i40 > 1) {
                        i10 = (i17 - i5) / (i40 - 1);
                    } else {
                        i10 = i40 == 1 ? (i17 - i5) / 2 : i8;
                    }
                    if (i3 > 0) {
                        i10 = i8;
                    }
                    int i41 = i7;
                    for (int i42 = i8; i42 < size; i42++) {
                        o oVar4 = this.k.get(z ? size - (i42 + 1) : i42);
                        if (oVar4.f1929b.c0 == 8) {
                            oVar4.f1935h.c(i41);
                            oVar4.i.c(i41);
                        } else {
                            if (i42 > 0) {
                                i41 = z ? i41 - i10 : i41 + i10;
                            }
                            if (i42 > 0 && i42 >= i18) {
                                if (z) {
                                    i41 -= oVar4.f1935h.f1908f;
                                } else {
                                    i41 += oVar4.f1935h.f1908f;
                                }
                            }
                            if (z) {
                                oVar4.i.c(i41);
                            } else {
                                oVar4.f1935h.c(i41);
                            }
                            g gVar3 = oVar4.f1932e;
                            int i43 = gVar3.f1909g;
                            if (oVar4.f1931d == 3 && oVar4.f1928a == 1) {
                                i43 = gVar3.m;
                            }
                            i41 = z ? i41 - i43 : i41 + i43;
                            if (z) {
                                oVar4.f1935h.c(i41);
                            } else {
                                oVar4.i.c(i41);
                            }
                            oVar4.f1934g = true;
                            if (i42 < i19 && i42 < i) {
                                if (z) {
                                    i41 -= -oVar4.i.f1908f;
                                } else {
                                    i41 += -oVar4.i.f1908f;
                                }
                            }
                        }
                    }
                    return;
                }
                int i44 = i6;
                if (i39 == 0) {
                    int i45 = (i17 - i5) / (i44 + 1);
                    if (i3 > 0) {
                        i45 = i8;
                    }
                    int i46 = i7;
                    for (int i47 = i8; i47 < size; i47++) {
                        o oVar5 = this.k.get(z ? size - (i47 + 1) : i47);
                        if (oVar5.f1929b.c0 == 8) {
                            oVar5.f1935h.c(i46);
                            oVar5.i.c(i46);
                        } else {
                            int i48 = z ? i46 - i45 : i46 + i45;
                            if (i47 > 0 && i47 >= i18) {
                                if (z) {
                                    i48 -= oVar5.f1935h.f1908f;
                                } else {
                                    i48 += oVar5.f1935h.f1908f;
                                }
                            }
                            if (z) {
                                oVar5.i.c(i48);
                            } else {
                                oVar5.f1935h.c(i48);
                            }
                            g gVar4 = oVar5.f1932e;
                            int i49 = gVar4.f1909g;
                            if (oVar5.f1931d == 3 && oVar5.f1928a == 1) {
                                i49 = Math.min(i49, gVar4.m);
                            }
                            i46 = z ? i48 - i49 : i48 + i49;
                            if (z) {
                                oVar5.f1935h.c(i46);
                            } else {
                                oVar5.i.c(i46);
                            }
                            if (i47 < i19 && i47 < i) {
                                if (z) {
                                    i46 -= -oVar5.i.f1908f;
                                } else {
                                    i46 += -oVar5.i.f1908f;
                                }
                            }
                        }
                    }
                } else if (i39 == 2) {
                    if (this.f1933f == 0) {
                        f3 = this.f1929b.Z;
                    } else {
                        f3 = this.f1929b.a0;
                    }
                    if (z) {
                        f3 = 1.0f - f3;
                    }
                    int i50 = (int) (((i17 - i5) * f3) + 0.5f);
                    if (i50 < 0 || i3 > 0) {
                        i50 = i8;
                    }
                    int i51 = z ? i7 - i50 : i7 + i50;
                    for (int i52 = i8; i52 < size; i52++) {
                        o oVar6 = this.k.get(z ? size - (i52 + 1) : i52);
                        if (oVar6.f1929b.c0 == 8) {
                            oVar6.f1935h.c(i51);
                            oVar6.i.c(i51);
                        } else {
                            if (i52 > 0 && i52 >= i18) {
                                if (z) {
                                    i51 -= oVar6.f1935h.f1908f;
                                } else {
                                    i51 += oVar6.f1935h.f1908f;
                                }
                            }
                            if (z) {
                                oVar6.i.c(i51);
                            } else {
                                oVar6.f1935h.c(i51);
                            }
                            g gVar5 = oVar6.f1932e;
                            int i53 = gVar5.f1909g;
                            if (oVar6.f1931d == 3 && oVar6.f1928a == 1) {
                                i53 = gVar5.m;
                            }
                            i51 += i53;
                            if (z) {
                                oVar6.f1935h.c(i51);
                            } else {
                                oVar6.i.c(i51);
                            }
                            if (i52 < i19 && i52 < i) {
                                if (z) {
                                    i51 -= -oVar6.i.f1908f;
                                } else {
                                    i51 += -oVar6.i.f1908f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Override // b.h.b.i.l.o
    public void d() {
        Iterator<o> it = this.k.iterator();
        while (it.hasNext()) {
            it.next().d();
        }
        int size = this.k.size();
        if (size < 1) {
            return;
        }
        b.h.b.i.d dVar = this.k.get(0).f1929b;
        b.h.b.i.d dVar2 = this.k.get(size - 1).f1929b;
        if (this.f1933f == 0) {
            b.h.b.i.c cVar = dVar.D;
            b.h.b.i.c cVar2 = dVar2.F;
            f i = i(cVar, 0);
            int d2 = cVar.d();
            b.h.b.i.d m = m();
            if (m != null) {
                d2 = m.D.d();
            }
            if (i != null) {
                f fVar = this.f1935h;
                fVar.l.add(i);
                fVar.f1908f = d2;
                i.k.add(fVar);
            }
            f i2 = i(cVar2, 0);
            int d3 = cVar2.d();
            b.h.b.i.d n = n();
            if (n != null) {
                d3 = n.F.d();
            }
            if (i2 != null) {
                f fVar2 = this.i;
                fVar2.l.add(i2);
                fVar2.f1908f = -d3;
                i2.k.add(fVar2);
            }
        } else {
            b.h.b.i.c cVar3 = dVar.E;
            b.h.b.i.c cVar4 = dVar2.G;
            f i3 = i(cVar3, 1);
            int d4 = cVar3.d();
            b.h.b.i.d m2 = m();
            if (m2 != null) {
                d4 = m2.E.d();
            }
            if (i3 != null) {
                f fVar3 = this.f1935h;
                fVar3.l.add(i3);
                fVar3.f1908f = d4;
                i3.k.add(fVar3);
            }
            f i4 = i(cVar4, 1);
            int d5 = cVar4.d();
            b.h.b.i.d n2 = n();
            if (n2 != null) {
                d5 = n2.G.d();
            }
            if (i4 != null) {
                f fVar4 = this.i;
                fVar4.l.add(i4);
                fVar4.f1908f = -d5;
                i4.k.add(fVar4);
            }
        }
        this.f1935h.f1903a = this;
        this.i.f1903a = this;
    }

    @Override // b.h.b.i.l.o
    public void e() {
        for (int i = 0; i < this.k.size(); i++) {
            this.k.get(i).e();
        }
    }

    @Override // b.h.b.i.l.o
    public void f() {
        this.f1930c = null;
        Iterator<o> it = this.k.iterator();
        while (it.hasNext()) {
            it.next().f();
        }
    }

    @Override // b.h.b.i.l.o
    public long j() {
        int size = this.k.size();
        long j = 0;
        for (int i = 0; i < size; i++) {
            o oVar = this.k.get(i);
            j = oVar.i.f1908f + oVar.j() + j + oVar.f1935h.f1908f;
        }
        return j;
    }

    @Override // b.h.b.i.l.o
    public boolean k() {
        int size = this.k.size();
        for (int i = 0; i < size; i++) {
            if (!this.k.get(i).k()) {
                return false;
            }
        }
        return true;
    }

    public final b.h.b.i.d m() {
        for (int i = 0; i < this.k.size(); i++) {
            b.h.b.i.d dVar = this.k.get(i).f1929b;
            if (dVar.c0 != 8) {
                return dVar;
            }
        }
        return null;
    }

    public final b.h.b.i.d n() {
        for (int size = this.k.size() - 1; size >= 0; size--) {
            b.h.b.i.d dVar = this.k.get(size).f1929b;
            if (dVar.c0 != 8) {
                return dVar;
            }
        }
        return null;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ChainRun ");
        x.append(this.f1933f == 0 ? "horizontal : " : "vertical : ");
        String sb = x.toString();
        Iterator<o> it = this.k.iterator();
        while (it.hasNext()) {
            String q = c.b.a.a.a.q(sb, "<");
            sb = c.b.a.a.a.q(q + it.next(), "> ");
        }
        return sb;
    }
}