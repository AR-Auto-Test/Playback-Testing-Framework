package b.h.b.i.l;

import b.h.b.i.c;
import b.h.b.i.l.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.HashSet;
import java.util.Iterator;

/* compiled from: Direct.java */
/* loaded from: classes.dex */
public class h {

    /* renamed from: a  reason: collision with root package name */
    public static b.a f1918a = new b.a();

    public static boolean a(b.h.b.i.d dVar) {
        int m = dVar.m();
        int q = dVar.q();
        b.h.b.i.d dVar2 = dVar.P;
        b.h.b.i.e eVar = dVar2 != null ? (b.h.b.i.e) dVar2 : null;
        if (eVar != null) {
            eVar.m();
        }
        if (eVar != null) {
            eVar.q();
        }
        boolean z = m == 1 || m == 2 || (m == 3 && dVar.l == 0 && dVar.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && dVar.u(0)) || dVar.z();
        boolean z2 = q == 1 || q == 2 || (q == 3 && dVar.m == 0 && dVar.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && dVar.u(1)) || dVar.A();
        if (dVar.S <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || !(z || z2)) {
            return z && z2;
        }
        return true;
    }

    public static void b(b.h.b.i.d dVar, b.InterfaceC0029b interfaceC0029b, boolean z) {
        HashSet<b.h.b.i.c> hashSet;
        b.h.b.i.c cVar;
        b.h.b.i.c cVar2;
        b.h.b.i.c cVar3;
        b.h.b.i.c cVar4;
        b.h.b.i.c cVar5;
        if (!(dVar instanceof b.h.b.i.e) && dVar.y() && a(dVar)) {
            b.h.b.i.e.X(dVar, interfaceC0029b, new b.a(), 0);
        }
        b.h.b.i.c i = dVar.i(c.a.LEFT);
        b.h.b.i.c i2 = dVar.i(c.a.RIGHT);
        int c2 = i.c();
        int c3 = i2.c();
        HashSet<b.h.b.i.c> hashSet2 = i.f1860a;
        if (hashSet2 != null && i.f1862c) {
            Iterator<b.h.b.i.c> it = hashSet2.iterator();
            while (it.hasNext()) {
                b.h.b.i.c next = it.next();
                b.h.b.i.d dVar2 = next.f1863d;
                boolean a2 = a(dVar2);
                if (dVar2.y() && a2) {
                    b.h.b.i.e.X(dVar2, interfaceC0029b, new b.a(), 0);
                }
                if (dVar2.m() == 3 && !a2) {
                    if (dVar2.m() == 3 && dVar2.p >= 0 && dVar2.o >= 0 && (dVar2.c0 == 8 || (dVar2.l == 0 && dVar2.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
                        if (!dVar2.w() && !dVar2.A) {
                            b.h.b.i.c cVar6 = dVar2.D;
                            if (((next == cVar6 && (cVar5 = dVar2.F.f1865f) != null && cVar5.f1862c) || (next == dVar2.F && (cVar4 = cVar6.f1865f) != null && cVar4.f1862c)) && !dVar2.w()) {
                                d(dVar, interfaceC0029b, dVar2, z);
                            }
                        }
                    }
                } else if (!dVar2.y()) {
                    b.h.b.i.c cVar7 = dVar2.D;
                    if (next == cVar7 && dVar2.F.f1865f == null) {
                        int d2 = cVar7.d() + c2;
                        dVar2.F(d2, dVar2.r() + d2);
                        b(dVar2, interfaceC0029b, z);
                    } else {
                        b.h.b.i.c cVar8 = dVar2.F;
                        if (next == cVar8 && cVar7.f1865f == null) {
                            int d3 = c2 - cVar8.d();
                            dVar2.F(d3 - dVar2.r(), d3);
                            b(dVar2, interfaceC0029b, z);
                        } else if (next == cVar7 && (cVar3 = cVar8.f1865f) != null && cVar3.f1862c && !dVar2.w()) {
                            c(interfaceC0029b, dVar2, z);
                        }
                    }
                }
            }
        }
        if ((dVar instanceof b.h.b.i.f) || (hashSet = i2.f1860a) == null || !i2.f1862c) {
            return;
        }
        Iterator<b.h.b.i.c> it2 = hashSet.iterator();
        while (it2.hasNext()) {
            b.h.b.i.c next2 = it2.next();
            b.h.b.i.d dVar3 = next2.f1863d;
            boolean a3 = a(dVar3);
            if (dVar3.y() && a3) {
                b.h.b.i.e.X(dVar3, interfaceC0029b, new b.a(), 0);
            }
            b.h.b.i.c cVar9 = dVar3.D;
            boolean z2 = (next2 == cVar9 && (cVar2 = dVar3.F.f1865f) != null && cVar2.f1862c) || (next2 == dVar3.F && (cVar = cVar9.f1865f) != null && cVar.f1862c);
            if (dVar3.m() == 3 && !a3) {
                if (dVar3.m() == 3 && dVar3.p >= 0 && dVar3.o >= 0 && (dVar3.c0 == 8 || (dVar3.l == 0 && dVar3.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
                    if (!dVar3.w() && !dVar3.A && z2 && !dVar3.w()) {
                        d(dVar, interfaceC0029b, dVar3, z);
                    }
                }
            } else if (!dVar3.y()) {
                b.h.b.i.c cVar10 = dVar3.D;
                if (next2 == cVar10 && dVar3.F.f1865f == null) {
                    int d4 = cVar10.d() + c3;
                    dVar3.F(d4, dVar3.r() + d4);
                    b(dVar3, interfaceC0029b, z);
                } else {
                    b.h.b.i.c cVar11 = dVar3.F;
                    if (next2 == cVar11 && cVar10.f1865f == null) {
                        int d5 = c3 - cVar11.d();
                        dVar3.F(d5 - dVar3.r(), d5);
                        b(dVar3, interfaceC0029b, z);
                    } else if (z2 && !dVar3.w()) {
                        c(interfaceC0029b, dVar3, z);
                    }
                }
            }
        }
    }

    public static void c(b.InterfaceC0029b interfaceC0029b, b.h.b.i.d dVar, boolean z) {
        float f2 = dVar.Z;
        int c2 = dVar.D.f1865f.c();
        int c3 = dVar.F.f1865f.c();
        int d2 = dVar.D.d() + c2;
        int d3 = c3 - dVar.F.d();
        if (c2 == c3) {
            f2 = 0.5f;
        } else {
            c2 = d2;
            c3 = d3;
        }
        int r = dVar.r();
        int i = (c3 - c2) - r;
        if (c2 > c3) {
            i = (c2 - c3) - r;
        }
        int i2 = ((int) ((f2 * i) + 0.5f)) + c2;
        int i3 = i2 + r;
        if (c2 > c3) {
            i3 = i2 - r;
        }
        dVar.F(i2, i3);
        b(dVar, interfaceC0029b, z);
    }

    public static void d(b.h.b.i.d dVar, b.InterfaceC0029b interfaceC0029b, b.h.b.i.d dVar2, boolean z) {
        int r;
        float f2 = dVar2.Z;
        int d2 = dVar2.D.d() + dVar2.D.f1865f.c();
        int c2 = dVar2.F.f1865f.c() - dVar2.F.d();
        if (c2 >= d2) {
            int r2 = dVar2.r();
            if (dVar2.c0 != 8) {
                int i = dVar2.l;
                if (i == 2) {
                    if (dVar instanceof b.h.b.i.e) {
                        r = dVar.r();
                    } else {
                        r = dVar.P.r();
                    }
                    r2 = (int) (dVar2.Z * 0.5f * r);
                } else if (i == 0) {
                    r2 = c2 - d2;
                }
                r2 = Math.max(dVar2.o, r2);
                int i2 = dVar2.p;
                if (i2 > 0) {
                    r2 = Math.min(i2, r2);
                }
            }
            int i3 = d2 + ((int) ((f2 * ((c2 - d2) - r2)) + 0.5f));
            dVar2.F(i3, r2 + i3);
            b(dVar2, interfaceC0029b, z);
        }
    }

    public static void e(b.InterfaceC0029b interfaceC0029b, b.h.b.i.d dVar) {
        float f2 = dVar.a0;
        int c2 = dVar.E.f1865f.c();
        int c3 = dVar.G.f1865f.c();
        int d2 = dVar.E.d() + c2;
        int d3 = c3 - dVar.G.d();
        if (c2 == c3) {
            f2 = 0.5f;
        } else {
            c2 = d2;
            c3 = d3;
        }
        int l = dVar.l();
        int i = (c3 - c2) - l;
        if (c2 > c3) {
            i = (c2 - c3) - l;
        }
        int i2 = (int) ((f2 * i) + 0.5f);
        int i3 = c2 + i2;
        int i4 = i3 + l;
        if (c2 > c3) {
            i3 = c2 - i2;
            i4 = i3 - l;
        }
        dVar.G(i3, i4);
        g(dVar, interfaceC0029b);
    }

    public static void f(b.h.b.i.d dVar, b.InterfaceC0029b interfaceC0029b, b.h.b.i.d dVar2) {
        int l;
        float f2 = dVar2.a0;
        int d2 = dVar2.E.d() + dVar2.E.f1865f.c();
        int c2 = dVar2.G.f1865f.c() - dVar2.G.d();
        if (c2 >= d2) {
            int l2 = dVar2.l();
            if (dVar2.c0 != 8) {
                int i = dVar2.m;
                if (i == 2) {
                    if (dVar instanceof b.h.b.i.e) {
                        l = dVar.l();
                    } else {
                        l = dVar.P.l();
                    }
                    l2 = (int) (f2 * 0.5f * l);
                } else if (i == 0) {
                    l2 = c2 - d2;
                }
                l2 = Math.max(dVar2.r, l2);
                int i2 = dVar2.s;
                if (i2 > 0) {
                    l2 = Math.min(i2, l2);
                }
            }
            int i3 = d2 + ((int) ((f2 * ((c2 - d2) - l2)) + 0.5f));
            dVar2.G(i3, l2 + i3);
            g(dVar2, interfaceC0029b);
        }
    }

    public static void g(b.h.b.i.d dVar, b.InterfaceC0029b interfaceC0029b) {
        b.h.b.i.c cVar;
        b.h.b.i.c cVar2;
        b.h.b.i.c cVar3;
        b.h.b.i.c cVar4;
        b.h.b.i.c cVar5;
        if (!(dVar instanceof b.h.b.i.e) && dVar.y() && a(dVar)) {
            b.h.b.i.e.X(dVar, interfaceC0029b, new b.a(), 0);
        }
        b.h.b.i.c i = dVar.i(c.a.TOP);
        b.h.b.i.c i2 = dVar.i(c.a.BOTTOM);
        int c2 = i.c();
        int c3 = i2.c();
        HashSet<b.h.b.i.c> hashSet = i.f1860a;
        if (hashSet != null && i.f1862c) {
            Iterator<b.h.b.i.c> it = hashSet.iterator();
            while (it.hasNext()) {
                b.h.b.i.c next = it.next();
                b.h.b.i.d dVar2 = next.f1863d;
                boolean a2 = a(dVar2);
                if (dVar2.y() && a2) {
                    b.h.b.i.e.X(dVar2, interfaceC0029b, new b.a(), 0);
                }
                if (dVar2.q() == 3 && !a2) {
                    if (dVar2.q() == 3 && dVar2.s >= 0 && dVar2.r >= 0 && (dVar2.c0 == 8 || (dVar2.m == 0 && dVar2.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
                        if (!dVar2.x() && !dVar2.A) {
                            b.h.b.i.c cVar6 = dVar2.E;
                            if (((next == cVar6 && (cVar5 = dVar2.G.f1865f) != null && cVar5.f1862c) || (next == dVar2.G && (cVar4 = cVar6.f1865f) != null && cVar4.f1862c)) && !dVar2.x()) {
                                f(dVar, interfaceC0029b, dVar2);
                            }
                        }
                    }
                } else if (!dVar2.y()) {
                    b.h.b.i.c cVar7 = dVar2.E;
                    if (next == cVar7 && dVar2.G.f1865f == null) {
                        int d2 = cVar7.d() + c2;
                        dVar2.G(d2, dVar2.l() + d2);
                        g(dVar2, interfaceC0029b);
                    } else {
                        b.h.b.i.c cVar8 = dVar2.G;
                        if (next == cVar8 && cVar8.f1865f == null) {
                            int d3 = c2 - cVar8.d();
                            dVar2.G(d3 - dVar2.l(), d3);
                            g(dVar2, interfaceC0029b);
                        } else if (next == cVar7 && (cVar3 = cVar8.f1865f) != null && cVar3.f1862c) {
                            e(interfaceC0029b, dVar2);
                        }
                    }
                }
            }
        }
        if (dVar instanceof b.h.b.i.f) {
            return;
        }
        HashSet<b.h.b.i.c> hashSet2 = i2.f1860a;
        if (hashSet2 != null && i2.f1862c) {
            Iterator<b.h.b.i.c> it2 = hashSet2.iterator();
            while (it2.hasNext()) {
                b.h.b.i.c next2 = it2.next();
                b.h.b.i.d dVar3 = next2.f1863d;
                boolean a3 = a(dVar3);
                if (dVar3.y() && a3) {
                    b.h.b.i.e.X(dVar3, interfaceC0029b, new b.a(), 0);
                }
                b.h.b.i.c cVar9 = dVar3.E;
                boolean z = (next2 == cVar9 && (cVar2 = dVar3.G.f1865f) != null && cVar2.f1862c) || (next2 == dVar3.G && (cVar = cVar9.f1865f) != null && cVar.f1862c);
                if (dVar3.q() == 3 && !a3) {
                    if (dVar3.q() == 3 && dVar3.s >= 0 && dVar3.r >= 0 && (dVar3.c0 == 8 || (dVar3.m == 0 && dVar3.S == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
                        if (!dVar3.x() && !dVar3.A && z && !dVar3.x()) {
                            f(dVar, interfaceC0029b, dVar3);
                        }
                    }
                } else if (!dVar3.y()) {
                    b.h.b.i.c cVar10 = dVar3.E;
                    if (next2 == cVar10 && dVar3.G.f1865f == null) {
                        int d4 = cVar10.d() + c3;
                        dVar3.G(d4, dVar3.l() + d4);
                        g(dVar3, interfaceC0029b);
                    } else {
                        b.h.b.i.c cVar11 = dVar3.G;
                        if (next2 == cVar11 && cVar10.f1865f == null) {
                            int d5 = c3 - cVar11.d();
                            dVar3.G(d5 - dVar3.l(), d5);
                            g(dVar3, interfaceC0029b);
                        } else if (z && !dVar3.x()) {
                            e(interfaceC0029b, dVar3);
                        }
                    }
                }
            }
        }
        b.h.b.i.c i3 = dVar.i(c.a.BASELINE);
        if (i3.f1860a == null || !i3.f1862c) {
            return;
        }
        int c4 = i3.c();
        Iterator<b.h.b.i.c> it3 = i3.f1860a.iterator();
        while (it3.hasNext()) {
            b.h.b.i.c next3 = it3.next();
            b.h.b.i.d dVar4 = next3.f1863d;
            boolean a4 = a(dVar4);
            if (dVar4.y() && a4) {
                b.h.b.i.e.X(dVar4, interfaceC0029b, new b.a(), 0);
            }
            if (dVar4.q() != 3 || a4) {
                if (!dVar4.y() && next3 == dVar4.H) {
                    if (dVar4.y) {
                        int i4 = c4 - dVar4.W;
                        int i5 = dVar4.R + i4;
                        dVar4.V = i4;
                        dVar4.E.j(i4);
                        dVar4.G.j(i5);
                        b.h.b.i.c cVar12 = dVar4.H;
                        cVar12.f1861b = c4;
                        cVar12.f1862c = true;
                        dVar4.i = true;
                    }
                    g(dVar4, interfaceC0029b);
                }
            }
        }
    }
}