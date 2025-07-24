package b.e;

import b.h.b.d;
import b.h.b.h;
import b.h.b.i.c;
import b.h.b.i.e;
import b.h.b.i.f;
import b.h.b.i.l.n;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;

/* loaded from: classes.dex */
public final class a {
    /* JADX WARN: Code restructure failed: missing block: B:116:0x01d5, code lost:
        if (r3 == r6) goto L423;
     */
    /* JADX WARN: Code restructure failed: missing block: B:125:0x01eb, code lost:
        if (r3 == r6) goto L423;
     */
    /* JADX WARN: Code restructure failed: missing block: B:126:0x01ed, code lost:
        r3 = true;
     */
    /* JADX WARN: Code restructure failed: missing block: B:127:0x01ef, code lost:
        r3 = false;
     */
    /* JADX WARN: Code restructure failed: missing block: B:174:0x02b6, code lost:
        if (r4[r16].f1865f.f1863d == r6) goto L154;
     */
    /* JADX WARN: Code restructure failed: missing block: B:71:0x0139, code lost:
        if (r5[r2].f1865f.f1863d == r6) goto L70;
     */
    /* JADX WARN: Removed duplicated region for block: B:203:0x0339  */
    /* JADX WARN: Removed duplicated region for block: B:206:0x0356  */
    /* JADX WARN: Removed duplicated region for block: B:216:0x0371  */
    /* JADX WARN: Removed duplicated region for block: B:249:0x0488 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:269:0x04ed A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:335:0x05ef A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:344:0x0602  */
    /* JADX WARN: Removed duplicated region for block: B:387:0x06b4  */
    /* JADX WARN: Removed duplicated region for block: B:392:0x06ed  */
    /* JADX WARN: Removed duplicated region for block: B:397:0x0702 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:402:0x0716  */
    /* JADX WARN: Removed duplicated region for block: B:403:0x0719  */
    /* JADX WARN: Removed duplicated region for block: B:406:0x071f  */
    /* JADX WARN: Removed duplicated region for block: B:407:0x0722  */
    /* JADX WARN: Removed duplicated region for block: B:409:0x0726  */
    /* JADX WARN: Removed duplicated region for block: B:414:0x0736  */
    /* JADX WARN: Removed duplicated region for block: B:416:0x073c A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:426:0x075b A[ADDED_TO_REGION, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void a(e eVar, d dVar, ArrayList<b.h.b.i.d> arrayList, int i) {
        int i2;
        b.h.b.i.b[] bVarArr;
        int i3;
        boolean z;
        boolean z2;
        int i4;
        boolean z3;
        boolean z4;
        int i5;
        b.h.b.i.b[] bVarArr2;
        ArrayList<b.h.b.i.d> arrayList2;
        b.h.b.i.b bVar;
        b.h.b.i.d dVar2;
        b.h.b.i.d dVar3;
        int i6;
        b.h.b.i.d dVar4;
        c cVar;
        c cVar2;
        c cVar3;
        d dVar5;
        int i7;
        b.h.b.i.d dVar6;
        c cVar4;
        h hVar;
        h hVar2;
        b.h.b.i.d dVar7;
        int i8;
        c cVar5;
        h hVar3;
        h hVar4;
        b.h.b.i.d dVar8;
        b.h.b.i.d dVar9;
        int i9;
        c cVar6;
        int i10;
        c cVar7;
        h hVar5;
        h hVar6;
        float f2;
        int size;
        int i11;
        ArrayList<b.h.b.i.d> arrayList3;
        float f3;
        b.h.b.i.b bVar2;
        int i12;
        b.h.b.i.d dVar10;
        float f4;
        boolean z5;
        int i13;
        b.h.b.i.d dVar11;
        b.h.b.i.b[] bVarArr3;
        b.h.b.i.d dVar12;
        int i14;
        int i15;
        b.h.b.i.d dVar13;
        int l;
        e eVar2 = eVar;
        d dVar14 = dVar;
        ArrayList<b.h.b.i.d> arrayList4 = arrayList;
        int i16 = 2;
        if (i == 0) {
            i2 = eVar2.t0;
            bVarArr = eVar2.w0;
            i3 = 0;
        } else {
            i2 = eVar2.u0;
            bVarArr = eVar2.v0;
            i3 = 2;
        }
        int i17 = 0;
        while (i17 < i2) {
            b.h.b.i.b bVar3 = bVarArr[i17];
            int i18 = 8;
            int i19 = 1;
            if (bVar3.t) {
                z = true;
            } else {
                int i20 = bVar3.o * i16;
                b.h.b.i.d dVar15 = bVar3.f1852a;
                b.h.b.i.d dVar16 = dVar15;
                boolean z6 = false;
                while (!z6) {
                    bVar3.i += i19;
                    b.h.b.i.d[] dVarArr = dVar15.i0;
                    int i21 = bVar3.o;
                    dVarArr[i21] = null;
                    dVar15.h0[i21] = null;
                    if (dVar15.c0 != i18) {
                        bVar3.l += i19;
                        if (dVar15.k(i21) != 3) {
                            int i22 = bVar3.m;
                            int i23 = bVar3.o;
                            if (i23 == 0) {
                                l = dVar15.r();
                            } else {
                                l = i23 == i19 ? dVar15.l() : 0;
                            }
                            bVar3.m = i22 + l;
                        }
                        int d2 = dVar15.L[i20].d() + bVar3.m;
                        bVar3.m = d2;
                        int i24 = i20 + 1;
                        bVar3.m = dVar15.L[i24].d() + d2;
                        int d3 = dVar15.L[i20].d() + bVar3.n;
                        bVar3.n = d3;
                        bVar3.n = dVar15.L[i24].d() + d3;
                        if (bVar3.f1853b == null) {
                            bVar3.f1853b = dVar15;
                        }
                        bVar3.f1855d = dVar15;
                        int[] iArr = dVar15.O;
                        int i25 = bVar3.o;
                        if (iArr[i25] == 3) {
                            int[] iArr2 = dVar15.n;
                            if (iArr2[i25] == 0 || iArr2[i25] == 3 || iArr2[i25] == i16) {
                                bVar3.j++;
                                float[] fArr = dVar15.g0;
                                float f5 = fArr[i25];
                                if (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                    bVar3.k += fArr[i25];
                                }
                                if (dVar15.c0 != 8 && iArr[i25] == 3 && (iArr2[i25] == 0 || iArr2[i25] == 3)) {
                                    if (f5 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                        bVar3.q = true;
                                    } else {
                                        bVar3.r = true;
                                    }
                                    if (bVar3.f1859h == null) {
                                        bVar3.f1859h = new ArrayList<>();
                                    }
                                    bVar3.f1859h.add(dVar15);
                                }
                                if (bVar3.f1857f == null) {
                                    bVar3.f1857f = dVar15;
                                }
                                b.h.b.i.d dVar17 = bVar3.f1858g;
                                if (dVar17 != null) {
                                    dVar17.h0[bVar3.o] = dVar15;
                                }
                                bVar3.f1858g = dVar15;
                            }
                            int i26 = bVar3.o;
                        }
                    }
                    if (dVar16 != dVar15) {
                        dVar16.i0[bVar3.o] = dVar15;
                    }
                    c cVar8 = dVar15.L[i20 + 1].f1865f;
                    if (cVar8 != null) {
                        dVar13 = cVar8.f1863d;
                        c[] cVarArr = dVar13.L;
                        if (cVarArr[i20].f1865f != null) {
                        }
                    }
                    dVar13 = null;
                    if (dVar13 == null) {
                        dVar13 = dVar15;
                        z6 = true;
                    }
                    dVar16 = dVar15;
                    i19 = 1;
                    i18 = 8;
                    i16 = 2;
                    dVar15 = dVar13;
                }
                b.h.b.i.d dVar18 = bVar3.f1853b;
                if (dVar18 != null) {
                    bVar3.m -= dVar18.L[i20].d();
                }
                b.h.b.i.d dVar19 = bVar3.f1855d;
                if (dVar19 != null) {
                    bVar3.m -= dVar19.L[i20 + 1].d();
                }
                bVar3.f1854c = dVar15;
                if (bVar3.o == 0 && bVar3.p) {
                    bVar3.f1856e = dVar15;
                } else {
                    bVar3.f1856e = bVar3.f1852a;
                }
                bVar3.s = bVar3.r && bVar3.q;
                z = true;
            }
            bVar3.t = z;
            if (arrayList4 == null || arrayList4.contains(bVar3.f1852a)) {
                b.h.b.i.d dVar20 = bVar3.f1852a;
                b.h.b.i.d dVar21 = bVar3.f1854c;
                b.h.b.i.d dVar22 = bVar3.f1853b;
                b.h.b.i.d dVar23 = bVar3.f1855d;
                b.h.b.i.d dVar24 = bVar3.f1856e;
                float f6 = bVar3.k;
                boolean z7 = eVar2.O[i] == 2;
                if (i == 0) {
                    int i27 = dVar24.e0;
                    boolean z8 = i27 == 0;
                    if (i27 == 1) {
                        z3 = true;
                        i15 = 2;
                    } else {
                        i15 = 2;
                        z3 = false;
                    }
                    z2 = z8;
                } else {
                    int i28 = dVar24.f0;
                    z2 = i28 == 0;
                    if (i28 == 1) {
                        i4 = 2;
                        z3 = true;
                    } else {
                        i4 = 2;
                        z3 = false;
                    }
                }
                b.h.b.i.d dVar25 = dVar20;
                boolean z9 = false;
                while (!z9) {
                    c cVar9 = dVar25.L[i3];
                    int i29 = z4 ? 1 : 4;
                    int d4 = cVar9.d();
                    int i30 = i17;
                    boolean z10 = dVar25.O[i] == 3 && dVar25.n[i] == 0;
                    c cVar10 = cVar9.f1865f;
                    if (cVar10 != null && dVar25 != dVar20) {
                        d4 = cVar10.d() + d4;
                    }
                    int i31 = d4;
                    if (!z4 || dVar25 == dVar20 || dVar25 == dVar22) {
                        i13 = i2;
                    } else {
                        i13 = i2;
                        i29 = 8;
                    }
                    c cVar11 = cVar9.f1865f;
                    if (cVar11 != null) {
                        if (dVar25 == dVar22) {
                            bVarArr3 = bVarArr;
                            dVar11 = dVar24;
                            dVar14.f(cVar9.i, cVar11.i, i31, 6);
                        } else {
                            dVar11 = dVar24;
                            bVarArr3 = bVarArr;
                            dVar14.f(cVar9.i, cVar11.i, i31, 8);
                        }
                        dVar14.d(cVar9.i, cVar9.f1865f.i, i31, (!z10 || z4) ? i29 : 5);
                    } else {
                        dVar11 = dVar24;
                        bVarArr3 = bVarArr;
                    }
                    if (z7) {
                        if (dVar25.c0 == 8 || dVar25.O[i] != 3) {
                            i14 = 0;
                        } else {
                            c[] cVarArr2 = dVar25.L;
                            i14 = 0;
                            dVar14.f(cVarArr2[i3 + 1].i, cVarArr2[i3].i, 0, 5);
                        }
                        dVar14.f(dVar25.L[i3].i, eVar2.L[i3].i, i14, 8);
                    }
                    c cVar12 = dVar25.L[i3 + 1].f1865f;
                    if (cVar12 != null) {
                        dVar12 = cVar12.f1863d;
                        c[] cVarArr3 = dVar12.L;
                        if (cVarArr3[i3].f1865f != null) {
                        }
                    }
                    dVar12 = null;
                    if (dVar12 != null) {
                        dVar25 = dVar12;
                    } else {
                        z9 = true;
                    }
                    i2 = i13;
                    i17 = i30;
                    bVarArr = bVarArr3;
                    dVar24 = dVar11;
                }
                b.h.b.i.d dVar26 = dVar24;
                int i32 = i17;
                i5 = i2;
                bVarArr2 = bVarArr;
                if (dVar23 != null) {
                    int i33 = i3 + 1;
                    if (dVar21.L[i33].f1865f != null) {
                        c cVar13 = dVar23.L[i33];
                        if ((dVar23.O[i] == 3 && dVar23.n[i] == 0) && !z4) {
                            c cVar14 = cVar13.f1865f;
                            if (cVar14.f1863d == eVar2) {
                                dVar14.d(cVar13.i, cVar14.i, -cVar13.d(), 5);
                                dVar14.g(cVar13.i, dVar21.L[i33].f1865f.i, -cVar13.d(), 6);
                                if (z7) {
                                    int i34 = i3 + 1;
                                    h hVar7 = eVar2.L[i34].i;
                                    c[] cVarArr4 = dVar21.L;
                                    dVar14.f(hVar7, cVarArr4[i34].i, cVarArr4[i34].d(), 8);
                                }
                                arrayList2 = bVar3.f1859h;
                                if (arrayList2 != null && (size = arrayList2.size()) > 1) {
                                    float f7 = (bVar3.q || bVar3.s) ? f6 : bVar3.j;
                                    b.h.b.i.d dVar27 = null;
                                    float f8 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                    i11 = 0;
                                    while (i11 < size) {
                                        b.h.b.i.d dVar28 = arrayList2.get(i11);
                                        float f9 = dVar28.g0[i];
                                        if (f9 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                            arrayList3 = arrayList2;
                                            f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                        } else if (bVar3.s) {
                                            c[] cVarArr5 = dVar28.L;
                                            arrayList3 = arrayList2;
                                            dVar14.d(cVarArr5[i3 + 1].i, cVarArr5[i3].i, 0, 4);
                                            z5 = false;
                                            bVar2 = bVar3;
                                            i12 = size;
                                            f4 = f7;
                                            i11++;
                                            arrayList2 = arrayList3;
                                            f7 = f4;
                                            size = i12;
                                            bVar3 = bVar2;
                                        } else {
                                            arrayList3 = arrayList2;
                                            f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                            f9 = 1.0f;
                                        }
                                        int i35 = (f9 > f3 ? 1 : (f9 == f3 ? 0 : -1));
                                        if (i35 == 0) {
                                            c[] cVarArr6 = dVar28.L;
                                            z5 = false;
                                            dVar14.d(cVarArr6[i3 + 1].i, cVarArr6[i3].i, 0, 8);
                                            bVar2 = bVar3;
                                            i12 = size;
                                            f4 = f7;
                                            i11++;
                                            arrayList2 = arrayList3;
                                            f7 = f4;
                                            size = i12;
                                            bVar3 = bVar2;
                                        } else {
                                            if (dVar27 != null) {
                                                c[] cVarArr7 = dVar27.L;
                                                h hVar8 = cVarArr7[i3].i;
                                                int i36 = i3 + 1;
                                                h hVar9 = cVarArr7[i36].i;
                                                c[] cVarArr8 = dVar28.L;
                                                i12 = size;
                                                h hVar10 = cVarArr8[i3].i;
                                                h hVar11 = cVarArr8[i36].i;
                                                dVar10 = dVar28;
                                                b.h.b.b m = dVar.m();
                                                bVar2 = bVar3;
                                                m.f1821b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                                if (f7 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || f8 == f9) {
                                                    f4 = f7;
                                                    m.f1823d.i(hVar8, 1.0f);
                                                    m.f1823d.i(hVar9, -1.0f);
                                                    m.f1823d.i(hVar11, 1.0f);
                                                    m.f1823d.i(hVar10, -1.0f);
                                                } else {
                                                    if (f8 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                                        m.f1823d.i(hVar8, 1.0f);
                                                        m.f1823d.i(hVar9, -1.0f);
                                                    } else if (i35 == 0) {
                                                        m.f1823d.i(hVar10, 1.0f);
                                                        m.f1823d.i(hVar11, -1.0f);
                                                    } else {
                                                        float f10 = (f8 / f7) / (f9 / f7);
                                                        f4 = f7;
                                                        m.f1823d.i(hVar8, 1.0f);
                                                        m.f1823d.i(hVar9, -1.0f);
                                                        m.f1823d.i(hVar11, f10);
                                                        m.f1823d.i(hVar10, -f10);
                                                    }
                                                    f4 = f7;
                                                }
                                                dVar14.c(m);
                                            } else {
                                                bVar2 = bVar3;
                                                i12 = size;
                                                dVar10 = dVar28;
                                                f4 = f7;
                                            }
                                            f8 = f9;
                                            dVar27 = dVar10;
                                            i11++;
                                            arrayList2 = arrayList3;
                                            f7 = f4;
                                            size = i12;
                                            bVar3 = bVar2;
                                        }
                                    }
                                }
                                bVar = bVar3;
                                if (dVar22 == null && (dVar22 == dVar23 || z4)) {
                                    c cVar15 = dVar20.L[i3];
                                    int i37 = i3 + 1;
                                    c cVar16 = dVar21.L[i37];
                                    c cVar17 = cVar15.f1865f;
                                    h hVar12 = cVar17 != null ? cVar17.i : null;
                                    c cVar18 = cVar16.f1865f;
                                    h hVar13 = cVar18 != null ? cVar18.i : null;
                                    c cVar19 = dVar22.L[i3];
                                    c cVar20 = dVar23.L[i37];
                                    if (hVar12 != null && hVar13 != null) {
                                        if (i == 0) {
                                            f2 = dVar26.Z;
                                        } else {
                                            f2 = dVar26.a0;
                                        }
                                        int d5 = cVar19.d();
                                        dVar2 = dVar23;
                                        dVar3 = dVar22;
                                        i6 = i32;
                                        dVar.b(cVar19.i, hVar12, d5, f2, hVar13, cVar20.i, cVar20.d(), 7);
                                        dVar5 = dVar;
                                        if (!z2) {
                                        }
                                        c[] cVarArr9 = dVar3.L;
                                        cVar6 = cVarArr9[i3];
                                        i10 = i3 + 1;
                                        cVar7 = dVar2.L[i10];
                                        c cVar21 = cVar6.f1865f;
                                        if (cVar21 != null) {
                                        }
                                        c cVar22 = cVar7.f1865f;
                                        if (cVar22 != null) {
                                        }
                                        if (dVar21 != dVar2) {
                                        }
                                        if (dVar3 == dVar2) {
                                        }
                                        if (hVar5 == null) {
                                        }
                                    } else {
                                        dVar2 = dVar23;
                                        dVar3 = dVar22;
                                        i6 = i32;
                                        dVar5 = dVar14;
                                        if (!z2) {
                                        }
                                        c[] cVarArr92 = dVar3.L;
                                        cVar6 = cVarArr92[i3];
                                        i10 = i3 + 1;
                                        cVar7 = dVar2.L[i10];
                                        c cVar212 = cVar6.f1865f;
                                        if (cVar212 != null) {
                                        }
                                        c cVar222 = cVar7.f1865f;
                                        if (cVar222 != null) {
                                        }
                                        if (dVar21 != dVar2) {
                                        }
                                        if (dVar3 == dVar2) {
                                        }
                                        if (hVar5 == null) {
                                        }
                                    }
                                } else {
                                    dVar2 = dVar23;
                                    dVar3 = dVar22;
                                    i6 = i32;
                                    if (!z2 && dVar3 != null) {
                                        int i38 = bVar.j;
                                        boolean z11 = i38 > 0 && bVar.i == i38;
                                        b.h.b.i.d dVar29 = dVar3;
                                        b.h.b.i.d dVar30 = dVar29;
                                        while (dVar30 != null) {
                                            b.h.b.i.d dVar31 = dVar30.i0[i];
                                            while (true) {
                                                if (dVar31 == null) {
                                                    i8 = 8;
                                                    break;
                                                }
                                                i8 = 8;
                                                if (dVar31.c0 != 8) {
                                                    break;
                                                }
                                                dVar31 = dVar31.i0[i];
                                            }
                                            if (dVar31 != null || dVar30 == dVar2) {
                                                c cVar23 = dVar30.L[i3];
                                                h hVar14 = cVar23.i;
                                                c cVar24 = cVar23.f1865f;
                                                h hVar15 = cVar24 != null ? cVar24.i : null;
                                                if (dVar29 != dVar30) {
                                                    hVar15 = dVar29.L[i3 + 1].i;
                                                } else if (dVar30 == dVar3 && dVar29 == dVar30) {
                                                    c[] cVarArr10 = dVar20.L;
                                                    hVar15 = cVarArr10[i3].f1865f != null ? cVarArr10[i3].f1865f.i : null;
                                                }
                                                int d6 = cVar23.d();
                                                int i39 = i3 + 1;
                                                int d7 = dVar30.L[i39].d();
                                                if (dVar31 != null) {
                                                    cVar5 = dVar31.L[i3];
                                                    hVar3 = cVar5.i;
                                                    hVar4 = dVar30.L[i39].i;
                                                } else {
                                                    cVar5 = dVar21.L[i39].f1865f;
                                                    hVar3 = cVar5 != null ? cVar5.i : null;
                                                    hVar4 = dVar30.L[i39].i;
                                                }
                                                if (cVar5 != null) {
                                                    d7 += cVar5.d();
                                                }
                                                if (dVar29 != null) {
                                                    d6 += dVar29.L[i39].d();
                                                }
                                                if (hVar14 == null || hVar15 == null || hVar3 == null || hVar4 == null) {
                                                    dVar8 = dVar31;
                                                    dVar9 = dVar29;
                                                    i9 = 8;
                                                } else {
                                                    if (dVar30 == dVar3) {
                                                        d6 = dVar3.L[i3].d();
                                                    }
                                                    int i40 = d6;
                                                    h hVar16 = hVar3;
                                                    h hVar17 = hVar4;
                                                    dVar8 = dVar31;
                                                    i9 = 8;
                                                    dVar9 = dVar29;
                                                    dVar.b(hVar14, hVar15, i40, 0.5f, hVar16, hVar17, dVar30 == dVar2 ? dVar2.L[i39].d() : d7, z11 ? 8 : 5);
                                                }
                                            } else {
                                                i9 = i8;
                                                dVar8 = dVar31;
                                                dVar9 = dVar29;
                                            }
                                            dVar29 = dVar30.c0 != i9 ? dVar30 : dVar9;
                                            dVar30 = dVar8;
                                        }
                                    } else if (z3 && dVar3 != null) {
                                        int i41 = bVar.j;
                                        boolean z12 = i41 <= 0 && bVar.i == i41;
                                        dVar4 = dVar3;
                                        b.h.b.i.d dVar32 = dVar4;
                                        while (dVar4 != null) {
                                            b.h.b.i.d dVar33 = dVar4.i0[i];
                                            while (dVar33 != null && dVar33.c0 == 8) {
                                                dVar33 = dVar33.i0[i];
                                            }
                                            if (dVar4 == dVar3 || dVar4 == dVar2 || dVar33 == null) {
                                                dVar6 = dVar33;
                                            } else {
                                                b.h.b.i.d dVar34 = dVar33 == dVar2 ? null : dVar33;
                                                c cVar25 = dVar4.L[i3];
                                                h hVar18 = cVar25.i;
                                                int i42 = i3 + 1;
                                                h hVar19 = dVar32.L[i42].i;
                                                int d8 = cVar25.d();
                                                int d9 = dVar4.L[i42].d();
                                                if (dVar34 != null) {
                                                    cVar4 = dVar34.L[i3];
                                                    hVar = cVar4.i;
                                                    c cVar26 = cVar4.f1865f;
                                                    hVar2 = cVar26 != null ? cVar26.i : null;
                                                } else {
                                                    cVar4 = dVar2.L[i3];
                                                    hVar = cVar4 != null ? cVar4.i : null;
                                                    hVar2 = dVar4.L[i42].i;
                                                }
                                                int d10 = cVar4 != null ? cVar4.d() + d9 : d9;
                                                int d11 = dVar32.L[i42].d() + d8;
                                                int i43 = z12 ? 8 : 4;
                                                if (hVar18 == null || hVar19 == null || hVar == null || hVar2 == null) {
                                                    dVar7 = dVar34;
                                                } else {
                                                    h hVar20 = hVar;
                                                    h hVar21 = hVar2;
                                                    int i44 = d10;
                                                    dVar7 = dVar34;
                                                    dVar.b(hVar18, hVar19, d11, 0.5f, hVar20, hVar21, i44, i43);
                                                }
                                                dVar6 = dVar7;
                                            }
                                            if (dVar4.c0 != 8) {
                                                dVar32 = dVar4;
                                            }
                                            dVar4 = dVar6;
                                        }
                                        c cVar27 = dVar3.L[i3];
                                        cVar = dVar20.L[i3].f1865f;
                                        int i45 = i3 + 1;
                                        cVar2 = dVar2.L[i45];
                                        cVar3 = dVar21.L[i45].f1865f;
                                        if (cVar != null) {
                                            dVar5 = dVar;
                                            i7 = 5;
                                        } else if (dVar3 != dVar2) {
                                            dVar5 = dVar;
                                            i7 = 5;
                                            dVar5.d(cVar27.i, cVar.i, cVar27.d(), 5);
                                        } else {
                                            dVar5 = dVar;
                                            i7 = 5;
                                            if (cVar3 != null) {
                                                dVar.b(cVar27.i, cVar.i, cVar27.d(), 0.5f, cVar2.i, cVar3.i, cVar2.d(), 5);
                                            }
                                        }
                                        if (cVar3 != null && dVar3 != dVar2) {
                                            dVar5.d(cVar2.i, cVar3.i, -cVar2.d(), i7);
                                        }
                                        if ((!z2 || z3) && dVar3 != null && dVar3 != dVar2) {
                                            c[] cVarArr922 = dVar3.L;
                                            cVar6 = cVarArr922[i3];
                                            i10 = i3 + 1;
                                            cVar7 = dVar2.L[i10];
                                            c cVar2122 = cVar6.f1865f;
                                            hVar5 = cVar2122 != null ? cVar2122.i : null;
                                            c cVar2222 = cVar7.f1865f;
                                            hVar6 = cVar2222 != null ? cVar2222.i : null;
                                            if (dVar21 != dVar2) {
                                                c cVar28 = dVar21.L[i10].f1865f;
                                                hVar6 = cVar28 != null ? cVar28.i : null;
                                            }
                                            if (dVar3 == dVar2) {
                                                cVar6 = cVarArr922[i3];
                                                cVar7 = cVarArr922[i10];
                                            }
                                            if (hVar5 == null && hVar6 != null) {
                                                dVar.b(cVar6.i, hVar5, cVar6.d(), 0.5f, hVar6, cVar7.i, dVar2.L[i10].d(), 5);
                                            }
                                        }
                                    }
                                    dVar5 = dVar;
                                    if (!z2) {
                                    }
                                    c[] cVarArr9222 = dVar3.L;
                                    cVar6 = cVarArr9222[i3];
                                    i10 = i3 + 1;
                                    cVar7 = dVar2.L[i10];
                                    c cVar21222 = cVar6.f1865f;
                                    if (cVar21222 != null) {
                                    }
                                    c cVar22222 = cVar7.f1865f;
                                    if (cVar22222 != null) {
                                    }
                                    if (dVar21 != dVar2) {
                                    }
                                    if (dVar3 == dVar2) {
                                    }
                                    if (hVar5 == null) {
                                        dVar.b(cVar6.i, hVar5, cVar6.d(), 0.5f, hVar6, cVar7.i, dVar2.L[i10].d(), 5);
                                    }
                                }
                            }
                        }
                        if (z4) {
                            c cVar29 = cVar13.f1865f;
                            if (cVar29.f1863d == eVar2) {
                                dVar14.d(cVar13.i, cVar29.i, -cVar13.d(), 4);
                            }
                        }
                        dVar14.g(cVar13.i, dVar21.L[i33].f1865f.i, -cVar13.d(), 6);
                        if (z7) {
                        }
                        arrayList2 = bVar3.f1859h;
                        if (arrayList2 != null) {
                            if (bVar3.q) {
                            }
                            b.h.b.i.d dVar272 = null;
                            float f82 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                            i11 = 0;
                            while (i11 < size) {
                            }
                        }
                        bVar = bVar3;
                        if (dVar22 == null) {
                        }
                        dVar2 = dVar23;
                        dVar3 = dVar22;
                        i6 = i32;
                        if (!z2) {
                        }
                        if (z3) {
                            int i412 = bVar.j;
                            if (i412 <= 0) {
                            }
                            dVar4 = dVar3;
                            b.h.b.i.d dVar322 = dVar4;
                            while (dVar4 != null) {
                            }
                            c cVar272 = dVar3.L[i3];
                            cVar = dVar20.L[i3].f1865f;
                            int i452 = i3 + 1;
                            cVar2 = dVar2.L[i452];
                            cVar3 = dVar21.L[i452].f1865f;
                            if (cVar != null) {
                            }
                            if (cVar3 != null) {
                                dVar5.d(cVar2.i, cVar3.i, -cVar2.d(), i7);
                            }
                            if (!z2) {
                            }
                            c[] cVarArr92222 = dVar3.L;
                            cVar6 = cVarArr92222[i3];
                            i10 = i3 + 1;
                            cVar7 = dVar2.L[i10];
                            c cVar212222 = cVar6.f1865f;
                            if (cVar212222 != null) {
                            }
                            c cVar222222 = cVar7.f1865f;
                            if (cVar222222 != null) {
                            }
                            if (dVar21 != dVar2) {
                            }
                            if (dVar3 == dVar2) {
                            }
                            if (hVar5 == null) {
                            }
                        }
                        dVar5 = dVar;
                        if (!z2) {
                        }
                        c[] cVarArr922222 = dVar3.L;
                        cVar6 = cVarArr922222[i3];
                        i10 = i3 + 1;
                        cVar7 = dVar2.L[i10];
                        c cVar2122222 = cVar6.f1865f;
                        if (cVar2122222 != null) {
                        }
                        c cVar2222222 = cVar7.f1865f;
                        if (cVar2222222 != null) {
                        }
                        if (dVar21 != dVar2) {
                        }
                        if (dVar3 == dVar2) {
                        }
                        if (hVar5 == null) {
                        }
                    }
                }
                if (z7) {
                }
                arrayList2 = bVar3.f1859h;
                if (arrayList2 != null) {
                }
                bVar = bVar3;
                if (dVar22 == null) {
                }
                dVar2 = dVar23;
                dVar3 = dVar22;
                i6 = i32;
                if (!z2) {
                }
                if (z3) {
                }
                dVar5 = dVar;
                if (!z2) {
                }
                c[] cVarArr9222222 = dVar3.L;
                cVar6 = cVarArr9222222[i3];
                i10 = i3 + 1;
                cVar7 = dVar2.L[i10];
                c cVar21222222 = cVar6.f1865f;
                if (cVar21222222 != null) {
                }
                c cVar22222222 = cVar7.f1865f;
                if (cVar22222222 != null) {
                }
                if (dVar21 != dVar2) {
                }
                if (dVar3 == dVar2) {
                }
                if (hVar5 == null) {
                }
            } else {
                i6 = i17;
                dVar5 = dVar14;
                i5 = i2;
                bVarArr2 = bVarArr;
            }
            i17 = i6 + 1;
            eVar2 = eVar;
            arrayList4 = arrayList;
            dVar14 = dVar5;
            i2 = i5;
            bVarArr = bVarArr2;
            i16 = 2;
        }
    }

    public static n b(b.h.b.i.d dVar, int i, ArrayList<n> arrayList, n nVar) {
        int i2;
        int i3;
        if (i == 0) {
            i2 = dVar.j0;
        } else {
            i2 = dVar.k0;
        }
        if (i2 != -1 && (nVar == null || i2 != nVar.f1924c)) {
            int i4 = 0;
            while (true) {
                if (i4 >= arrayList.size()) {
                    break;
                }
                n nVar2 = arrayList.get(i4);
                if (nVar2.f1924c == i2) {
                    if (nVar != null) {
                        nVar.d(i, nVar2);
                        arrayList.remove(nVar);
                    }
                    nVar = nVar2;
                } else {
                    i4++;
                }
            }
        } else if (i2 != -1) {
            return nVar;
        }
        if (nVar == null) {
            if (dVar instanceof b.h.b.i.h) {
                b.h.b.i.h hVar = (b.h.b.i.h) dVar;
                int i5 = 0;
                while (true) {
                    if (i5 >= hVar.m0) {
                        i3 = -1;
                        break;
                    }
                    b.h.b.i.d dVar2 = hVar.l0[i5];
                    if ((i == 0 && (i3 = dVar2.j0) != -1) || (i == 1 && (i3 = dVar2.k0) != -1)) {
                        break;
                    }
                    i5++;
                }
                if (i3 != -1) {
                    int i6 = 0;
                    while (true) {
                        if (i6 >= arrayList.size()) {
                            break;
                        }
                        n nVar3 = arrayList.get(i6);
                        if (nVar3.f1924c == i3) {
                            nVar = nVar3;
                            break;
                        }
                        i6++;
                    }
                }
            }
            if (nVar == null) {
                nVar = new n(i);
            }
            arrayList.add(nVar);
        }
        if (nVar.a(dVar)) {
            if (dVar instanceof f) {
                f fVar = (f) dVar;
                fVar.o0.b(fVar.p0 == 0 ? 1 : 0, arrayList, nVar);
            }
            if (i == 0) {
                dVar.j0 = nVar.f1924c;
                dVar.D.b(i, arrayList, nVar);
                dVar.F.b(i, arrayList, nVar);
            } else {
                dVar.k0 = nVar.f1924c;
                dVar.E.b(i, arrayList, nVar);
                dVar.H.b(i, arrayList, nVar);
                dVar.G.b(i, arrayList, nVar);
            }
            dVar.K.b(i, arrayList, nVar);
        }
        return nVar;
    }

    public static n c(ArrayList<n> arrayList, int i) {
        int size = arrayList.size();
        for (int i2 = 0; i2 < size; i2++) {
            n nVar = arrayList.get(i2);
            if (i == nVar.f1924c) {
                return nVar;
            }
        }
        return null;
    }

    public static <T> ListenableFuture<T> d(b.g.a.d<T> dVar) {
        b.g.a.b<T> bVar = new b.g.a.b<>();
        b.g.a.e<T> eVar = new b.g.a.e<>(bVar);
        bVar.f1806b = eVar;
        bVar.f1805a = dVar.getClass();
        try {
            Object a2 = dVar.a(bVar);
            if (a2 != null) {
                bVar.f1805a = a2;
            }
        } catch (Exception e2) {
            eVar.f1810c.i(e2);
        }
        return eVar;
    }

    public static boolean e(int i, int i2, int i3, int i4) {
        return (i3 == 1 || i3 == 2 || (i3 == 4 && i != 2)) || (i4 == 1 || i4 == 2 || (i4 == 4 && i2 != 2));
    }
}