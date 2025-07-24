package b.h.b.i;

import androidx.constraintlayout.widget.ConstraintLayout;
import b.h.b.i.c;
import b.h.b.i.l.b;
import b.h.b.i.l.n;
import b.h.b.i.l.o;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Objects;
import org.opencv.imgcodecs.Imgcodecs;

/* compiled from: ConstraintWidgetContainer.java */
/* loaded from: classes.dex */
public class e extends k {
    public int r0;
    public int s0;
    public b.h.b.i.l.b m0 = new b.h.b.i.l.b(this);
    public b.h.b.i.l.e n0 = new b.h.b.i.l.e(this);
    public b.InterfaceC0029b o0 = null;
    public boolean p0 = false;
    public b.h.b.d q0 = new b.h.b.d();
    public int t0 = 0;
    public int u0 = 0;
    public b[] v0 = new b[4];
    public b[] w0 = new b[4];
    public int x0 = Imgcodecs.IMWRITE_TIFF_XDPI;
    public boolean y0 = false;
    public boolean z0 = false;
    public WeakReference<c> A0 = null;
    public WeakReference<c> B0 = null;
    public WeakReference<c> C0 = null;
    public WeakReference<c> D0 = null;
    public b.a E0 = new b.a();

    public static boolean X(d dVar, b.InterfaceC0029b interfaceC0029b, b.a aVar, int i) {
        int i2;
        int i3;
        if (interfaceC0029b == null) {
            return false;
        }
        aVar.f1887a = dVar.m();
        aVar.f1888b = dVar.q();
        aVar.f1889c = dVar.r();
        aVar.f1890d = dVar.l();
        aVar.i = false;
        aVar.j = i;
        boolean z = aVar.f1887a == 3;
        boolean z2 = aVar.f1888b == 3;
        boolean z3 = z && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        boolean z4 = z2 && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        if (z && dVar.u(0) && dVar.l == 0 && !z3) {
            aVar.f1887a = 2;
            if (z2 && dVar.m == 0) {
                aVar.f1887a = 1;
            }
            z = false;
        }
        if (z2 && dVar.u(1) && dVar.m == 0 && !z4) {
            aVar.f1888b = 2;
            if (z && dVar.l == 0) {
                aVar.f1888b = 1;
            }
            z2 = false;
        }
        if (dVar.z()) {
            aVar.f1887a = 1;
            z = false;
        }
        if (dVar.A()) {
            aVar.f1888b = 1;
            z2 = false;
        }
        if (z3) {
            if (dVar.n[0] == 4) {
                aVar.f1887a = 1;
            } else if (!z2) {
                if (aVar.f1888b == 1) {
                    i3 = aVar.f1890d;
                } else {
                    aVar.f1887a = 2;
                    ((ConstraintLayout.b) interfaceC0029b).b(dVar, aVar);
                    i3 = aVar.f1892f;
                }
                aVar.f1887a = 1;
                int i4 = dVar.T;
                if (i4 != 0 && i4 != -1) {
                    aVar.f1889c = (int) (dVar.S / i3);
                } else {
                    aVar.f1889c = (int) (dVar.S * i3);
                }
            }
        }
        if (z4) {
            if (dVar.n[1] == 4) {
                aVar.f1888b = 1;
            } else if (!z) {
                if (aVar.f1887a == 1) {
                    i2 = aVar.f1889c;
                } else {
                    aVar.f1888b = 2;
                    ((ConstraintLayout.b) interfaceC0029b).b(dVar, aVar);
                    i2 = aVar.f1891e;
                }
                aVar.f1888b = 1;
                int i5 = dVar.T;
                if (i5 != 0 && i5 != -1) {
                    aVar.f1890d = (int) (i2 * dVar.S);
                } else {
                    aVar.f1890d = (int) (i2 / dVar.S);
                }
            }
        }
        ((ConstraintLayout.b) interfaceC0029b).b(dVar, aVar);
        dVar.M(aVar.f1891e);
        dVar.H(aVar.f1892f);
        dVar.y = aVar.f1894h;
        dVar.E(aVar.f1893g);
        aVar.j = 0;
        return aVar.i;
    }

    @Override // b.h.b.i.k, b.h.b.i.d
    public void B() {
        this.q0.u();
        this.r0 = 0;
        this.s0 = 0;
        super.B();
    }

    @Override // b.h.b.i.d
    public void N(boolean z, boolean z2) {
        super.N(z, z2);
        int size = this.l0.size();
        for (int i = 0; i < size; i++) {
            this.l0.get(i).N(z, z2);
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v25, resolved type: int[] */
    /* JADX DEBUG: Multi-variable search result rejected for r0v26, resolved type: int[] */
    /* JADX DEBUG: Multi-variable search result rejected for r0v28, resolved type: int[] */
    /* JADX WARN: Code restructure failed: missing block: B:337:0x05ac, code lost:
        r0 = false;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:321:0x0577  */
    /* JADX WARN: Removed duplicated region for block: B:335:0x05a9 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:351:0x05d4  */
    /* JADX WARN: Removed duplicated region for block: B:356:0x05ea  */
    /* JADX WARN: Removed duplicated region for block: B:364:0x0609  */
    /* JADX WARN: Removed duplicated region for block: B:371:0x0621 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:375:0x062f  */
    /* JADX WARN: Removed duplicated region for block: B:382:0x0644  */
    /* JADX WARN: Removed duplicated region for block: B:388:0x0661  */
    /* JADX WARN: Removed duplicated region for block: B:427:0x075e  */
    /* JADX WARN: Removed duplicated region for block: B:430:0x0788  */
    /* JADX WARN: Removed duplicated region for block: B:454:0x0819  */
    /* JADX WARN: Removed duplicated region for block: B:455:0x0826  */
    /* JADX WARN: Removed duplicated region for block: B:458:0x0839  */
    /* JADX WARN: Removed duplicated region for block: B:459:0x0843  */
    /* JADX WARN: Removed duplicated region for block: B:461:0x0848  */
    /* JADX WARN: Removed duplicated region for block: B:473:0x087e  */
    /* JADX WARN: Removed duplicated region for block: B:478:0x088a  */
    /* JADX WARN: Type inference failed for: r6v10, types: [boolean] */
    /* JADX WARN: Type inference failed for: r6v12 */
    /* JADX WARN: Type inference failed for: r6v9 */
    @Override // b.h.b.i.k
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void P() {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        boolean z;
        b.h.b.d dVar;
        int i6;
        boolean z2;
        int i7;
        int i8;
        boolean z3;
        boolean z4;
        boolean z5;
        int max;
        ?? r6;
        boolean z6;
        int max2;
        boolean z7;
        boolean z8;
        int i9;
        int i10;
        int i11;
        int i12;
        n nVar;
        n nVar2;
        boolean z9;
        int c2;
        int c3;
        int i13;
        int i14;
        int i15;
        int i16;
        this.U = 0;
        this.V = 0;
        this.y0 = false;
        this.z0 = false;
        int size = this.l0.size();
        int max3 = Math.max(0, r());
        int max4 = Math.max(0, l());
        int[] iArr = this.O;
        int i17 = iArr[1];
        int i18 = iArr[0];
        if (i.b(this.x0, 1)) {
            b.InterfaceC0029b interfaceC0029b = this.o0;
            int m = m();
            int q = q();
            C();
            ArrayList<d> arrayList = this.l0;
            int size2 = arrayList.size();
            for (int i19 = 0; i19 < size2; i19++) {
                arrayList.get(i19).C();
            }
            boolean z10 = this.p0;
            if (m == 1) {
                F(0, r());
            } else {
                c cVar = this.D;
                cVar.f1861b = 0;
                cVar.f1862c = true;
                this.U = 0;
            }
            boolean z11 = false;
            boolean z12 = false;
            for (int i20 = 0; i20 < size2; i20++) {
                d dVar2 = arrayList.get(i20);
                if (dVar2 instanceof f) {
                    f fVar = (f) dVar2;
                    if (fVar.p0 == 1) {
                        int i21 = fVar.m0;
                        if (i21 != -1) {
                            fVar.P(i21);
                        } else if (fVar.n0 != -1 && z()) {
                            fVar.P(r() - fVar.n0);
                        } else if (z()) {
                            fVar.P((int) ((fVar.l0 * r()) + 0.5f));
                        }
                        z11 = true;
                    }
                } else if ((dVar2 instanceof a) && ((a) dVar2).R() == 0) {
                    z12 = true;
                }
            }
            if (z11) {
                for (int i22 = 0; i22 < size2; i22++) {
                    d dVar3 = arrayList.get(i22);
                    if (dVar3 instanceof f) {
                        f fVar2 = (f) dVar3;
                        if (fVar2.p0 == 1) {
                            b.h.b.i.l.h.b(fVar2, interfaceC0029b, z10);
                        }
                    }
                }
            }
            b.h.b.i.l.h.b(this, interfaceC0029b, z10);
            if (z12) {
                for (int i23 = 0; i23 < size2; i23++) {
                    d dVar4 = arrayList.get(i23);
                    if (dVar4 instanceof a) {
                        a aVar = (a) dVar4;
                        if (aVar.R() == 0 && aVar.Q()) {
                            b.h.b.i.l.h.b(aVar, interfaceC0029b, z10);
                        }
                    }
                }
            }
            if (q == 1) {
                G(0, l());
            } else {
                c cVar2 = this.E;
                cVar2.f1861b = 0;
                cVar2.f1862c = true;
                this.V = 0;
            }
            boolean z13 = false;
            boolean z14 = false;
            for (int i24 = 0; i24 < size2; i24++) {
                d dVar5 = arrayList.get(i24);
                if (dVar5 instanceof f) {
                    f fVar3 = (f) dVar5;
                    if (fVar3.p0 == 0) {
                        int i25 = fVar3.m0;
                        if (i25 != -1) {
                            fVar3.P(i25);
                        } else if (fVar3.n0 != -1 && A()) {
                            fVar3.P(l() - fVar3.n0);
                        } else if (A()) {
                            fVar3.P((int) ((fVar3.l0 * l()) + 0.5f));
                        }
                        z13 = true;
                    }
                } else if ((dVar5 instanceof a) && ((a) dVar5).R() == 1) {
                    z14 = true;
                }
            }
            if (z13) {
                for (int i26 = 0; i26 < size2; i26++) {
                    d dVar6 = arrayList.get(i26);
                    if (dVar6 instanceof f) {
                        f fVar4 = (f) dVar6;
                        if (fVar4.p0 == 0) {
                            b.h.b.i.l.h.g(fVar4, interfaceC0029b);
                        }
                    }
                }
            }
            b.h.b.i.l.h.g(this, interfaceC0029b);
            if (z14) {
                for (int i27 = 0; i27 < size2; i27++) {
                    d dVar7 = arrayList.get(i27);
                    if (dVar7 instanceof a) {
                        a aVar2 = (a) dVar7;
                        if (aVar2.R() == 1 && aVar2.Q()) {
                            b.h.b.i.l.h.g(aVar2, interfaceC0029b);
                        }
                    }
                }
            }
            for (int i28 = 0; i28 < size2; i28++) {
                d dVar8 = arrayList.get(i28);
                if (dVar8.y() && b.h.b.i.l.h.a(dVar8)) {
                    X(dVar8, interfaceC0029b, b.h.b.i.l.h.f1918a, 0);
                    b.h.b.i.l.h.b(dVar8, interfaceC0029b, z10);
                    b.h.b.i.l.h.g(dVar8, interfaceC0029b);
                }
            }
            for (int i29 = 0; i29 < size; i29++) {
                d dVar9 = this.l0.get(i29);
                if (dVar9.y() && !(dVar9 instanceof f) && !(dVar9 instanceof a) && !(dVar9 instanceof j) && !dVar9.A) {
                    if (!(dVar9.k(0) == 3 && dVar9.l != 1 && dVar9.k(1) == 3 && dVar9.m != 1)) {
                        X(dVar9, this.o0, new b.a(), 0);
                    }
                }
            }
        }
        if (size <= 2 || !((i18 == 2 || i17 == 2) && i.b(this.x0, 1024))) {
            i = size;
            i2 = i18;
            i3 = max3;
            i4 = max4;
            i5 = i17;
        } else {
            b.InterfaceC0029b interfaceC0029b2 = this.o0;
            c.a aVar3 = c.a.CENTER;
            ArrayList<d> arrayList2 = this.l0;
            int size3 = arrayList2.size();
            int i30 = 0;
            while (true) {
                if (i30 < size3) {
                    d dVar10 = arrayList2.get(i30);
                    if (!b.e.a.e(m(), q(), dVar10.m(), dVar10.q())) {
                        i9 = max3;
                        i = size;
                        i10 = max4;
                        i11 = i18;
                        i12 = i17;
                        break;
                    }
                    i30++;
                } else {
                    ArrayList arrayList3 = null;
                    int i31 = 0;
                    ArrayList arrayList4 = null;
                    ArrayList arrayList5 = null;
                    ArrayList arrayList6 = null;
                    ArrayList arrayList7 = null;
                    ArrayList arrayList8 = null;
                    while (i31 < size3) {
                        int i32 = size;
                        d dVar11 = arrayList2.get(i31);
                        int i33 = max4;
                        int i34 = i17;
                        int i35 = max3;
                        int i36 = i18;
                        if (!b.e.a.e(m(), q(), dVar11.m(), dVar11.q())) {
                            X(dVar11, interfaceC0029b2, this.E0, 0);
                        }
                        boolean z15 = dVar11 instanceof f;
                        if (z15) {
                            f fVar5 = (f) dVar11;
                            if (fVar5.p0 == 0) {
                                if (arrayList5 == null) {
                                    arrayList5 = new ArrayList();
                                }
                                arrayList5.add(fVar5);
                            }
                            if (fVar5.p0 == 1) {
                                if (arrayList3 == null) {
                                    arrayList3 = new ArrayList();
                                }
                                arrayList3.add(fVar5);
                            }
                        }
                        if (dVar11 instanceof h) {
                            if (dVar11 instanceof a) {
                                a aVar4 = (a) dVar11;
                                if (aVar4.R() == 0) {
                                    if (arrayList4 == null) {
                                        arrayList4 = new ArrayList();
                                    }
                                    arrayList4.add(aVar4);
                                }
                                if (aVar4.R() == 1) {
                                    if (arrayList6 == null) {
                                        arrayList6 = new ArrayList();
                                    }
                                    arrayList6.add(aVar4);
                                }
                            } else {
                                h hVar = (h) dVar11;
                                if (arrayList4 == null) {
                                    arrayList4 = new ArrayList();
                                }
                                arrayList4.add(hVar);
                                if (arrayList6 == null) {
                                    arrayList6 = new ArrayList();
                                }
                                arrayList6.add(hVar);
                            }
                        }
                        if (dVar11.D.f1865f == null && dVar11.F.f1865f == null && !z15 && !(dVar11 instanceof a)) {
                            if (arrayList7 == null) {
                                arrayList7 = new ArrayList();
                            }
                            arrayList7.add(dVar11);
                        }
                        if (dVar11.E.f1865f == null && dVar11.G.f1865f == null && dVar11.H.f1865f == null && !z15 && !(dVar11 instanceof a)) {
                            if (arrayList8 == null) {
                                arrayList8 = new ArrayList();
                            }
                            ArrayList arrayList9 = arrayList8;
                            arrayList9.add(dVar11);
                            arrayList8 = arrayList9;
                        }
                        i31++;
                        max4 = i33;
                        size = i32;
                        i17 = i34;
                        max3 = i35;
                        i18 = i36;
                    }
                    i9 = max3;
                    i = size;
                    i10 = max4;
                    i11 = i18;
                    i12 = i17;
                    ArrayList<n> arrayList10 = new ArrayList<>();
                    if (arrayList3 != null) {
                        Iterator it = arrayList3.iterator();
                        while (it.hasNext()) {
                            b.e.a.b((f) it.next(), 0, arrayList10, null);
                        }
                    }
                    n nVar3 = null;
                    int i37 = 0;
                    if (arrayList4 != null) {
                        Iterator it2 = arrayList4.iterator();
                        while (it2.hasNext()) {
                            h hVar2 = (h) it2.next();
                            n b2 = b.e.a.b(hVar2, i37, arrayList10, nVar3);
                            hVar2.P(arrayList10, i37, b2);
                            b2.b(arrayList10);
                            nVar3 = null;
                            i37 = 0;
                        }
                    }
                    HashSet<c> hashSet = i(c.a.LEFT).f1860a;
                    if (hashSet != null) {
                        Iterator<c> it3 = hashSet.iterator();
                        while (it3.hasNext()) {
                            b.e.a.b(it3.next().f1863d, 0, arrayList10, null);
                        }
                    }
                    HashSet<c> hashSet2 = i(c.a.RIGHT).f1860a;
                    if (hashSet2 != null) {
                        Iterator<c> it4 = hashSet2.iterator();
                        while (it4.hasNext()) {
                            b.e.a.b(it4.next().f1863d, 0, arrayList10, null);
                        }
                    }
                    HashSet<c> hashSet3 = i(aVar3).f1860a;
                    if (hashSet3 != null) {
                        Iterator<c> it5 = hashSet3.iterator();
                        while (it5.hasNext()) {
                            b.e.a.b(it5.next().f1863d, 0, arrayList10, null);
                        }
                    }
                    n nVar4 = null;
                    if (arrayList7 != null) {
                        Iterator it6 = arrayList7.iterator();
                        while (it6.hasNext()) {
                            b.e.a.b((d) it6.next(), 0, arrayList10, null);
                        }
                    }
                    if (arrayList5 != null) {
                        Iterator it7 = arrayList5.iterator();
                        while (it7.hasNext()) {
                            b.e.a.b((f) it7.next(), 1, arrayList10, null);
                        }
                    }
                    int i38 = 1;
                    if (arrayList6 != null) {
                        Iterator it8 = arrayList6.iterator();
                        while (it8.hasNext()) {
                            h hVar3 = (h) it8.next();
                            n b3 = b.e.a.b(hVar3, i38, arrayList10, nVar4);
                            hVar3.P(arrayList10, i38, b3);
                            b3.b(arrayList10);
                            nVar4 = null;
                            i38 = 1;
                        }
                    }
                    HashSet<c> hashSet4 = i(c.a.TOP).f1860a;
                    if (hashSet4 != null) {
                        Iterator<c> it9 = hashSet4.iterator();
                        while (it9.hasNext()) {
                            b.e.a.b(it9.next().f1863d, 1, arrayList10, null);
                        }
                    }
                    HashSet<c> hashSet5 = i(c.a.BASELINE).f1860a;
                    if (hashSet5 != null) {
                        Iterator<c> it10 = hashSet5.iterator();
                        while (it10.hasNext()) {
                            b.e.a.b(it10.next().f1863d, 1, arrayList10, null);
                        }
                    }
                    HashSet<c> hashSet6 = i(c.a.BOTTOM).f1860a;
                    if (hashSet6 != null) {
                        Iterator<c> it11 = hashSet6.iterator();
                        while (it11.hasNext()) {
                            b.e.a.b(it11.next().f1863d, 1, arrayList10, null);
                        }
                    }
                    HashSet<c> hashSet7 = i(aVar3).f1860a;
                    if (hashSet7 != null) {
                        Iterator<c> it12 = hashSet7.iterator();
                        while (it12.hasNext()) {
                            b.e.a.b(it12.next().f1863d, 1, arrayList10, null);
                        }
                    }
                    char c4 = 1;
                    if (arrayList8 != null) {
                        Iterator it13 = arrayList8.iterator();
                        while (it13.hasNext()) {
                            b.e.a.b((d) it13.next(), 1, arrayList10, null);
                        }
                    }
                    int i39 = 0;
                    while (i39 < size3) {
                        d dVar12 = arrayList2.get(i39);
                        int[] iArr2 = dVar12.O;
                        if (iArr2[0] == 3 && iArr2[c4] == 3) {
                            n c5 = b.e.a.c(arrayList10, dVar12.j0);
                            n c6 = b.e.a.c(arrayList10, dVar12.k0);
                            if (c5 != null && c6 != null) {
                                c5.d(0, c6);
                                c6.f1925d = 2;
                                arrayList10.remove(c5);
                            }
                        }
                        i39++;
                        c4 = 1;
                    }
                    if (arrayList10.size() > 1) {
                        if (m() == 2) {
                            Iterator<n> it14 = arrayList10.iterator();
                            nVar = null;
                            int i40 = 0;
                            while (it14.hasNext()) {
                                n next = it14.next();
                                if (next.f1925d != 1 && (c3 = next.c(this.q0, 0)) > i40) {
                                    nVar = next;
                                    i40 = c3;
                                }
                            }
                            if (nVar != null) {
                                this.O[0] = 1;
                                M(i40);
                                if (q() == 2) {
                                    Iterator<n> it15 = arrayList10.iterator();
                                    nVar2 = null;
                                    int i41 = 0;
                                    while (it15.hasNext()) {
                                        n next2 = it15.next();
                                        if (next2.f1925d != 0 && (c2 = next2.c(this.q0, 1)) > i41) {
                                            nVar2 = next2;
                                            i41 = c2;
                                        }
                                    }
                                    if (nVar2 != null) {
                                        this.O[1] = 1;
                                        H(i41);
                                        if (nVar == null || nVar2 != null) {
                                            z9 = true;
                                        }
                                    }
                                }
                                nVar2 = null;
                                if (nVar == null) {
                                }
                                z9 = true;
                            }
                        }
                        nVar = null;
                        if (q() == 2) {
                        }
                        nVar2 = null;
                        if (nVar == null) {
                        }
                        z9 = true;
                    }
                }
            }
            if (z9) {
                i2 = i11;
                if (i2 == 2) {
                    i13 = i9;
                    if (i13 < r() && i13 > 0) {
                        M(i13);
                        this.y0 = true;
                    } else {
                        i14 = r();
                        i5 = i12;
                        if (i5 != 2) {
                            i15 = i10;
                            if (i15 < l() && i15 > 0) {
                                H(i15);
                                this.z0 = true;
                            } else {
                                i16 = l();
                                i4 = i16;
                                i3 = i14;
                                z = true;
                                boolean z16 = !Y(64) || Y(128);
                                dVar = this.q0;
                                Objects.requireNonNull(dVar);
                                dVar.j = false;
                                if (this.x0 != 0 && z16) {
                                    dVar.j = true;
                                }
                                ArrayList<d> arrayList11 = this.l0;
                                if (m() != 2 || q() == 2) {
                                    i6 = 0;
                                    z2 = true;
                                } else {
                                    i6 = 0;
                                    z2 = false;
                                }
                                this.t0 = i6;
                                this.u0 = i6;
                                i7 = i;
                                for (i8 = 0; i8 < i7; i8++) {
                                    d dVar13 = this.l0.get(i8);
                                    if (dVar13 instanceof k) {
                                        ((k) dVar13).P();
                                    }
                                }
                                boolean Y = Y(64);
                                z3 = z;
                                int i42 = 0;
                                z4 = true;
                                while (z4) {
                                    int i43 = i42 + 1;
                                    try {
                                        this.q0.u();
                                        this.t0 = 0;
                                        this.u0 = 0;
                                        g(this.q0);
                                        for (int i44 = 0; i44 < i7; i44++) {
                                            this.l0.get(i44).g(this.q0);
                                        }
                                        R(this.q0);
                                        try {
                                            WeakReference<c> weakReference = this.A0;
                                            if (weakReference != null && weakReference.get() != null) {
                                                this.q0.f(this.q0.l(this.A0.get()), this.q0.l(this.E), 0, 5);
                                                this.A0 = null;
                                            }
                                            WeakReference<c> weakReference2 = this.C0;
                                            if (weakReference2 != null && weakReference2.get() != null) {
                                                this.q0.f(this.q0.l(this.G), this.q0.l(this.C0.get()), 0, 5);
                                                this.C0 = null;
                                            }
                                            WeakReference<c> weakReference3 = this.B0;
                                            if (weakReference3 != null && weakReference3.get() != null) {
                                                this.q0.f(this.q0.l(this.B0.get()), this.q0.l(this.D), 0, 5);
                                                this.B0 = null;
                                            }
                                            WeakReference<c> weakReference4 = this.D0;
                                            if (weakReference4 != null && weakReference4.get() != null) {
                                                this.q0.f(this.q0.l(this.F), this.q0.l(this.D0.get()), 0, 5);
                                                try {
                                                    this.D0 = null;
                                                } catch (Exception e2) {
                                                    e = e2;
                                                    z4 = true;
                                                    e.printStackTrace();
                                                    System.out.println("EXCEPTION : " + e);
                                                    if (!z4) {
                                                    }
                                                    if (z2) {
                                                    }
                                                    z5 = false;
                                                    max = Math.max(this.X, r());
                                                    if (max <= r()) {
                                                    }
                                                    max2 = Math.max(this.Y, l());
                                                    if (max2 <= l()) {
                                                    }
                                                    if (!z8) {
                                                    }
                                                    z4 = z7;
                                                    z3 = z8;
                                                    i42 = i43;
                                                }
                                            }
                                            this.q0.q();
                                            z4 = true;
                                        } catch (Exception e3) {
                                            e = e3;
                                        }
                                    } catch (Exception e4) {
                                        e = e4;
                                    }
                                    if (!z4) {
                                        b.h.b.d dVar14 = this.q0;
                                        i.f1883a[2] = false;
                                        boolean Y2 = Y(64);
                                        O(dVar14, Y2);
                                        int size4 = this.l0.size();
                                        for (int i45 = 0; i45 < size4; i45++) {
                                            this.l0.get(i45).O(dVar14, Y2);
                                        }
                                    } else {
                                        O(this.q0, Y);
                                        for (int i46 = 0; i46 < i7; i46++) {
                                            this.l0.get(i46).O(this.q0, Y);
                                        }
                                    }
                                    if (z2 || i43 >= 8 || !i.f1883a[2]) {
                                        z5 = false;
                                    } else {
                                        int i47 = 0;
                                        int i48 = 0;
                                        for (int i49 = 0; i49 < i7; i49++) {
                                            d dVar15 = this.l0.get(i49);
                                            i47 = Math.max(i47, dVar15.r() + dVar15.U);
                                            i48 = Math.max(i48, dVar15.l() + dVar15.V);
                                        }
                                        int max5 = Math.max(this.X, i47);
                                        int max6 = Math.max(this.Y, i48);
                                        if (i2 != 2 || r() >= max5) {
                                            z5 = false;
                                        } else {
                                            M(max5);
                                            this.O[0] = 2;
                                            z5 = true;
                                            z3 = true;
                                        }
                                        if (i5 == 2 && l() < max6) {
                                            H(max6);
                                            this.O[1] = 2;
                                            z5 = true;
                                            z3 = true;
                                        }
                                    }
                                    max = Math.max(this.X, r());
                                    if (max <= r()) {
                                        M(max);
                                        r6 = 1;
                                        this.O[0] = 1;
                                        z5 = true;
                                        z6 = true;
                                    } else {
                                        r6 = 1;
                                        z6 = z3;
                                    }
                                    max2 = Math.max(this.Y, l());
                                    if (max2 <= l()) {
                                        H(max2);
                                        this.O[r6] = r6;
                                        z8 = r6;
                                        z7 = z8;
                                    } else {
                                        z7 = z5;
                                        z8 = z6;
                                    }
                                    if (!z8) {
                                        if (this.O[0] == 2 && i3 > 0 && r() > i3) {
                                            this.y0 = r6;
                                            this.O[0] = r6;
                                            M(i3);
                                            z8 = r6;
                                            z7 = z8;
                                        }
                                        if (this.O[r6] == 2 && i4 > 0 && l() > i4) {
                                            this.z0 = r6;
                                            this.O[r6] = r6;
                                            H(i4);
                                            z3 = true;
                                            z4 = true;
                                            i42 = i43;
                                        }
                                    }
                                    z4 = z7;
                                    z3 = z8;
                                    i42 = i43;
                                }
                                this.l0 = arrayList11;
                                if (z3) {
                                    int[] iArr3 = this.O;
                                    iArr3[0] = i2;
                                    iArr3[1] = i5;
                                }
                                D(this.q0.o);
                            }
                        } else {
                            i15 = i10;
                        }
                        i16 = i15;
                        i4 = i16;
                        i3 = i14;
                        z = true;
                        if (Y(64)) {
                        }
                        dVar = this.q0;
                        Objects.requireNonNull(dVar);
                        dVar.j = false;
                        if (this.x0 != 0) {
                            dVar.j = true;
                        }
                        ArrayList<d> arrayList112 = this.l0;
                        if (m() != 2) {
                        }
                        i6 = 0;
                        z2 = true;
                        this.t0 = i6;
                        this.u0 = i6;
                        i7 = i;
                        while (i8 < i7) {
                        }
                        boolean Y3 = Y(64);
                        z3 = z;
                        int i422 = 0;
                        z4 = true;
                        while (z4) {
                        }
                        this.l0 = arrayList112;
                        if (z3) {
                        }
                        D(this.q0.o);
                    }
                } else {
                    i13 = i9;
                }
                i14 = i13;
                i5 = i12;
                if (i5 != 2) {
                }
                i16 = i15;
                i4 = i16;
                i3 = i14;
                z = true;
                if (Y(64)) {
                }
                dVar = this.q0;
                Objects.requireNonNull(dVar);
                dVar.j = false;
                if (this.x0 != 0) {
                }
                ArrayList<d> arrayList1122 = this.l0;
                if (m() != 2) {
                }
                i6 = 0;
                z2 = true;
                this.t0 = i6;
                this.u0 = i6;
                i7 = i;
                while (i8 < i7) {
                }
                boolean Y32 = Y(64);
                z3 = z;
                int i4222 = 0;
                z4 = true;
                while (z4) {
                }
                this.l0 = arrayList1122;
                if (z3) {
                }
                D(this.q0.o);
            }
            i4 = i10;
            i5 = i12;
            i3 = i9;
            i2 = i11;
        }
        z = false;
        if (Y(64)) {
        }
        dVar = this.q0;
        Objects.requireNonNull(dVar);
        dVar.j = false;
        if (this.x0 != 0) {
        }
        ArrayList<d> arrayList11222 = this.l0;
        if (m() != 2) {
        }
        i6 = 0;
        z2 = true;
        this.t0 = i6;
        this.u0 = i6;
        i7 = i;
        while (i8 < i7) {
        }
        boolean Y322 = Y(64);
        z3 = z;
        int i42222 = 0;
        z4 = true;
        while (z4) {
        }
        this.l0 = arrayList11222;
        if (z3) {
        }
        D(this.q0.o);
    }

    public void Q(d dVar, int i) {
        if (i == 0) {
            int i2 = this.t0 + 1;
            b[] bVarArr = this.w0;
            if (i2 >= bVarArr.length) {
                this.w0 = (b[]) Arrays.copyOf(bVarArr, bVarArr.length * 2);
            }
            b[] bVarArr2 = this.w0;
            int i3 = this.t0;
            bVarArr2[i3] = new b(dVar, 0, this.p0);
            this.t0 = i3 + 1;
        } else if (i == 1) {
            int i4 = this.u0 + 1;
            b[] bVarArr3 = this.v0;
            if (i4 >= bVarArr3.length) {
                this.v0 = (b[]) Arrays.copyOf(bVarArr3, bVarArr3.length * 2);
            }
            b[] bVarArr4 = this.v0;
            int i5 = this.u0;
            bVarArr4[i5] = new b(dVar, 1, this.p0);
            this.u0 = i5 + 1;
        }
    }

    public boolean R(b.h.b.d dVar) {
        boolean Y = Y(64);
        d(dVar, Y);
        int size = this.l0.size();
        boolean z = false;
        for (int i = 0; i < size; i++) {
            d dVar2 = this.l0.get(i);
            boolean[] zArr = dVar2.N;
            zArr[0] = false;
            zArr[1] = false;
            if (dVar2 instanceof a) {
                z = true;
            }
        }
        if (z) {
            for (int i2 = 0; i2 < size; i2++) {
                d dVar3 = this.l0.get(i2);
                if (dVar3 instanceof a) {
                    a aVar = (a) dVar3;
                    for (int i3 = 0; i3 < aVar.m0; i3++) {
                        d dVar4 = aVar.l0[i3];
                        int i4 = aVar.n0;
                        if (i4 == 0 || i4 == 1) {
                            dVar4.N[0] = true;
                        } else if (i4 == 2 || i4 == 3) {
                            dVar4.N[1] = true;
                        }
                    }
                }
            }
        }
        for (int i5 = 0; i5 < size; i5++) {
            d dVar5 = this.l0.get(i5);
            if (dVar5.c()) {
                dVar5.d(dVar, Y);
            }
        }
        if (b.h.b.d.f1829a) {
            HashSet<d> hashSet = new HashSet<>();
            for (int i6 = 0; i6 < size; i6++) {
                d dVar6 = this.l0.get(i6);
                if (!dVar6.c()) {
                    hashSet.add(dVar6);
                }
            }
            b(this, dVar, hashSet, m() == 2 ? 0 : 1, false);
            Iterator<d> it = hashSet.iterator();
            while (it.hasNext()) {
                d next = it.next();
                i.a(this, dVar, next);
                next.d(dVar, Y);
            }
        } else {
            for (int i7 = 0; i7 < size; i7++) {
                d dVar7 = this.l0.get(i7);
                if (dVar7 instanceof e) {
                    int[] iArr = dVar7.O;
                    int i8 = iArr[0];
                    int i9 = iArr[1];
                    if (i8 == 2) {
                        iArr[0] = 1;
                    }
                    if (i9 == 2) {
                        iArr[1] = 1;
                    }
                    dVar7.d(dVar, Y);
                    if (i8 == 2) {
                        dVar7.I(i8);
                    }
                    if (i9 == 2) {
                        dVar7.L(i9);
                    }
                } else {
                    i.a(this, dVar, dVar7);
                    if (!dVar7.c()) {
                        dVar7.d(dVar, Y);
                    }
                }
            }
        }
        if (this.t0 > 0) {
            b.e.a.a(this, dVar, null, 0);
        }
        if (this.u0 > 0) {
            b.e.a.a(this, dVar, null, 1);
        }
        return true;
    }

    public void S(c cVar) {
        WeakReference<c> weakReference = this.D0;
        if (weakReference == null || weakReference.get() == null || cVar.c() > this.D0.get().c()) {
            this.D0 = new WeakReference<>(cVar);
        }
    }

    public void T(c cVar) {
        WeakReference<c> weakReference = this.C0;
        if (weakReference == null || weakReference.get() == null || cVar.c() > this.C0.get().c()) {
            this.C0 = new WeakReference<>(cVar);
        }
    }

    public void U(c cVar) {
        WeakReference<c> weakReference = this.A0;
        if (weakReference == null || weakReference.get() == null || cVar.c() > this.A0.get().c()) {
            this.A0 = new WeakReference<>(cVar);
        }
    }

    public boolean V(boolean z, int i) {
        boolean z2;
        b.h.b.i.l.e eVar = this.n0;
        boolean z3 = true;
        boolean z4 = z & true;
        int k = eVar.f1895a.k(0);
        int k2 = eVar.f1895a.k(1);
        int s = eVar.f1895a.s();
        int t = eVar.f1895a.t();
        if (z4 && (k == 2 || k2 == 2)) {
            Iterator<o> it = eVar.f1899e.iterator();
            while (true) {
                if (!it.hasNext()) {
                    break;
                }
                o next = it.next();
                if (next.f1933f == i && !next.k()) {
                    z4 = false;
                    break;
                }
            }
            if (i == 0) {
                if (z4 && k == 2) {
                    e eVar2 = eVar.f1895a;
                    eVar2.O[0] = 1;
                    eVar2.M(eVar.d(eVar2, 0));
                    e eVar3 = eVar.f1895a;
                    eVar3.f1878d.f1932e.c(eVar3.r());
                }
            } else if (z4 && k2 == 2) {
                e eVar4 = eVar.f1895a;
                eVar4.O[1] = 1;
                eVar4.H(eVar.d(eVar4, 1));
                e eVar5 = eVar.f1895a;
                eVar5.f1879e.f1932e.c(eVar5.l());
            }
        }
        if (i == 0) {
            e eVar6 = eVar.f1895a;
            int[] iArr = eVar6.O;
            if (iArr[0] == 1 || iArr[0] == 4) {
                int r = eVar6.r() + s;
                eVar.f1895a.f1878d.i.c(r);
                eVar.f1895a.f1878d.f1932e.c(r - s);
                z2 = true;
            }
            z2 = false;
        } else {
            e eVar7 = eVar.f1895a;
            int[] iArr2 = eVar7.O;
            if (iArr2[1] == 1 || iArr2[1] == 4) {
                int l = eVar7.l() + t;
                eVar.f1895a.f1879e.i.c(l);
                eVar.f1895a.f1879e.f1932e.c(l - t);
                z2 = true;
            }
            z2 = false;
        }
        eVar.g();
        Iterator<o> it2 = eVar.f1899e.iterator();
        while (it2.hasNext()) {
            o next2 = it2.next();
            if (next2.f1933f == i && (next2.f1929b != eVar.f1895a || next2.f1934g)) {
                next2.e();
            }
        }
        Iterator<o> it3 = eVar.f1899e.iterator();
        while (it3.hasNext()) {
            o next3 = it3.next();
            if (next3.f1933f == i && (z2 || next3.f1929b != eVar.f1895a)) {
                if (!next3.f1935h.j || !next3.i.j || (!(next3 instanceof b.h.b.i.l.c) && !next3.f1932e.j)) {
                    z3 = false;
                    break;
                }
            }
        }
        eVar.f1895a.I(k);
        eVar.f1895a.L(k2);
        return z3;
    }

    public void W() {
        this.n0.f1896b = true;
    }

    public boolean Y(int i) {
        return (this.x0 & i) == i;
    }

    public void Z(int i) {
        this.x0 = i;
        b.h.b.d.f1829a = Y(512);
    }
}