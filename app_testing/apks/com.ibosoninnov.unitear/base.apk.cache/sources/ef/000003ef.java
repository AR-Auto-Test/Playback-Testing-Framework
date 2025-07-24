package b.h.b.i;

import b.d.b.m0;
import b.h.b.i.c;
import b.h.b.i.l.m;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

/* compiled from: ConstraintWidget.java */
/* loaded from: classes.dex */
public class d {
    public c D;
    public c E;
    public c F;
    public c G;
    public c H;
    public c I;
    public c J;
    public c K;
    public c[] L;
    public ArrayList<c> M;
    public boolean[] N;
    public int[] O;
    public d P;
    public int Q;
    public int R;
    public float S;
    public int T;
    public int U;
    public int V;
    public int W;
    public int X;
    public int Y;
    public float Z;
    public float a0;

    /* renamed from: b  reason: collision with root package name */
    public b.h.b.i.l.c f1876b;
    public Object b0;

    /* renamed from: c  reason: collision with root package name */
    public b.h.b.i.l.c f1877c;
    public int c0;
    public String d0;
    public int e0;
    public int f0;
    public float[] g0;
    public d[] h0;
    public d[] i0;
    public int j0;
    public int k0;
    public boolean z;

    /* renamed from: a  reason: collision with root package name */
    public boolean f1875a = false;

    /* renamed from: d  reason: collision with root package name */
    public b.h.b.i.l.k f1878d = null;

    /* renamed from: e  reason: collision with root package name */
    public m f1879e = null;

    /* renamed from: f  reason: collision with root package name */
    public boolean[] f1880f = {true, true};

    /* renamed from: g  reason: collision with root package name */
    public boolean f1881g = true;

    /* renamed from: h  reason: collision with root package name */
    public boolean f1882h = false;
    public boolean i = false;
    public int j = -1;
    public int k = -1;
    public int l = 0;
    public int m = 0;
    public int[] n = new int[2];
    public int o = 0;
    public int p = 0;
    public float q = 1.0f;
    public int r = 0;
    public int s = 0;
    public float t = 1.0f;
    public int u = -1;
    public float v = 1.0f;
    public int[] w = {Integer.MAX_VALUE, Integer.MAX_VALUE};
    public float x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    public boolean y = false;
    public boolean A = false;
    public int B = 0;
    public int C = 0;

    public d() {
        c cVar = new c(this, c.a.LEFT);
        this.D = cVar;
        c cVar2 = new c(this, c.a.TOP);
        this.E = cVar2;
        c cVar3 = new c(this, c.a.RIGHT);
        this.F = cVar3;
        c cVar4 = new c(this, c.a.BOTTOM);
        this.G = cVar4;
        c cVar5 = new c(this, c.a.BASELINE);
        this.H = cVar5;
        this.I = new c(this, c.a.CENTER_X);
        this.J = new c(this, c.a.CENTER_Y);
        c cVar6 = new c(this, c.a.CENTER);
        this.K = cVar6;
        this.L = new c[]{cVar, cVar3, cVar2, cVar4, cVar5, cVar6};
        ArrayList<c> arrayList = new ArrayList<>();
        this.M = arrayList;
        this.N = new boolean[2];
        this.O = new int[]{1, 1};
        this.P = null;
        this.Q = 0;
        this.R = 0;
        this.S = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.T = -1;
        this.U = 0;
        this.V = 0;
        this.W = 0;
        this.Z = 0.5f;
        this.a0 = 0.5f;
        this.c0 = 0;
        this.d0 = null;
        this.e0 = 0;
        this.f0 = 0;
        this.g0 = new float[]{-1.0f, -1.0f};
        this.h0 = new d[]{null, null};
        this.i0 = new d[]{null, null};
        this.j0 = -1;
        this.k0 = -1;
        arrayList.add(this.D);
        this.M.add(this.E);
        this.M.add(this.F);
        this.M.add(this.G);
        this.M.add(this.I);
        this.M.add(this.J);
        this.M.add(this.K);
        this.M.add(this.H);
    }

    public boolean A() {
        return this.i || (this.E.f1862c && this.G.f1862c);
    }

    public void B() {
        this.D.h();
        this.E.h();
        this.F.h();
        this.G.h();
        this.H.h();
        this.I.h();
        this.J.h();
        this.K.h();
        this.P = null;
        this.x = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.Q = 0;
        this.R = 0;
        this.S = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.T = -1;
        this.U = 0;
        this.V = 0;
        this.W = 0;
        this.X = 0;
        this.Y = 0;
        this.Z = 0.5f;
        this.a0 = 0.5f;
        int[] iArr = this.O;
        iArr[0] = 1;
        iArr[1] = 1;
        this.b0 = null;
        this.c0 = 0;
        this.e0 = 0;
        this.f0 = 0;
        float[] fArr = this.g0;
        fArr[0] = -1.0f;
        fArr[1] = -1.0f;
        this.j = -1;
        this.k = -1;
        int[] iArr2 = this.w;
        iArr2[0] = Integer.MAX_VALUE;
        iArr2[1] = Integer.MAX_VALUE;
        this.l = 0;
        this.m = 0;
        this.q = 1.0f;
        this.t = 1.0f;
        this.p = Integer.MAX_VALUE;
        this.s = Integer.MAX_VALUE;
        this.o = 0;
        this.r = 0;
        this.u = -1;
        this.v = 1.0f;
        boolean[] zArr = this.f1880f;
        zArr[0] = true;
        zArr[1] = true;
        this.A = false;
        boolean[] zArr2 = this.N;
        zArr2[0] = false;
        zArr2[1] = false;
        this.f1881g = true;
    }

    public void C() {
        this.f1882h = false;
        this.i = false;
        int size = this.M.size();
        for (int i = 0; i < size; i++) {
            c cVar = this.M.get(i);
            cVar.f1862c = false;
            cVar.f1861b = 0;
        }
    }

    public void D(b.h.b.c cVar) {
        this.D.i();
        this.E.i();
        this.F.i();
        this.G.i();
        this.H.i();
        this.K.i();
        this.I.i();
        this.J.i();
    }

    public void E(int i) {
        this.W = i;
        this.y = i > 0;
    }

    public void F(int i, int i2) {
        c cVar = this.D;
        cVar.f1861b = i;
        cVar.f1862c = true;
        c cVar2 = this.F;
        cVar2.f1861b = i2;
        cVar2.f1862c = true;
        this.U = i;
        this.Q = i2 - i;
        this.f1882h = true;
    }

    public void G(int i, int i2) {
        c cVar = this.E;
        cVar.f1861b = i;
        cVar.f1862c = true;
        c cVar2 = this.G;
        cVar2.f1861b = i2;
        cVar2.f1862c = true;
        this.V = i;
        this.R = i2 - i;
        if (this.y) {
            this.H.j(i + this.W);
        }
        this.i = true;
    }

    public void H(int i) {
        this.R = i;
        int i2 = this.Y;
        if (i < i2) {
            this.R = i2;
        }
    }

    public void I(int i) {
        this.O[0] = i;
    }

    public void J(int i) {
        if (i < 0) {
            this.Y = 0;
        } else {
            this.Y = i;
        }
    }

    public void K(int i) {
        if (i < 0) {
            this.X = 0;
        } else {
            this.X = i;
        }
    }

    public void L(int i) {
        this.O[1] = i;
    }

    public void M(int i) {
        this.Q = i;
        int i2 = this.X;
        if (i < i2) {
            this.Q = i2;
        }
    }

    public void N(boolean z, boolean z2) {
        int i;
        int i2;
        b.h.b.i.l.k kVar = this.f1878d;
        boolean z3 = z & kVar.f1934g;
        m mVar = this.f1879e;
        boolean z4 = z2 & mVar.f1934g;
        int i3 = kVar.f1935h.f1909g;
        int i4 = mVar.f1935h.f1909g;
        int i5 = kVar.i.f1909g;
        int i6 = mVar.i.f1909g;
        int i7 = i6 - i4;
        if (i5 - i3 < 0 || i7 < 0 || i3 == Integer.MIN_VALUE || i3 == Integer.MAX_VALUE || i4 == Integer.MIN_VALUE || i4 == Integer.MAX_VALUE || i5 == Integer.MIN_VALUE || i5 == Integer.MAX_VALUE || i6 == Integer.MIN_VALUE || i6 == Integer.MAX_VALUE) {
            i5 = 0;
            i6 = 0;
            i3 = 0;
            i4 = 0;
        }
        int i8 = i5 - i3;
        int i9 = i6 - i4;
        if (z3) {
            this.U = i3;
        }
        if (z4) {
            this.V = i4;
        }
        if (this.c0 == 8) {
            this.Q = 0;
            this.R = 0;
            return;
        }
        if (z3) {
            if (this.O[0] == 1 && i8 < (i2 = this.Q)) {
                i8 = i2;
            }
            this.Q = i8;
            int i10 = this.X;
            if (i8 < i10) {
                this.Q = i10;
            }
        }
        if (z4) {
            if (this.O[1] == 1 && i9 < (i = this.R)) {
                i9 = i;
            }
            this.R = i9;
            int i11 = this.Y;
            if (i9 < i11) {
                this.R = i11;
            }
        }
    }

    public void O(b.h.b.d dVar, boolean z) {
        int i;
        int i2;
        m mVar;
        b.h.b.i.l.k kVar;
        int o = dVar.o(this.D);
        int o2 = dVar.o(this.E);
        int o3 = dVar.o(this.F);
        int o4 = dVar.o(this.G);
        if (z && (kVar = this.f1878d) != null) {
            b.h.b.i.l.f fVar = kVar.f1935h;
            if (fVar.j) {
                b.h.b.i.l.f fVar2 = kVar.i;
                if (fVar2.j) {
                    o = fVar.f1909g;
                    o3 = fVar2.f1909g;
                }
            }
        }
        if (z && (mVar = this.f1879e) != null) {
            b.h.b.i.l.f fVar3 = mVar.f1935h;
            if (fVar3.j) {
                b.h.b.i.l.f fVar4 = mVar.i;
                if (fVar4.j) {
                    o2 = fVar3.f1909g;
                    o4 = fVar4.f1909g;
                }
            }
        }
        int i3 = o4 - o2;
        if (o3 - o < 0 || i3 < 0 || o == Integer.MIN_VALUE || o == Integer.MAX_VALUE || o2 == Integer.MIN_VALUE || o2 == Integer.MAX_VALUE || o3 == Integer.MIN_VALUE || o3 == Integer.MAX_VALUE || o4 == Integer.MIN_VALUE || o4 == Integer.MAX_VALUE) {
            o4 = 0;
            o = 0;
            o2 = 0;
            o3 = 0;
        }
        int i4 = o3 - o;
        int i5 = o4 - o2;
        this.U = o;
        this.V = o2;
        if (this.c0 == 8) {
            this.Q = 0;
            this.R = 0;
            return;
        }
        int[] iArr = this.O;
        if (iArr[0] == 1 && i4 < (i2 = this.Q)) {
            i4 = i2;
        }
        if (iArr[1] == 1 && i5 < (i = this.R)) {
            i5 = i;
        }
        this.Q = i4;
        this.R = i5;
        int i6 = this.Y;
        if (i5 < i6) {
            this.R = i6;
        }
        int i7 = this.X;
        if (i4 < i7) {
            this.Q = i7;
        }
    }

    public void b(e eVar, b.h.b.d dVar, HashSet<d> hashSet, int i, boolean z) {
        if (z) {
            if (!hashSet.contains(this)) {
                return;
            }
            i.a(eVar, dVar, this);
            hashSet.remove(this);
            d(dVar, eVar.Y(64));
        }
        if (i == 0) {
            HashSet<c> hashSet2 = this.D.f1860a;
            if (hashSet2 != null) {
                Iterator<c> it = hashSet2.iterator();
                while (it.hasNext()) {
                    it.next().f1863d.b(eVar, dVar, hashSet, i, true);
                }
            }
            HashSet<c> hashSet3 = this.F.f1860a;
            if (hashSet3 != null) {
                Iterator<c> it2 = hashSet3.iterator();
                while (it2.hasNext()) {
                    it2.next().f1863d.b(eVar, dVar, hashSet, i, true);
                }
                return;
            }
            return;
        }
        HashSet<c> hashSet4 = this.E.f1860a;
        if (hashSet4 != null) {
            Iterator<c> it3 = hashSet4.iterator();
            while (it3.hasNext()) {
                it3.next().f1863d.b(eVar, dVar, hashSet, i, true);
            }
        }
        HashSet<c> hashSet5 = this.G.f1860a;
        if (hashSet5 != null) {
            Iterator<c> it4 = hashSet5.iterator();
            while (it4.hasNext()) {
                it4.next().f1863d.b(eVar, dVar, hashSet, i, true);
            }
        }
        HashSet<c> hashSet6 = this.H.f1860a;
        if (hashSet6 != null) {
            Iterator<c> it5 = hashSet6.iterator();
            while (it5.hasNext()) {
                it5.next().f1863d.b(eVar, dVar, hashSet, i, true);
            }
        }
    }

    public boolean c() {
        return (this instanceof j) || (this instanceof f);
    }

    /* JADX WARN: Removed duplicated region for block: B:229:0x0367  */
    /* JADX WARN: Removed duplicated region for block: B:233:0x0371  */
    /* JADX WARN: Removed duplicated region for block: B:236:0x0376  */
    /* JADX WARN: Removed duplicated region for block: B:243:0x038a  */
    /* JADX WARN: Removed duplicated region for block: B:248:0x0393  */
    /* JADX WARN: Removed duplicated region for block: B:249:0x0396  */
    /* JADX WARN: Removed duplicated region for block: B:252:0x03af  */
    /* JADX WARN: Removed duplicated region for block: B:263:0x03c8  */
    /* JADX WARN: Removed duplicated region for block: B:274:0x0402  */
    /* JADX WARN: Removed duplicated region for block: B:275:0x040a  */
    /* JADX WARN: Removed duplicated region for block: B:278:0x0410  */
    /* JADX WARN: Removed duplicated region for block: B:279:0x0418  */
    /* JADX WARN: Removed duplicated region for block: B:282:0x0438  */
    /* JADX WARN: Removed duplicated region for block: B:283:0x043b  */
    /* JADX WARN: Removed duplicated region for block: B:287:0x049c  */
    /* JADX WARN: Removed duplicated region for block: B:304:0x0500  */
    /* JADX WARN: Removed duplicated region for block: B:308:0x0514  */
    /* JADX WARN: Removed duplicated region for block: B:309:0x0516  */
    /* JADX WARN: Removed duplicated region for block: B:311:0x0519  */
    /* JADX WARN: Removed duplicated region for block: B:346:0x059f  */
    /* JADX WARN: Removed duplicated region for block: B:347:0x05a2  */
    /* JADX WARN: Removed duplicated region for block: B:351:0x05e9  */
    /* JADX WARN: Removed duplicated region for block: B:355:0x0614  */
    /* JADX WARN: Removed duplicated region for block: B:358:0x061e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void d(b.h.b.d dVar, boolean z) {
        boolean z2;
        boolean z3;
        d dVar2;
        d dVar3;
        boolean z4;
        boolean z5;
        b.h.b.h hVar;
        b.h.b.h hVar2;
        int i;
        int i2;
        int i3;
        int i4;
        boolean z6;
        int i5;
        boolean z7;
        boolean z8;
        int i6;
        boolean z9;
        boolean z10;
        b.h.b.h hVar3;
        b.h.b.h hVar4;
        b.h.b.h hVar5;
        b.h.b.h hVar6;
        b.h.b.h hVar7;
        int i7;
        int i8;
        int i9;
        d dVar4;
        b.h.b.d dVar5;
        b.h.b.h hVar8;
        b.h.b.h hVar9;
        b.h.b.h hVar10;
        int i10;
        b.h.b.h hVar11;
        b.h.b.h hVar12;
        d dVar6;
        boolean z11;
        b.h.b.i.l.k kVar;
        b.h.b.i.l.f fVar;
        int i11;
        int i12;
        int i13;
        boolean w;
        boolean x;
        b.h.b.i.l.k kVar2;
        m mVar;
        boolean z12;
        b.h.b.h l = dVar.l(this.D);
        b.h.b.h l2 = dVar.l(this.F);
        b.h.b.h l3 = dVar.l(this.E);
        b.h.b.h l4 = dVar.l(this.G);
        b.h.b.h l5 = dVar.l(this.H);
        d dVar7 = this.P;
        if (dVar7 != null) {
            int[] iArr = dVar7.O;
            boolean z13 = iArr[0] == 2;
            z2 = iArr[1] == 2;
            z3 = z13;
        } else {
            z2 = false;
            z3 = false;
        }
        if (this.c0 == 8) {
            int size = this.M.size();
            int i14 = 0;
            while (true) {
                if (i14 >= size) {
                    z12 = false;
                    break;
                } else if (this.M.get(i14).f()) {
                    z12 = true;
                    break;
                } else {
                    i14++;
                }
            }
            if (!z12) {
                boolean[] zArr = this.N;
                if (!zArr[0] && !zArr[1]) {
                    return;
                }
            }
        }
        boolean z14 = this.f1882h;
        if (z14 || this.i) {
            if (z14) {
                dVar.e(l, this.U);
                dVar.e(l2, this.U + this.Q);
                if (z3 && (dVar3 = this.P) != null) {
                    e eVar = (e) dVar3;
                    eVar.U(this.D);
                    eVar.S(this.F);
                }
            }
            if (this.i) {
                dVar.e(l3, this.V);
                dVar.e(l4, this.V + this.R);
                if (this.H.f()) {
                    dVar.e(l5, this.V + this.W);
                }
                if (z2 && (dVar2 = this.P) != null) {
                    e eVar2 = (e) dVar2;
                    eVar2.U(this.E);
                    eVar2.T(this.G);
                }
            }
            if (this.f1882h && this.i) {
                this.f1882h = false;
                this.i = false;
                return;
            }
        }
        if (z && (kVar2 = this.f1878d) != null && (mVar = this.f1879e) != null) {
            b.h.b.i.l.f fVar2 = kVar2.f1935h;
            if (fVar2.j && kVar2.i.j && mVar.f1935h.j && mVar.i.j) {
                dVar.e(l, fVar2.f1909g);
                dVar.e(l2, this.f1878d.i.f1909g);
                dVar.e(l3, this.f1879e.f1935h.f1909g);
                dVar.e(l4, this.f1879e.i.f1909g);
                dVar.e(l5, this.f1879e.k.f1909g);
                if (this.P != null) {
                    if (z3 && this.f1880f[0] && !w()) {
                        dVar.f(dVar.l(this.P.F), l2, 0, 8);
                    }
                    if (z2 && this.f1880f[1] && !x()) {
                        dVar.f(dVar.l(this.P.G), l4, 0, 8);
                    }
                }
                this.f1882h = false;
                this.i = false;
                return;
            }
        }
        if (this.P != null) {
            if (v(0)) {
                ((e) this.P).Q(this, 0);
                w = true;
            } else {
                w = w();
            }
            if (v(1)) {
                ((e) this.P).Q(this, 1);
                x = true;
            } else {
                x = x();
            }
            if (!w && z3 && this.c0 != 8 && this.D.f1865f == null && this.F.f1865f == null) {
                dVar.f(dVar.l(this.P.F), l2, 0, 1);
            }
            if (!x && z2 && this.c0 != 8 && this.E.f1865f == null && this.G.f1865f == null && this.H == null) {
                dVar.f(dVar.l(this.P.G), l4, 0, 1);
            }
            z5 = w;
            z4 = x;
        } else {
            z4 = false;
            z5 = false;
        }
        int i15 = this.Q;
        int i16 = this.X;
        if (i15 >= i16) {
            i16 = i15;
        }
        int i17 = this.R;
        int i18 = this.Y;
        if (i17 >= i18) {
            i18 = i17;
        }
        int[] iArr2 = this.O;
        boolean z15 = iArr2[0] != 3;
        boolean z16 = iArr2[1] != 3;
        int i19 = this.T;
        this.u = i19;
        float f2 = this.S;
        this.v = f2;
        int i20 = i16;
        int i21 = this.l;
        int i22 = i18;
        int i23 = this.m;
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            hVar = l4;
            if (this.c0 != 8) {
                if (iArr2[0] == 3 && i21 == 0) {
                    i21 = 3;
                }
                hVar2 = l3;
                if (iArr2[1] == 3 && i23 == 0) {
                    i23 = 3;
                }
                if (iArr2[0] != 3) {
                    i12 = 0;
                } else if (iArr2[1] == 3 && i21 == 3 && i23 == 3) {
                    if (i19 == -1) {
                        if (z15 && !z16) {
                            this.u = 0;
                        } else if (!z15 && z16) {
                            this.u = 1;
                            if (i19 == -1) {
                                this.v = 1.0f / f2;
                            }
                        }
                    }
                    if (this.u == 0 && (!this.E.g() || !this.G.g())) {
                        this.u = 1;
                    } else if (this.u == 1 && (!this.D.g() || !this.F.g())) {
                        this.u = 0;
                    }
                    if (this.u == -1 && (!this.E.g() || !this.G.g() || !this.D.g() || !this.F.g())) {
                        if (this.E.g() && this.G.g()) {
                            this.u = 0;
                        } else if (this.D.g() && this.F.g()) {
                            this.v = 1.0f / this.v;
                            this.u = 1;
                        }
                    }
                    if (this.u == -1) {
                        int i24 = this.o;
                        if (i24 > 0 && this.r == 0) {
                            this.u = 0;
                        } else if (i24 == 0 && this.r > 0) {
                            this.v = 1.0f / this.v;
                            this.u = 1;
                        }
                    }
                    i13 = i22;
                    i4 = i13;
                    i = i21;
                    i2 = i23;
                    i3 = i20;
                    z6 = true;
                    int[] iArr3 = this.n;
                    iArr3[0] = i;
                    iArr3[1] = i2;
                    if (z6) {
                        i5 = -1;
                    } else {
                        int i25 = this.u;
                        i5 = -1;
                        if (i25 == 0 || i25 == -1) {
                            z7 = true;
                            boolean z17 = !z6 && ((i11 = this.u) == 1 || i11 == i5);
                            z8 = this.O[0] != 2 && (this instanceof e);
                            i6 = z8 ? 0 : i3;
                            z9 = !this.K.g();
                            boolean[] zArr2 = this.N;
                            z10 = zArr2[0];
                            boolean z18 = zArr2[1];
                            if (this.j != 2 && !this.f1882h) {
                                if (z && (kVar = this.f1878d) != null) {
                                    fVar = kVar.f1935h;
                                    if (fVar.j && kVar.i.j) {
                                        if (z) {
                                            dVar.e(l, fVar.f1909g);
                                            dVar.e(l2, this.f1878d.i.f1909g);
                                            if (this.P != null && z3 && this.f1880f[0] && !w()) {
                                                dVar.f(dVar.l(this.P.F), l2, 0, 8);
                                            }
                                        }
                                    }
                                }
                                d dVar8 = this.P;
                                b.h.b.h l6 = dVar8 == null ? dVar.l(dVar8.F) : null;
                                d dVar9 = this.P;
                                b.h.b.h l7 = dVar9 == null ? dVar.l(dVar9.D) : null;
                                boolean z19 = this.f1880f[0];
                                int[] iArr4 = this.O;
                                hVar3 = l5;
                                hVar4 = hVar;
                                hVar5 = hVar2;
                                hVar6 = l2;
                                hVar7 = l;
                                f(dVar, true, z3, z2, z19, l7, l6, iArr4[0], z8, this.D, this.F, this.U, i6, this.X, this.w[0], this.Z, z7, iArr4[1] != 3, z5, z4, z10, i, i2, this.o, this.p, this.q, z9);
                                if (z) {
                                    dVar4 = this;
                                    m mVar2 = dVar4.f1879e;
                                    if (mVar2 != null) {
                                        b.h.b.i.l.f fVar3 = mVar2.f1935h;
                                        if (fVar3.j && mVar2.i.j) {
                                            dVar5 = dVar;
                                            hVar10 = hVar5;
                                            dVar5.e(hVar10, fVar3.f1909g);
                                            hVar9 = hVar4;
                                            dVar5.e(hVar9, dVar4.f1879e.i.f1909g);
                                            hVar8 = hVar3;
                                            dVar5.e(hVar8, dVar4.f1879e.k.f1909g);
                                            d dVar10 = dVar4.P;
                                            if (dVar10 == null || z4 || !z2) {
                                                i7 = 8;
                                                i8 = 0;
                                                i9 = 1;
                                            } else {
                                                i9 = 1;
                                                if (dVar4.f1880f[1]) {
                                                    i7 = 8;
                                                    i8 = 0;
                                                    dVar5.f(dVar5.l(dVar10.G), hVar9, 0, 8);
                                                } else {
                                                    i7 = 8;
                                                    i8 = 0;
                                                }
                                            }
                                            i10 = i8;
                                            if ((dVar4.k != 2 ? i8 : i10) != 0 || dVar4.i) {
                                                hVar11 = hVar9;
                                                hVar12 = hVar10;
                                            } else {
                                                boolean z20 = (dVar4.O[i9] == 2 && (dVar4 instanceof e)) ? i9 : i8;
                                                if (z20) {
                                                    i4 = i8;
                                                }
                                                d dVar11 = dVar4.P;
                                                b.h.b.h l8 = dVar11 != null ? dVar5.l(dVar11.G) : null;
                                                d dVar12 = dVar4.P;
                                                b.h.b.h l9 = dVar12 != null ? dVar5.l(dVar12.E) : null;
                                                int i26 = dVar4.W;
                                                if (i26 > 0 || dVar4.c0 == i7) {
                                                    if (dVar4.H.f1865f != null) {
                                                        dVar5.d(hVar8, hVar10, i26, i7);
                                                        dVar5.d(hVar8, dVar5.l(dVar4.H.f1865f), i8, i7);
                                                        if (z2) {
                                                            dVar5.f(l8, dVar5.l(dVar4.G), i8, 5);
                                                        }
                                                        z11 = i8;
                                                        boolean z21 = dVar4.f1880f[i9];
                                                        int[] iArr5 = dVar4.O;
                                                        boolean z22 = z2;
                                                        boolean z23 = z3;
                                                        hVar11 = hVar9;
                                                        hVar12 = hVar10;
                                                        f(dVar, false, z22, z23, z21, l9, l8, iArr5[i9], z20, dVar4.E, dVar4.G, dVar4.V, i4, dVar4.Y, dVar4.w[i9], dVar4.a0, z17, iArr5[0] != 3, z4, z5, z18, i2, i, dVar4.r, dVar4.s, dVar4.t, z11);
                                                    } else if (dVar4.c0 == i7) {
                                                        dVar5.d(hVar8, hVar10, i8, i7);
                                                    } else {
                                                        dVar5.d(hVar8, hVar10, i26, i7);
                                                    }
                                                }
                                                z11 = z9;
                                                boolean z212 = dVar4.f1880f[i9];
                                                int[] iArr52 = dVar4.O;
                                                boolean z222 = z2;
                                                boolean z232 = z3;
                                                hVar11 = hVar9;
                                                hVar12 = hVar10;
                                                f(dVar, false, z222, z232, z212, l9, l8, iArr52[i9], z20, dVar4.E, dVar4.G, dVar4.V, i4, dVar4.Y, dVar4.w[i9], dVar4.a0, z17, iArr52[0] != 3, z4, z5, z18, i2, i, dVar4.r, dVar4.s, dVar4.t, z11);
                                            }
                                            if (z6) {
                                                dVar6 = this;
                                            } else {
                                                dVar6 = this;
                                                if (dVar6.u == 1) {
                                                    dVar.h(hVar11, hVar12, hVar6, hVar7, dVar6.v, 8);
                                                } else {
                                                    dVar.h(hVar6, hVar7, hVar11, hVar12, dVar6.v, 8);
                                                }
                                            }
                                            if (dVar6.K.g()) {
                                                d dVar13 = dVar6.K.f1865f.f1863d;
                                                int d2 = dVar6.K.d();
                                                c.a aVar = c.a.LEFT;
                                                b.h.b.h l10 = dVar.l(dVar6.i(aVar));
                                                c.a aVar2 = c.a.TOP;
                                                b.h.b.h l11 = dVar.l(dVar6.i(aVar2));
                                                c.a aVar3 = c.a.RIGHT;
                                                b.h.b.h l12 = dVar.l(dVar6.i(aVar3));
                                                c.a aVar4 = c.a.BOTTOM;
                                                b.h.b.h l13 = dVar.l(dVar6.i(aVar4));
                                                b.h.b.h l14 = dVar.l(dVar13.i(aVar));
                                                b.h.b.h l15 = dVar.l(dVar13.i(aVar2));
                                                b.h.b.h l16 = dVar.l(dVar13.i(aVar3));
                                                b.h.b.h l17 = dVar.l(dVar13.i(aVar4));
                                                b.h.b.b m = dVar.m();
                                                double radians = (float) Math.toRadians(dVar6.x + 90.0f);
                                                double d3 = d2;
                                                m.g(l11, l13, l15, l17, (float) (Math.sin(radians) * d3));
                                                dVar.c(m);
                                                b.h.b.b m2 = dVar.m();
                                                m2.g(l10, l12, l14, l16, (float) (Math.cos(radians) * d3));
                                                dVar.c(m2);
                                            }
                                            this.f1882h = false;
                                            this.i = false;
                                        }
                                    }
                                    dVar5 = dVar;
                                    hVar8 = hVar3;
                                    hVar9 = hVar4;
                                    hVar10 = hVar5;
                                    i7 = 8;
                                    i8 = 0;
                                    i9 = 1;
                                } else {
                                    i7 = 8;
                                    i8 = 0;
                                    i9 = 1;
                                    dVar4 = this;
                                    dVar5 = dVar;
                                    hVar8 = hVar3;
                                    hVar9 = hVar4;
                                    hVar10 = hVar5;
                                }
                                i10 = i9;
                                if ((dVar4.k != 2 ? i8 : i10) != 0) {
                                }
                                hVar11 = hVar9;
                                hVar12 = hVar10;
                                if (z6) {
                                }
                                if (dVar6.K.g()) {
                                }
                                this.f1882h = false;
                                this.i = false;
                            }
                            hVar3 = l5;
                            hVar4 = hVar;
                            hVar5 = hVar2;
                            hVar6 = l2;
                            hVar7 = l;
                            if (z) {
                            }
                            i10 = i9;
                            if ((dVar4.k != 2 ? i8 : i10) != 0) {
                            }
                            hVar11 = hVar9;
                            hVar12 = hVar10;
                            if (z6) {
                            }
                            if (dVar6.K.g()) {
                            }
                            this.f1882h = false;
                            this.i = false;
                        }
                    }
                    z7 = false;
                    if (z6) {
                    }
                    if (this.O[0] != 2) {
                    }
                    if (z8) {
                    }
                    z9 = !this.K.g();
                    boolean[] zArr22 = this.N;
                    z10 = zArr22[0];
                    boolean z182 = zArr22[1];
                    if (this.j != 2) {
                        if (z) {
                            fVar = kVar.f1935h;
                            if (fVar.j) {
                                if (z) {
                                }
                            }
                        }
                        d dVar82 = this.P;
                        if (dVar82 == null) {
                        }
                        d dVar92 = this.P;
                        if (dVar92 == null) {
                        }
                        boolean z192 = this.f1880f[0];
                        int[] iArr42 = this.O;
                        hVar3 = l5;
                        hVar4 = hVar;
                        hVar5 = hVar2;
                        hVar6 = l2;
                        hVar7 = l;
                        f(dVar, true, z3, z2, z192, l7, l6, iArr42[0], z8, this.D, this.F, this.U, i6, this.X, this.w[0], this.Z, z7, iArr42[1] != 3, z5, z4, z10, i, i2, this.o, this.p, this.q, z9);
                        if (z) {
                        }
                        i10 = i9;
                        if ((dVar4.k != 2 ? i8 : i10) != 0) {
                        }
                        hVar11 = hVar9;
                        hVar12 = hVar10;
                        if (z6) {
                        }
                        if (dVar6.K.g()) {
                        }
                        this.f1882h = false;
                        this.i = false;
                    }
                    hVar3 = l5;
                    hVar4 = hVar;
                    hVar5 = hVar2;
                    hVar6 = l2;
                    hVar7 = l;
                    if (z) {
                    }
                    i10 = i9;
                    if ((dVar4.k != 2 ? i8 : i10) != 0) {
                    }
                    hVar11 = hVar9;
                    hVar12 = hVar10;
                    if (z6) {
                    }
                    if (dVar6.K.g()) {
                    }
                    this.f1882h = false;
                    this.i = false;
                } else {
                    i12 = 0;
                }
                if (iArr2[i12] == 3 && i21 == 3) {
                    this.u = i12;
                    int i27 = (int) (f2 * i17);
                    if (iArr2[1] == 3) {
                        i = i21;
                        z6 = true;
                        i2 = i23;
                        i4 = i22;
                        i3 = i27;
                        int[] iArr32 = this.n;
                        iArr32[0] = i;
                        iArr32[1] = i2;
                        if (z6) {
                        }
                        z7 = false;
                        if (z6) {
                        }
                        if (this.O[0] != 2) {
                        }
                        if (z8) {
                        }
                        z9 = !this.K.g();
                        boolean[] zArr222 = this.N;
                        z10 = zArr222[0];
                        boolean z1822 = zArr222[1];
                        if (this.j != 2) {
                        }
                        hVar3 = l5;
                        hVar4 = hVar;
                        hVar5 = hVar2;
                        hVar6 = l2;
                        hVar7 = l;
                        if (z) {
                        }
                        i10 = i9;
                        if ((dVar4.k != 2 ? i8 : i10) != 0) {
                        }
                        hVar11 = hVar9;
                        hVar12 = hVar10;
                        if (z6) {
                        }
                        if (dVar6.K.g()) {
                        }
                        this.f1882h = false;
                        this.i = false;
                    }
                    i3 = i27;
                    i2 = i23;
                    i4 = i22;
                    i = 4;
                } else {
                    if (iArr2[1] == 3 && i23 == 3) {
                        this.u = 1;
                        if (i19 == -1) {
                            this.v = 1.0f / f2;
                        }
                        i13 = (int) (this.v * i15);
                        if (iArr2[0] != 3) {
                            i4 = i13;
                            i = i21;
                            i3 = i20;
                            i2 = 4;
                        }
                        i4 = i13;
                        i = i21;
                        i2 = i23;
                        i3 = i20;
                        z6 = true;
                        int[] iArr322 = this.n;
                        iArr322[0] = i;
                        iArr322[1] = i2;
                        if (z6) {
                        }
                        z7 = false;
                        if (z6) {
                        }
                        if (this.O[0] != 2) {
                        }
                        if (z8) {
                        }
                        z9 = !this.K.g();
                        boolean[] zArr2222 = this.N;
                        z10 = zArr2222[0];
                        boolean z18222 = zArr2222[1];
                        if (this.j != 2) {
                        }
                        hVar3 = l5;
                        hVar4 = hVar;
                        hVar5 = hVar2;
                        hVar6 = l2;
                        hVar7 = l;
                        if (z) {
                        }
                        i10 = i9;
                        if ((dVar4.k != 2 ? i8 : i10) != 0) {
                        }
                        hVar11 = hVar9;
                        hVar12 = hVar10;
                        if (z6) {
                        }
                        if (dVar6.K.g()) {
                        }
                        this.f1882h = false;
                        this.i = false;
                    }
                    i13 = i22;
                    i4 = i13;
                    i = i21;
                    i2 = i23;
                    i3 = i20;
                    z6 = true;
                    int[] iArr3222 = this.n;
                    iArr3222[0] = i;
                    iArr3222[1] = i2;
                    if (z6) {
                    }
                    z7 = false;
                    if (z6) {
                    }
                    if (this.O[0] != 2) {
                    }
                    if (z8) {
                    }
                    z9 = !this.K.g();
                    boolean[] zArr22222 = this.N;
                    z10 = zArr22222[0];
                    boolean z182222 = zArr22222[1];
                    if (this.j != 2) {
                    }
                    hVar3 = l5;
                    hVar4 = hVar;
                    hVar5 = hVar2;
                    hVar6 = l2;
                    hVar7 = l;
                    if (z) {
                    }
                    i10 = i9;
                    if ((dVar4.k != 2 ? i8 : i10) != 0) {
                    }
                    hVar11 = hVar9;
                    hVar12 = hVar10;
                    if (z6) {
                    }
                    if (dVar6.K.g()) {
                    }
                    this.f1882h = false;
                    this.i = false;
                }
                z6 = false;
                int[] iArr32222 = this.n;
                iArr32222[0] = i;
                iArr32222[1] = i2;
                if (z6) {
                }
                z7 = false;
                if (z6) {
                }
                if (this.O[0] != 2) {
                }
                if (z8) {
                }
                z9 = !this.K.g();
                boolean[] zArr222222 = this.N;
                z10 = zArr222222[0];
                boolean z1822222 = zArr222222[1];
                if (this.j != 2) {
                }
                hVar3 = l5;
                hVar4 = hVar;
                hVar5 = hVar2;
                hVar6 = l2;
                hVar7 = l;
                if (z) {
                }
                i10 = i9;
                if ((dVar4.k != 2 ? i8 : i10) != 0) {
                }
                hVar11 = hVar9;
                hVar12 = hVar10;
                if (z6) {
                }
                if (dVar6.K.g()) {
                }
                this.f1882h = false;
                this.i = false;
            }
        } else {
            hVar = l4;
        }
        hVar2 = l3;
        i = i21;
        i2 = i23;
        i3 = i20;
        i4 = i22;
        z6 = false;
        int[] iArr322222 = this.n;
        iArr322222[0] = i;
        iArr322222[1] = i2;
        if (z6) {
        }
        z7 = false;
        if (z6) {
        }
        if (this.O[0] != 2) {
        }
        if (z8) {
        }
        z9 = !this.K.g();
        boolean[] zArr2222222 = this.N;
        z10 = zArr2222222[0];
        boolean z18222222 = zArr2222222[1];
        if (this.j != 2) {
        }
        hVar3 = l5;
        hVar4 = hVar;
        hVar5 = hVar2;
        hVar6 = l2;
        hVar7 = l;
        if (z) {
        }
        i10 = i9;
        if ((dVar4.k != 2 ? i8 : i10) != 0) {
        }
        hVar11 = hVar9;
        hVar12 = hVar10;
        if (z6) {
        }
        if (dVar6.K.g()) {
        }
        this.f1882h = false;
        this.i = false;
    }

    public boolean e() {
        return this.c0 != 8;
    }

    /* JADX WARN: Removed duplicated region for block: B:100:0x0181  */
    /* JADX WARN: Removed duplicated region for block: B:164:0x026f  */
    /* JADX WARN: Removed duplicated region for block: B:165:0x0273  */
    /* JADX WARN: Removed duplicated region for block: B:202:0x031b A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:207:0x0328  */
    /* JADX WARN: Removed duplicated region for block: B:215:0x0370  */
    /* JADX WARN: Removed duplicated region for block: B:218:0x0389  */
    /* JADX WARN: Removed duplicated region for block: B:222:0x0392  */
    /* JADX WARN: Removed duplicated region for block: B:233:0x03b9  */
    /* JADX WARN: Removed duplicated region for block: B:235:0x03bd A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:243:0x03d3  */
    /* JADX WARN: Removed duplicated region for block: B:25:0x006b  */
    /* JADX WARN: Removed duplicated region for block: B:26:0x006f  */
    /* JADX WARN: Removed duplicated region for block: B:276:0x0426  */
    /* JADX WARN: Removed duplicated region for block: B:282:0x0438  */
    /* JADX WARN: Removed duplicated region for block: B:284:0x043b A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:28:0x0073  */
    /* JADX WARN: Removed duplicated region for block: B:295:0x045b A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:303:0x0474  */
    /* JADX WARN: Removed duplicated region for block: B:305:0x0484 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:332:? A[ADDED_TO_REGION, RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:335:? A[ADDED_TO_REGION, RETURN, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:36:0x0094  */
    /* JADX WARN: Removed duplicated region for block: B:38:0x0099  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x00b5  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void f(b.h.b.d dVar, boolean z, boolean z2, boolean z3, boolean z4, b.h.b.h hVar, b.h.b.h hVar2, int i, boolean z5, c cVar, c cVar2, int i2, int i3, int i4, int i5, float f2, boolean z6, boolean z7, boolean z8, boolean z9, boolean z10, int i6, int i7, int i8, int i9, float f3, boolean z11) {
        boolean z12;
        boolean z13;
        int i10;
        boolean z14;
        int i11;
        int i12;
        int i13;
        boolean z15;
        boolean z16;
        b.h.b.h l;
        b.h.b.h l2;
        int i14;
        b.h.b.h hVar3;
        int i15;
        b.h.b.h hVar4;
        b.h.b.h hVar5;
        int i16;
        boolean z17;
        int i17;
        int i18;
        b.h.b.h hVar6;
        int i19;
        int i20;
        c cVar3;
        b.h.b.h hVar7;
        int i21;
        int i22;
        boolean z18;
        b.h.b.h hVar8;
        boolean z19;
        int i23;
        b.h.b.h hVar9;
        int i24;
        boolean z20;
        int i25;
        int i26;
        int i27;
        boolean z21;
        boolean z22;
        boolean z23;
        d dVar2;
        d dVar3;
        b.h.b.h hVar10;
        b.h.b.h hVar11;
        b.h.b.h hVar12;
        boolean z24;
        boolean z25;
        int i28;
        d dVar4;
        b.h.b.h hVar13;
        boolean z26;
        boolean z27;
        b.h.b.h hVar14;
        int i29;
        int i30;
        b.h.b.h hVar15;
        int i31;
        d dVar5;
        int i32;
        int i33;
        int i34;
        boolean z28;
        boolean z29;
        int i35;
        int i36;
        int i37;
        boolean z30;
        boolean z31;
        boolean z32;
        boolean z33;
        boolean z34;
        boolean z35;
        int i38;
        boolean z36;
        int i39;
        b.h.b.h hVar16;
        int i40;
        int i41;
        int i42 = i8;
        int i43 = i9;
        c.a aVar = c.a.BOTTOM;
        b.h.b.h l3 = dVar.l(cVar);
        b.h.b.h l4 = dVar.l(cVar2);
        b.h.b.h l5 = dVar.l(cVar.f1865f);
        b.h.b.h l6 = dVar.l(cVar2.f1865f);
        boolean g2 = cVar.g();
        boolean g3 = cVar2.g();
        boolean g4 = this.K.g();
        int i44 = g3 ? (g2 ? 1 : 0) + 1 : g2 ? 1 : 0;
        if (g4) {
            i44++;
        }
        int i45 = i44;
        int i46 = z6 ? 3 : i6;
        int f4 = m0.f(i);
        if (f4 != 0 && f4 != 1 && f4 == 2) {
            if (i46 != 4) {
                z12 = true;
                z13 = z12;
                if (this.c0 != 8) {
                    i10 = 0;
                    z13 = false;
                } else {
                    i10 = i3;
                }
                if (z11) {
                    z14 = g4;
                    i11 = 8;
                } else {
                    if (!g2 && !g3 && !g4) {
                        dVar.e(l3, i2);
                    } else if (g2 && !g3) {
                        z14 = g4;
                        i11 = 8;
                        dVar.d(l3, l5, cVar.d(), 8);
                    }
                    z14 = g4;
                    i11 = 8;
                }
                if (!z13) {
                    i12 = 3;
                    if (i45 == 2 || z6 || !(i46 == 1 || i46 == 0)) {
                        if (i42 == -2) {
                            i42 = i10;
                        }
                        if (i43 == -2) {
                            i43 = i10;
                        }
                        if (i10 > 0 && i46 != 1) {
                            i10 = 0;
                        }
                        if (i42 > 0) {
                            dVar.f(l4, l3, i42, 8);
                            i10 = Math.max(i10, i42);
                        }
                        if (i43 > 0) {
                            if ((z2 && i46 == 1) ? false : true) {
                                i13 = 8;
                                dVar.g(l4, l3, i43, 8);
                            } else {
                                i13 = 8;
                            }
                            i10 = Math.min(i10, i43);
                        } else {
                            i13 = 8;
                        }
                        if (i46 == 1) {
                            if (z2) {
                                dVar.d(l4, l3, i10, i13);
                            } else if (z8) {
                                dVar.d(l4, l3, i10, 5);
                                dVar.g(l4, l3, i10, i13);
                            } else {
                                dVar.d(l4, l3, i10, 5);
                                dVar.g(l4, l3, i10, i13);
                            }
                        } else if (i46 != 2) {
                            z15 = z13;
                            z16 = true;
                            i14 = i42;
                            if (z11) {
                                hVar3 = hVar;
                                i15 = i45;
                                hVar4 = l4;
                                hVar5 = l3;
                                i16 = i12;
                                z17 = z16;
                                i17 = 0;
                                i18 = 1;
                                hVar6 = hVar2;
                                i19 = 2;
                            } else if (z8) {
                                hVar3 = hVar;
                                i15 = i45;
                                hVar4 = l4;
                                hVar5 = l3;
                                i16 = i12;
                                z17 = z16;
                                i17 = 0;
                                i19 = 2;
                                i18 = 1;
                                hVar6 = hVar2;
                            } else {
                                if ((g2 || g3 || z14) && (!g2 || g3)) {
                                    if (g2 || !g3) {
                                        hVar7 = l6;
                                        i21 = 0;
                                        if (g2 && g3) {
                                            d dVar6 = cVar.f1865f.f1863d;
                                            d dVar7 = cVar2.f1865f.f1863d;
                                            d dVar8 = this.P;
                                            int i47 = 6;
                                            if (!z15) {
                                                i22 = i46;
                                                z18 = true;
                                                if (l5.f1850g && hVar7.f1850g) {
                                                    dVar.b(l3, l5, cVar.d(), f2, hVar7, l4, cVar2.d(), 8);
                                                    if (z2 && z16) {
                                                        if (cVar2.f1865f != null) {
                                                            i24 = cVar2.d();
                                                            hVar9 = hVar2;
                                                        } else {
                                                            hVar9 = hVar2;
                                                            i24 = 0;
                                                        }
                                                        if (hVar7 != hVar9) {
                                                            dVar.f(hVar9, l4, i24, 5);
                                                            return;
                                                        }
                                                        return;
                                                    }
                                                    return;
                                                }
                                                hVar8 = hVar2;
                                                z19 = false;
                                                i23 = 5;
                                            } else {
                                                if (i46 == 0) {
                                                    if (i43 != 0 || i14 != 0) {
                                                        z36 = true;
                                                        z34 = true;
                                                        z35 = false;
                                                        i38 = 5;
                                                        i39 = 5;
                                                    } else if (l5.f1850g && hVar7.f1850g) {
                                                        dVar.d(l3, l5, cVar.d(), 8);
                                                        dVar.d(l4, hVar7, -cVar2.d(), 8);
                                                        return;
                                                    } else {
                                                        z36 = false;
                                                        z34 = false;
                                                        z35 = true;
                                                        i38 = 8;
                                                        i39 = 8;
                                                    }
                                                    if (!(dVar6 instanceof a) && !(dVar7 instanceof a)) {
                                                        z30 = z36;
                                                        i35 = i39;
                                                        i37 = i38;
                                                        z32 = z35;
                                                        z31 = z34;
                                                        i22 = i46;
                                                        z18 = true;
                                                        z20 = z31;
                                                        z21 = z32;
                                                        i26 = i37;
                                                        i27 = i35;
                                                        z22 = z30;
                                                        i25 = 6;
                                                        hVar8 = hVar2;
                                                        if (z20 || l5 != hVar7 || dVar6 == dVar8) {
                                                            z23 = z18;
                                                        } else {
                                                            z23 = false;
                                                            z20 = false;
                                                        }
                                                        if (z22) {
                                                            dVar2 = dVar8;
                                                            dVar3 = dVar6;
                                                            hVar10 = hVar7;
                                                            hVar11 = l5;
                                                            hVar12 = l4;
                                                            z24 = z16;
                                                            z25 = z18;
                                                            i28 = 4;
                                                            dVar4 = dVar7;
                                                            hVar13 = l3;
                                                            z26 = z2;
                                                            z27 = z23;
                                                        } else {
                                                            if (z15 || z7 || z9 || l5 != hVar || hVar7 != hVar8) {
                                                                i33 = i26;
                                                                i34 = i25;
                                                                z26 = z2;
                                                            } else {
                                                                z26 = false;
                                                                i34 = 8;
                                                                z23 = false;
                                                                i33 = 8;
                                                            }
                                                            dVar3 = dVar6;
                                                            z24 = z16;
                                                            dVar4 = dVar7;
                                                            hVar10 = hVar7;
                                                            i28 = 4;
                                                            z25 = true;
                                                            hVar11 = l5;
                                                            hVar12 = l4;
                                                            dVar2 = dVar8;
                                                            hVar13 = l3;
                                                            dVar.b(l3, l5, cVar.d(), f2, hVar7, l4, cVar2.d(), i34);
                                                            z27 = z23;
                                                            i26 = i33;
                                                        }
                                                        if (this.c0 == 8 || cVar2.f()) {
                                                            if (z20) {
                                                                hVar14 = hVar10;
                                                            } else {
                                                                hVar14 = hVar10;
                                                                int i48 = (!z26 || hVar11 == hVar14 || z15 || !((dVar3 instanceof a) || (dVar4 instanceof a))) ? i26 : 6;
                                                                dVar.f(hVar13, hVar11, cVar.d(), i48);
                                                                dVar.g(hVar12, hVar14, -cVar2.d(), i48);
                                                                i26 = i48;
                                                            }
                                                            if (z26 || !z10 || (dVar3 instanceof a) || (dVar4 instanceof a)) {
                                                                i29 = i27;
                                                                i30 = i26;
                                                            } else {
                                                                i29 = 6;
                                                                i30 = 6;
                                                                z27 = z25;
                                                            }
                                                            if (z27) {
                                                                if (!z21 || (z9 && !z3)) {
                                                                    dVar5 = dVar2;
                                                                } else {
                                                                    dVar5 = dVar2;
                                                                    if (dVar3 != dVar5 && dVar4 != dVar5) {
                                                                        i47 = i29;
                                                                    }
                                                                    i47 = ((dVar3 instanceof f) || (dVar4 instanceof f)) ? 5 : 5;
                                                                    i29 = Math.max(z9 ? 5 : ((dVar3 instanceof a) || (dVar4 instanceof a)) ? 5 : 5, i29);
                                                                }
                                                                if (z26) {
                                                                    i29 = Math.min(i30, i29);
                                                                    if (z6 && !z9 && (dVar3 == dVar5 || dVar4 == dVar5)) {
                                                                        i32 = i28;
                                                                        dVar.d(hVar13, hVar11, cVar.d(), i32);
                                                                        dVar.d(hVar12, hVar14, -cVar2.d(), i32);
                                                                    }
                                                                }
                                                                i32 = i29;
                                                                dVar.d(hVar13, hVar11, cVar.d(), i32);
                                                                dVar.d(hVar12, hVar14, -cVar2.d(), i32);
                                                            }
                                                            if (z26) {
                                                                hVar15 = hVar12;
                                                            } else {
                                                                hVar15 = hVar12;
                                                                int d2 = hVar == hVar11 ? cVar.d() : 0;
                                                                if (hVar11 != hVar) {
                                                                    dVar.f(hVar13, hVar, d2, 5);
                                                                }
                                                            }
                                                            if (z26 || !z15 || i4 != 0 || i14 != 0) {
                                                                i31 = 0;
                                                            } else if (z15 && i22 == 3) {
                                                                i31 = 0;
                                                                dVar.f(hVar15, hVar13, 0, 8);
                                                            } else {
                                                                i31 = 0;
                                                                dVar.f(hVar15, hVar13, 0, 5);
                                                            }
                                                            if (z26 || !z24) {
                                                                return;
                                                            }
                                                            if (cVar2.f1865f != null) {
                                                                i40 = cVar2.d();
                                                                hVar16 = hVar2;
                                                            } else {
                                                                hVar16 = hVar2;
                                                                i40 = i31;
                                                            }
                                                            if (hVar14 != hVar16) {
                                                                dVar.f(hVar16, hVar15, i40, 5);
                                                                return;
                                                            }
                                                            return;
                                                        }
                                                        return;
                                                    }
                                                    z33 = z36;
                                                } else if (i46 == 1) {
                                                    z33 = true;
                                                    z34 = true;
                                                    z35 = false;
                                                    i38 = 8;
                                                } else {
                                                    if (i46 == 3) {
                                                        i22 = i46;
                                                        if (this.u == -1) {
                                                            hVar8 = hVar2;
                                                            i25 = z9 ? z2 ? 5 : 4 : 8;
                                                            z22 = true;
                                                            z18 = true;
                                                            z21 = true;
                                                            i27 = 5;
                                                            i26 = 8;
                                                            z20 = true;
                                                            if (z20) {
                                                            }
                                                            z23 = z18;
                                                            if (z22) {
                                                            }
                                                            if (this.c0 == 8) {
                                                            }
                                                            if (z20) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i29 = i27;
                                                            i30 = i26;
                                                            if (z27) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i31 = 0;
                                                            if (z26) {
                                                                return;
                                                            }
                                                            return;
                                                        } else if (z6) {
                                                            if (i7 != 2) {
                                                                z18 = true;
                                                                if (i7 != 1) {
                                                                    z29 = false;
                                                                    if (z29) {
                                                                        i35 = 5;
                                                                        i36 = 8;
                                                                    } else {
                                                                        i35 = 4;
                                                                        i36 = 5;
                                                                    }
                                                                    i37 = i36;
                                                                    z30 = z18;
                                                                    z31 = z30;
                                                                    z32 = z31;
                                                                    z20 = z31;
                                                                    z21 = z32;
                                                                    i26 = i37;
                                                                    i27 = i35;
                                                                    z22 = z30;
                                                                    i25 = 6;
                                                                    hVar8 = hVar2;
                                                                    if (z20) {
                                                                    }
                                                                    z23 = z18;
                                                                    if (z22) {
                                                                    }
                                                                    if (this.c0 == 8) {
                                                                    }
                                                                    if (z20) {
                                                                    }
                                                                    if (z26) {
                                                                    }
                                                                    i29 = i27;
                                                                    i30 = i26;
                                                                    if (z27) {
                                                                    }
                                                                    if (z26) {
                                                                    }
                                                                    if (z26) {
                                                                    }
                                                                    i31 = 0;
                                                                    if (z26) {
                                                                    }
                                                                }
                                                            } else {
                                                                z18 = true;
                                                            }
                                                            z29 = z18;
                                                            if (z29) {
                                                            }
                                                            i37 = i36;
                                                            z30 = z18;
                                                            z31 = z30;
                                                            z32 = z31;
                                                            z20 = z31;
                                                            z21 = z32;
                                                            i26 = i37;
                                                            i27 = i35;
                                                            z22 = z30;
                                                            i25 = 6;
                                                            hVar8 = hVar2;
                                                            if (z20) {
                                                            }
                                                            z23 = z18;
                                                            if (z22) {
                                                            }
                                                            if (this.c0 == 8) {
                                                            }
                                                            if (z20) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i29 = i27;
                                                            i30 = i26;
                                                            if (z27) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i31 = 0;
                                                            if (z26) {
                                                            }
                                                        } else {
                                                            z18 = true;
                                                            if (i43 > 0) {
                                                                z22 = true;
                                                                z28 = true;
                                                                z21 = true;
                                                                i27 = 5;
                                                            } else if (i43 != 0 || i14 != 0) {
                                                                z22 = true;
                                                                z28 = true;
                                                                z21 = true;
                                                            } else if (z9) {
                                                                hVar8 = hVar2;
                                                                i23 = (dVar6 == dVar8 || dVar7 == dVar8) ? 5 : 4;
                                                                z19 = true;
                                                            } else {
                                                                z22 = true;
                                                                z28 = true;
                                                                z21 = true;
                                                                i27 = 8;
                                                            }
                                                            z20 = z28;
                                                            i25 = 6;
                                                            i26 = 5;
                                                            hVar8 = hVar2;
                                                            if (z20) {
                                                            }
                                                            z23 = z18;
                                                            if (z22) {
                                                            }
                                                            if (this.c0 == 8) {
                                                            }
                                                            if (z20) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i29 = i27;
                                                            i30 = i26;
                                                            if (z27) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            if (z26) {
                                                            }
                                                            i31 = 0;
                                                            if (z26) {
                                                            }
                                                        }
                                                    } else {
                                                        i22 = i46;
                                                        z18 = true;
                                                        z22 = false;
                                                        z28 = false;
                                                        z21 = false;
                                                    }
                                                    i27 = 4;
                                                    z20 = z28;
                                                    i25 = 6;
                                                    i26 = 5;
                                                    hVar8 = hVar2;
                                                    if (z20) {
                                                    }
                                                    z23 = z18;
                                                    if (z22) {
                                                    }
                                                    if (this.c0 == 8) {
                                                    }
                                                    if (z20) {
                                                    }
                                                    if (z26) {
                                                    }
                                                    i29 = i27;
                                                    i30 = i26;
                                                    if (z27) {
                                                    }
                                                    if (z26) {
                                                    }
                                                    if (z26) {
                                                    }
                                                    i31 = 0;
                                                    if (z26) {
                                                    }
                                                }
                                                z30 = z33;
                                                i37 = i38;
                                                i35 = 4;
                                                z32 = z35;
                                                z31 = z34;
                                                i22 = i46;
                                                z18 = true;
                                                z20 = z31;
                                                z21 = z32;
                                                i26 = i37;
                                                i27 = i35;
                                                z22 = z30;
                                                i25 = 6;
                                                hVar8 = hVar2;
                                                if (z20) {
                                                }
                                                z23 = z18;
                                                if (z22) {
                                                }
                                                if (this.c0 == 8) {
                                                }
                                                if (z20) {
                                                }
                                                if (z26) {
                                                }
                                                i29 = i27;
                                                i30 = i26;
                                                if (z27) {
                                                }
                                                if (z26) {
                                                }
                                                if (z26) {
                                                }
                                                i31 = 0;
                                                if (z26) {
                                                }
                                            }
                                            z20 = z18;
                                            i25 = 6;
                                            i26 = i23;
                                            i27 = 4;
                                            z21 = z19;
                                            z22 = z20;
                                            if (z20) {
                                            }
                                            z23 = z18;
                                            if (z22) {
                                            }
                                            if (this.c0 == 8) {
                                            }
                                            if (z20) {
                                            }
                                            if (z26) {
                                            }
                                            i29 = i27;
                                            i30 = i26;
                                            if (z27) {
                                            }
                                            if (z26) {
                                            }
                                            if (z26) {
                                            }
                                            i31 = 0;
                                            if (z26) {
                                            }
                                        }
                                    } else {
                                        hVar7 = l6;
                                        dVar.d(l4, hVar7, -cVar2.d(), 8);
                                        if (z2) {
                                            i21 = 0;
                                            dVar.f(l3, hVar, 0, 5);
                                        }
                                    }
                                    i31 = i21;
                                    hVar14 = hVar7;
                                    hVar15 = l4;
                                    z24 = z16;
                                    z26 = z2;
                                    if (z26) {
                                    }
                                } else {
                                    hVar7 = l6;
                                }
                                i21 = 0;
                                i31 = i21;
                                hVar14 = hVar7;
                                hVar15 = l4;
                                z24 = z16;
                                z26 = z2;
                                if (z26) {
                                }
                            }
                            if (i15 >= i19 && z2 && z17) {
                                dVar.f(hVar5, hVar3, i17, 8);
                                int i49 = (z || this.H.f1865f == null) ? i18 : i17;
                                if (z || (cVar3 = this.H.f1865f) == null) {
                                    i20 = i49;
                                } else {
                                    d dVar9 = cVar3.f1863d;
                                    if (dVar9.S != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                        int[] iArr = dVar9.O;
                                        if (iArr[i17] == i16 && iArr[i18] == i16) {
                                            i20 = i18;
                                        }
                                    }
                                    i20 = i17;
                                }
                                if (i20 != 0) {
                                    dVar.f(hVar6, hVar4, i17, 8);
                                    return;
                                }
                                return;
                            }
                            return;
                        } else {
                            c.a aVar2 = cVar.f1864e;
                            c.a aVar3 = c.a.TOP;
                            if (aVar2 != aVar3 && aVar2 != aVar) {
                                l = dVar.l(this.P.i(c.a.LEFT));
                                l2 = dVar.l(this.P.i(c.a.RIGHT));
                            } else {
                                l = dVar.l(this.P.i(aVar3));
                                l2 = dVar.l(this.P.i(aVar));
                            }
                            b.h.b.h hVar17 = l;
                            b.h.b.b m = dVar.m();
                            m.d(l4, l3, l2, hVar17, f3);
                            dVar.c(m);
                        }
                    } else {
                        int max = Math.max(i42, i10);
                        if (i43 > 0) {
                            max = Math.min(i43, max);
                        }
                        dVar.d(l4, l3, max, 8);
                    }
                    z16 = z4;
                    i14 = i42;
                    z15 = false;
                    if (z11) {
                    }
                    if (i15 >= i19) {
                        return;
                    }
                    return;
                }
                if (z5) {
                    i41 = 3;
                    dVar.d(l4, l3, 0, 3);
                    if (i4 > 0) {
                        dVar.f(l4, l3, i4, i11);
                    }
                    if (i5 < Integer.MAX_VALUE) {
                        dVar.g(l4, l3, i5, i11);
                    }
                } else {
                    i41 = 3;
                    dVar.d(l4, l3, i10, i11);
                }
                i12 = i41;
                z16 = z4;
                z15 = z13;
                i14 = i42;
                if (z11) {
                }
                if (i15 >= i19) {
                }
            }
        }
        z12 = false;
        z13 = z12;
        if (this.c0 != 8) {
        }
        if (z11) {
        }
        if (!z13) {
        }
        z16 = z4;
        z15 = z13;
        i14 = i42;
        if (z11) {
        }
        if (i15 >= i19) {
        }
    }

    public void g(b.h.b.d dVar) {
        dVar.l(this.D);
        dVar.l(this.E);
        dVar.l(this.F);
        dVar.l(this.G);
        if (this.W > 0) {
            dVar.l(this.H);
        }
    }

    public void h() {
        if (this.f1878d == null) {
            this.f1878d = new b.h.b.i.l.k(this);
        }
        if (this.f1879e == null) {
            this.f1879e = new m(this);
        }
    }

    public c i(c.a aVar) {
        switch (aVar.ordinal()) {
            case 0:
                return null;
            case 1:
                return this.D;
            case 2:
                return this.E;
            case 3:
                return this.F;
            case 4:
                return this.G;
            case 5:
                return this.H;
            case 6:
                return this.K;
            case 7:
                return this.I;
            case 8:
                return this.J;
            default:
                throw new AssertionError(aVar.name());
        }
    }

    public int j() {
        return t() + this.R;
    }

    public int k(int i) {
        if (i == 0) {
            return m();
        }
        if (i == 1) {
            return q();
        }
        return 0;
    }

    public int l() {
        if (this.c0 == 8) {
            return 0;
        }
        return this.R;
    }

    public int m() {
        return this.O[0];
    }

    public d n(int i) {
        c cVar;
        c cVar2;
        if (i != 0) {
            if (i == 1 && (cVar2 = (cVar = this.G).f1865f) != null && cVar2.f1865f == cVar) {
                return cVar2.f1863d;
            }
            return null;
        }
        c cVar3 = this.F;
        c cVar4 = cVar3.f1865f;
        if (cVar4 == null || cVar4.f1865f != cVar3) {
            return null;
        }
        return cVar4.f1863d;
    }

    public d o(int i) {
        c cVar;
        c cVar2;
        if (i != 0) {
            if (i == 1 && (cVar2 = (cVar = this.E).f1865f) != null && cVar2.f1865f == cVar) {
                return cVar2.f1863d;
            }
            return null;
        }
        c cVar3 = this.D;
        c cVar4 = cVar3.f1865f;
        if (cVar4 == null || cVar4.f1865f != cVar3) {
            return null;
        }
        return cVar4.f1863d;
    }

    public int p() {
        return s() + this.Q;
    }

    public int q() {
        return this.O[1];
    }

    public int r() {
        if (this.c0 == 8) {
            return 0;
        }
        return this.Q;
    }

    public int s() {
        d dVar = this.P;
        if (dVar != null && (dVar instanceof e)) {
            return ((e) dVar).r0 + this.U;
        }
        return this.U;
    }

    public int t() {
        d dVar = this.P;
        if (dVar != null && (dVar instanceof e)) {
            return ((e) dVar).s0 + this.V;
        }
        return this.V;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("");
        x.append(this.d0 != null ? c.b.a.a.a.v(c.b.a.a.a.x("id: "), this.d0, " ") : "");
        x.append("(");
        x.append(this.U);
        x.append(", ");
        x.append(this.V);
        x.append(") - (");
        x.append(this.Q);
        x.append(" x ");
        return c.b.a.a.a.s(x, this.R, ")");
    }

    public boolean u(int i) {
        if (i == 0) {
            return (this.D.f1865f != null ? 1 : 0) + (this.F.f1865f != null ? 1 : 0) < 2;
        }
        return ((this.E.f1865f != null ? 1 : 0) + (this.G.f1865f != null ? 1 : 0)) + (this.H.f1865f != null ? 1 : 0) < 2;
    }

    public final boolean v(int i) {
        int i2 = i * 2;
        c[] cVarArr = this.L;
        if (cVarArr[i2].f1865f != null && cVarArr[i2].f1865f.f1865f != cVarArr[i2]) {
            int i3 = i2 + 1;
            if (cVarArr[i3].f1865f != null && cVarArr[i3].f1865f.f1865f == cVarArr[i3]) {
                return true;
            }
        }
        return false;
    }

    public boolean w() {
        c cVar = this.D;
        c cVar2 = cVar.f1865f;
        if (cVar2 == null || cVar2.f1865f != cVar) {
            c cVar3 = this.F;
            c cVar4 = cVar3.f1865f;
            return cVar4 != null && cVar4.f1865f == cVar3;
        }
        return true;
    }

    public boolean x() {
        c cVar = this.E;
        c cVar2 = cVar.f1865f;
        if (cVar2 == null || cVar2.f1865f != cVar) {
            c cVar3 = this.G;
            c cVar4 = cVar3.f1865f;
            return cVar4 != null && cVar4.f1865f == cVar3;
        }
        return true;
    }

    public boolean y() {
        return this.f1881g && this.c0 != 8;
    }

    public boolean z() {
        return this.f1882h || (this.D.f1862c && this.F.f1862c);
    }
}