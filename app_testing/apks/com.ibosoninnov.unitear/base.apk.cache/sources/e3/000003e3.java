package b.h.b;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.Objects;

/* compiled from: LinearSystem.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public static boolean f1829a = false;

    /* renamed from: b  reason: collision with root package name */
    public static int f1830b = 1000;

    /* renamed from: c  reason: collision with root package name */
    public static long f1831c;

    /* renamed from: f  reason: collision with root package name */
    public a f1834f;
    public b[] i;
    public final c o;
    public a r;

    /* renamed from: d  reason: collision with root package name */
    public boolean f1832d = false;

    /* renamed from: e  reason: collision with root package name */
    public int f1833e = 0;

    /* renamed from: g  reason: collision with root package name */
    public int f1835g = 32;

    /* renamed from: h  reason: collision with root package name */
    public int f1836h = 32;
    public boolean j = false;
    public boolean[] k = new boolean[32];
    public int l = 1;
    public int m = 0;
    public int n = 32;
    public h[] p = new h[f1830b];
    public int q = 0;

    /* compiled from: LinearSystem.java */
    /* loaded from: classes.dex */
    public interface a {
        void a(h hVar);

        h b(d dVar, boolean[] zArr);

        void clear();

        boolean isEmpty();
    }

    public d() {
        this.i = null;
        this.i = new b[32];
        t();
        c cVar = new c();
        this.o = cVar;
        this.f1834f = new g(cVar);
        this.r = new b(cVar);
    }

    public final h a(int i, String str) {
        h a2 = this.o.f1827c.a();
        if (a2 == null) {
            a2 = new h(i);
            a2.j = i;
        } else {
            a2.c();
            a2.j = i;
        }
        int i2 = this.q;
        int i3 = f1830b;
        if (i2 >= i3) {
            int i4 = i3 * 2;
            f1830b = i4;
            this.p = (h[]) Arrays.copyOf(this.p, i4);
        }
        h[] hVarArr = this.p;
        int i5 = this.q;
        this.q = i5 + 1;
        hVarArr[i5] = a2;
        return a2;
    }

    public void b(h hVar, h hVar2, int i, float f2, h hVar3, h hVar4, int i2, int i3) {
        b m = m();
        if (hVar2 == hVar3) {
            m.f1823d.i(hVar, 1.0f);
            m.f1823d.i(hVar4, 1.0f);
            m.f1823d.i(hVar2, -2.0f);
        } else if (f2 == 0.5f) {
            m.f1823d.i(hVar, 1.0f);
            m.f1823d.i(hVar2, -1.0f);
            m.f1823d.i(hVar3, -1.0f);
            m.f1823d.i(hVar4, 1.0f);
            if (i > 0 || i2 > 0) {
                m.f1821b = (-i) + i2;
            }
        } else if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            m.f1823d.i(hVar, -1.0f);
            m.f1823d.i(hVar2, 1.0f);
            m.f1821b = i;
        } else if (f2 >= 1.0f) {
            m.f1823d.i(hVar4, -1.0f);
            m.f1823d.i(hVar3, 1.0f);
            m.f1821b = -i2;
        } else {
            float f3 = 1.0f - f2;
            m.f1823d.i(hVar, f3 * 1.0f);
            m.f1823d.i(hVar2, f3 * (-1.0f));
            m.f1823d.i(hVar3, (-1.0f) * f2);
            m.f1823d.i(hVar4, 1.0f * f2);
            if (i > 0 || i2 > 0) {
                m.f1821b = (i2 * f2) + ((-i) * f3);
            }
        }
        if (i3 != 8) {
            m.c(this, i3);
        }
        c(m);
    }

    /* JADX WARN: Removed duplicated region for block: B:116:0x01af A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:117:0x01b0  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void c(b bVar) {
        boolean z;
        boolean z2;
        boolean z3;
        h hVar;
        h i;
        boolean h2;
        boolean h3;
        boolean z4 = true;
        if (this.m + 1 >= this.n || this.l + 1 >= this.f1836h) {
            p();
        }
        if (bVar.f1824e) {
            z = false;
        } else {
            if (this.i.length != 0) {
                boolean z5 = false;
                while (!z5) {
                    int a2 = bVar.f1823d.a();
                    for (int i2 = 0; i2 < a2; i2++) {
                        h b2 = bVar.f1823d.b(i2);
                        if (b2.f1847d != -1 || b2.f1850g) {
                            bVar.f1822c.add(b2);
                        }
                    }
                    int size = bVar.f1822c.size();
                    if (size > 0) {
                        for (int i3 = 0; i3 < size; i3++) {
                            h hVar2 = bVar.f1822c.get(i3);
                            if (hVar2.f1850g) {
                                bVar.k(this, hVar2, true);
                            } else {
                                bVar.l(this, this.i[hVar2.f1847d], true);
                            }
                        }
                        bVar.f1822c.clear();
                    } else {
                        z5 = true;
                    }
                }
                if (bVar.f1820a != null && bVar.f1823d.a() == 0) {
                    bVar.f1824e = true;
                    this.f1832d = true;
                }
            }
            if (bVar.isEmpty()) {
                return;
            }
            float f2 = bVar.f1821b;
            if (f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                bVar.f1821b = f2 * (-1.0f);
                bVar.f1823d.c();
            }
            int a3 = bVar.f1823d.a();
            float f3 = 0.0f;
            float f4 = 0.0f;
            h hVar3 = null;
            h hVar4 = null;
            boolean z6 = false;
            boolean z7 = false;
            for (int i4 = 0; i4 < a3; i4++) {
                float d2 = bVar.f1823d.d(i4);
                h b3 = bVar.f1823d.b(i4);
                if (b3.j == 1) {
                    if (hVar3 == null) {
                        h3 = bVar.h(b3);
                    } else if (f3 > d2) {
                        h3 = bVar.h(b3);
                    } else if (!z6 && bVar.h(b3)) {
                        z6 = true;
                        hVar3 = b3;
                        f3 = d2;
                    }
                    z6 = h3;
                    hVar3 = b3;
                    f3 = d2;
                } else if (hVar3 == null && d2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    if (hVar4 == null) {
                        h2 = bVar.h(b3);
                    } else if (f4 > d2) {
                        h2 = bVar.h(b3);
                    } else if (!z7 && bVar.h(b3)) {
                        z7 = true;
                        hVar4 = b3;
                        f4 = d2;
                    }
                    z7 = h2;
                    hVar4 = b3;
                    f4 = d2;
                }
            }
            if (hVar3 == null) {
                hVar3 = hVar4;
            }
            if (hVar3 == null) {
                z2 = true;
            } else {
                bVar.j(hVar3);
                z2 = false;
            }
            if (bVar.f1823d.a() == 0) {
                bVar.f1824e = true;
            }
            if (z2) {
                if (this.l + 1 >= this.f1836h) {
                    p();
                }
                h a4 = a(3, null);
                int i5 = this.f1833e + 1;
                this.f1833e = i5;
                this.l++;
                a4.f1846c = i5;
                this.o.f1828d[i5] = a4;
                bVar.f1820a = a4;
                int i6 = this.m;
                i(bVar);
                if (this.m == i6 + 1) {
                    b bVar2 = (b) this.r;
                    Objects.requireNonNull(bVar2);
                    bVar2.f1820a = null;
                    bVar2.f1823d.clear();
                    for (int i7 = 0; i7 < bVar.f1823d.a(); i7++) {
                        bVar2.f1823d.e(bVar.f1823d.b(i7), bVar.f1823d.d(i7), true);
                    }
                    s(this.r);
                    if (a4.f1847d == -1) {
                        if (bVar.f1820a == a4 && (i = bVar.i(null, a4)) != null) {
                            bVar.j(i);
                        }
                        if (!bVar.f1824e) {
                            bVar.f1820a.e(this, bVar);
                        }
                        this.o.f1826b.b(bVar);
                        this.m--;
                    }
                    z3 = true;
                    hVar = bVar.f1820a;
                    if (hVar != null || (hVar.j != 1 && bVar.f1821b < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
                        z4 = false;
                    }
                    if (z4) {
                        return;
                    }
                    z = z3;
                }
            }
            z3 = false;
            hVar = bVar.f1820a;
            if (hVar != null) {
            }
            z4 = false;
            if (z4) {
            }
        }
        if (z) {
            return;
        }
        i(bVar);
    }

    public b d(h hVar, h hVar2, int i, int i2) {
        if (i2 == 8 && hVar2.f1850g && hVar.f1847d == -1) {
            hVar.d(this, hVar2.f1849f + i);
            return null;
        }
        b m = m();
        boolean z = false;
        if (i != 0) {
            if (i < 0) {
                i *= -1;
                z = true;
            }
            m.f1821b = i;
        }
        if (!z) {
            m.f1823d.i(hVar, -1.0f);
            m.f1823d.i(hVar2, 1.0f);
        } else {
            m.f1823d.i(hVar, 1.0f);
            m.f1823d.i(hVar2, -1.0f);
        }
        if (i2 != 8) {
            m.c(this, i2);
        }
        c(m);
        return m;
    }

    public void e(h hVar, int i) {
        int i2 = hVar.f1847d;
        if (i2 == -1) {
            hVar.d(this, i);
            for (int i3 = 0; i3 < this.f1833e + 1; i3++) {
                h hVar2 = this.o.f1828d[i3];
            }
        } else if (i2 != -1) {
            b bVar = this.i[i2];
            if (bVar.f1824e) {
                bVar.f1821b = i;
            } else if (bVar.f1823d.a() == 0) {
                bVar.f1824e = true;
                bVar.f1821b = i;
            } else {
                b m = m();
                if (i < 0) {
                    m.f1821b = i * (-1);
                    m.f1823d.i(hVar, 1.0f);
                } else {
                    m.f1821b = i;
                    m.f1823d.i(hVar, -1.0f);
                }
                c(m);
            }
        } else {
            b m2 = m();
            m2.f1820a = hVar;
            float f2 = i;
            hVar.f1849f = f2;
            m2.f1821b = f2;
            m2.f1824e = true;
            c(m2);
        }
    }

    public void f(h hVar, h hVar2, int i, int i2) {
        b m = m();
        h n = n();
        n.f1848e = 0;
        m.e(hVar, hVar2, n, i);
        if (i2 != 8) {
            m.f1823d.i(k(i2, null), (int) (m.f1823d.f(n) * (-1.0f)));
        }
        c(m);
    }

    public void g(h hVar, h hVar2, int i, int i2) {
        b m = m();
        h n = n();
        n.f1848e = 0;
        m.f(hVar, hVar2, n, i);
        if (i2 != 8) {
            m.f1823d.i(k(i2, null), (int) (m.f1823d.f(n) * (-1.0f)));
        }
        c(m);
    }

    public void h(h hVar, h hVar2, h hVar3, h hVar4, float f2, int i) {
        b m = m();
        m.d(hVar, hVar2, hVar3, hVar4, f2);
        if (i != 8) {
            m.c(this, i);
        }
        c(m);
    }

    public final void i(b bVar) {
        int i;
        if (bVar.f1824e) {
            bVar.f1820a.d(this, bVar.f1821b);
        } else {
            b[] bVarArr = this.i;
            int i2 = this.m;
            bVarArr[i2] = bVar;
            h hVar = bVar.f1820a;
            hVar.f1847d = i2;
            this.m = i2 + 1;
            hVar.e(this, bVar);
        }
        if (this.f1832d) {
            int i3 = 0;
            while (i3 < this.m) {
                if (this.i[i3] == null) {
                    System.out.println("WTF");
                }
                b[] bVarArr2 = this.i;
                if (bVarArr2[i3] != null && bVarArr2[i3].f1824e) {
                    b bVar2 = bVarArr2[i3];
                    bVar2.f1820a.d(this, bVar2.f1821b);
                    this.o.f1826b.b(bVar2);
                    this.i[i3] = null;
                    int i4 = i3 + 1;
                    int i5 = i4;
                    while (true) {
                        i = this.m;
                        if (i4 >= i) {
                            break;
                        }
                        b[] bVarArr3 = this.i;
                        int i6 = i4 - 1;
                        bVarArr3[i6] = bVarArr3[i4];
                        if (bVarArr3[i6].f1820a.f1847d == i4) {
                            bVarArr3[i6].f1820a.f1847d = i6;
                        }
                        i5 = i4;
                        i4++;
                    }
                    if (i5 < i) {
                        this.i[i5] = null;
                    }
                    this.m = i - 1;
                    i3--;
                }
                i3++;
            }
            this.f1832d = false;
        }
    }

    public final void j() {
        for (int i = 0; i < this.m; i++) {
            b bVar = this.i[i];
            bVar.f1820a.f1849f = bVar.f1821b;
        }
    }

    public h k(int i, String str) {
        if (this.l + 1 >= this.f1836h) {
            p();
        }
        h a2 = a(4, str);
        int i2 = this.f1833e + 1;
        this.f1833e = i2;
        this.l++;
        a2.f1846c = i2;
        a2.f1848e = i;
        this.o.f1828d[i2] = a2;
        this.f1834f.a(a2);
        return a2;
    }

    public h l(Object obj) {
        h hVar = null;
        if (obj == null) {
            return null;
        }
        if (this.l + 1 >= this.f1836h) {
            p();
        }
        if (obj instanceof b.h.b.i.c) {
            b.h.b.i.c cVar = (b.h.b.i.c) obj;
            hVar = cVar.i;
            if (hVar == null) {
                cVar.i();
                hVar = cVar.i;
            }
            int i = hVar.f1846c;
            if (i == -1 || i > this.f1833e || this.o.f1828d[i] == null) {
                if (i != -1) {
                    hVar.c();
                }
                int i2 = this.f1833e + 1;
                this.f1833e = i2;
                this.l++;
                hVar.f1846c = i2;
                hVar.j = 1;
                this.o.f1828d[i2] = hVar;
            }
        }
        return hVar;
    }

    public b m() {
        b a2 = this.o.f1826b.a();
        if (a2 == null) {
            a2 = new b(this.o);
            f1831c++;
        } else {
            a2.f1820a = null;
            a2.f1823d.clear();
            a2.f1821b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            a2.f1824e = false;
        }
        h.f1844a++;
        return a2;
    }

    public h n() {
        if (this.l + 1 >= this.f1836h) {
            p();
        }
        h a2 = a(3, null);
        int i = this.f1833e + 1;
        this.f1833e = i;
        this.l++;
        a2.f1846c = i;
        this.o.f1828d[i] = a2;
        return a2;
    }

    public int o(Object obj) {
        h hVar = ((b.h.b.i.c) obj).i;
        if (hVar != null) {
            return (int) (hVar.f1849f + 0.5f);
        }
        return 0;
    }

    public final void p() {
        int i = this.f1835g * 2;
        this.f1835g = i;
        this.i = (b[]) Arrays.copyOf(this.i, i);
        c cVar = this.o;
        cVar.f1828d = (h[]) Arrays.copyOf(cVar.f1828d, this.f1835g);
        int i2 = this.f1835g;
        this.k = new boolean[i2];
        this.f1836h = i2;
        this.n = i2;
    }

    public void q() {
        if (this.f1834f.isEmpty()) {
            j();
        } else if (!this.j) {
            r(this.f1834f);
        } else {
            boolean z = false;
            int i = 0;
            while (true) {
                if (i >= this.m) {
                    z = true;
                    break;
                } else if (!this.i[i].f1824e) {
                    break;
                } else {
                    i++;
                }
            }
            if (!z) {
                r(this.f1834f);
            } else {
                j();
            }
        }
    }

    public void r(a aVar) {
        float f2;
        int i;
        boolean z;
        int i2 = 0;
        while (true) {
            int i3 = this.m;
            f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            i = 1;
            if (i2 >= i3) {
                z = false;
                break;
            }
            b[] bVarArr = this.i;
            if (bVarArr[i2].f1820a.j != 1 && bVarArr[i2].f1821b < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                z = true;
                break;
            }
            i2++;
        }
        if (z) {
            boolean z2 = false;
            int i4 = 0;
            while (!z2) {
                i4 += i;
                float f3 = Float.MAX_VALUE;
                int i5 = -1;
                int i6 = -1;
                int i7 = 0;
                int i8 = 0;
                while (i7 < this.m) {
                    b bVar = this.i[i7];
                    if (bVar.f1820a.j != i && !bVar.f1824e && bVar.f1821b < f2) {
                        int a2 = bVar.f1823d.a();
                        int i9 = 0;
                        while (i9 < a2) {
                            h b2 = bVar.f1823d.b(i9);
                            float f4 = bVar.f1823d.f(b2);
                            if (f4 > f2) {
                                for (int i10 = 0; i10 < 9; i10++) {
                                    float f5 = b2.f1851h[i10] / f4;
                                    if ((f5 < f3 && i10 == i8) || i10 > i8) {
                                        i6 = b2.f1846c;
                                        i8 = i10;
                                        f3 = f5;
                                        i5 = i7;
                                    }
                                }
                            }
                            i9++;
                            f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        }
                    }
                    i7++;
                    f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    i = 1;
                }
                if (i5 != -1) {
                    b bVar2 = this.i[i5];
                    bVar2.f1820a.f1847d = -1;
                    bVar2.j(this.o.f1828d[i6]);
                    h hVar = bVar2.f1820a;
                    hVar.f1847d = i5;
                    hVar.e(this, bVar2);
                } else {
                    z2 = true;
                }
                if (i4 > this.l / 2) {
                    z2 = true;
                }
                f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                i = 1;
            }
        }
        s(aVar);
        j();
    }

    public final int s(a aVar) {
        for (int i = 0; i < this.l; i++) {
            this.k[i] = false;
        }
        boolean z = false;
        int i2 = 0;
        while (!z) {
            i2++;
            if (i2 >= this.l * 2) {
                return i2;
            }
            h hVar = ((b) aVar).f1820a;
            if (hVar != null) {
                this.k[hVar.f1846c] = true;
            }
            h b2 = aVar.b(this, this.k);
            if (b2 != null) {
                boolean[] zArr = this.k;
                int i3 = b2.f1846c;
                if (zArr[i3]) {
                    return i2;
                }
                zArr[i3] = true;
            }
            if (b2 != null) {
                float f2 = Float.MAX_VALUE;
                int i4 = -1;
                for (int i5 = 0; i5 < this.m; i5++) {
                    b bVar = this.i[i5];
                    if (bVar.f1820a.j != 1 && !bVar.f1824e && bVar.f1823d.g(b2)) {
                        float f3 = bVar.f1823d.f(b2);
                        if (f3 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            float f4 = (-bVar.f1821b) / f3;
                            if (f4 < f2) {
                                i4 = i5;
                                f2 = f4;
                            }
                        }
                    }
                }
                if (i4 > -1) {
                    b bVar2 = this.i[i4];
                    bVar2.f1820a.f1847d = -1;
                    bVar2.j(b2);
                    h hVar2 = bVar2.f1820a;
                    hVar2.f1847d = i4;
                    hVar2.e(this, bVar2);
                }
            } else {
                z = true;
            }
        }
        return i2;
    }

    public final void t() {
        for (int i = 0; i < this.m; i++) {
            b bVar = this.i[i];
            if (bVar != null) {
                this.o.f1826b.b(bVar);
            }
            this.i[i] = null;
        }
    }

    public void u() {
        c cVar;
        int i = 0;
        while (true) {
            cVar = this.o;
            h[] hVarArr = cVar.f1828d;
            if (i >= hVarArr.length) {
                break;
            }
            h hVar = hVarArr[i];
            if (hVar != null) {
                hVar.c();
            }
            i++;
        }
        f<h> fVar = cVar.f1827c;
        h[] hVarArr2 = this.p;
        int i2 = this.q;
        Objects.requireNonNull(fVar);
        if (i2 > hVarArr2.length) {
            i2 = hVarArr2.length;
        }
        for (int i3 = 0; i3 < i2; i3++) {
            h hVar2 = hVarArr2[i3];
            int i4 = fVar.f1838b;
            Object[] objArr = fVar.f1837a;
            if (i4 < objArr.length) {
                objArr[i4] = hVar2;
                fVar.f1838b = i4 + 1;
            }
        }
        this.q = 0;
        Arrays.fill(this.o.f1828d, (Object) null);
        this.f1833e = 0;
        this.f1834f.clear();
        this.l = 1;
        for (int i5 = 0; i5 < this.m; i5++) {
            b[] bVarArr = this.i;
            if (bVarArr[i5] != null) {
                Objects.requireNonNull(bVarArr[i5]);
            }
        }
        t();
        this.m = 0;
        this.r = new b(this.o);
    }
}