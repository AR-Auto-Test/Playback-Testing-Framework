package b.h.b;

import b.h.b.d;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.google.firebase.crashlytics.internal.common.IdManager;
import java.util.ArrayList;

/* compiled from: ArrayRow.java */
/* loaded from: classes.dex */
public class b implements d.a {

    /* renamed from: d  reason: collision with root package name */
    public a f1823d;

    /* renamed from: a  reason: collision with root package name */
    public h f1820a = null;

    /* renamed from: b  reason: collision with root package name */
    public float f1821b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<h> f1822c = new ArrayList<>();

    /* renamed from: e  reason: collision with root package name */
    public boolean f1824e = false;

    /* compiled from: ArrayRow.java */
    /* loaded from: classes.dex */
    public interface a {
        int a();

        h b(int i);

        void c();

        void clear();

        float d(int i);

        void e(h hVar, float f2, boolean z);

        float f(h hVar);

        boolean g(h hVar);

        float h(b bVar, boolean z);

        void i(h hVar, float f2);

        float j(h hVar, boolean z);

        void k(float f2);
    }

    public b() {
    }

    @Override // b.h.b.d.a
    public void a(h hVar) {
        float f2;
        int i = hVar.f1848e;
        if (i != 1) {
            if (i == 2) {
                f2 = 1000.0f;
            } else if (i == 3) {
                f2 = 1000000.0f;
            } else if (i == 4) {
                f2 = 1.0E9f;
            } else if (i == 5) {
                f2 = 1.0E12f;
            }
            this.f1823d.i(hVar, f2);
        }
        f2 = 1.0f;
        this.f1823d.i(hVar, f2);
    }

    @Override // b.h.b.d.a
    public h b(d dVar, boolean[] zArr) {
        return i(zArr, null);
    }

    public b c(d dVar, int i) {
        this.f1823d.i(dVar.k(i, "ep"), 1.0f);
        this.f1823d.i(dVar.k(i, "em"), -1.0f);
        return this;
    }

    @Override // b.h.b.d.a
    public void clear() {
        this.f1823d.clear();
        this.f1820a = null;
        this.f1821b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public b d(h hVar, h hVar2, h hVar3, h hVar4, float f2) {
        this.f1823d.i(hVar, -1.0f);
        this.f1823d.i(hVar2, 1.0f);
        this.f1823d.i(hVar3, f2);
        this.f1823d.i(hVar4, -f2);
        return this;
    }

    public b e(h hVar, h hVar2, h hVar3, int i) {
        boolean z = false;
        if (i != 0) {
            if (i < 0) {
                i *= -1;
                z = true;
            }
            this.f1821b = i;
        }
        if (!z) {
            this.f1823d.i(hVar, -1.0f);
            this.f1823d.i(hVar2, 1.0f);
            this.f1823d.i(hVar3, 1.0f);
        } else {
            this.f1823d.i(hVar, 1.0f);
            this.f1823d.i(hVar2, -1.0f);
            this.f1823d.i(hVar3, -1.0f);
        }
        return this;
    }

    public b f(h hVar, h hVar2, h hVar3, int i) {
        boolean z = false;
        if (i != 0) {
            if (i < 0) {
                i *= -1;
                z = true;
            }
            this.f1821b = i;
        }
        if (!z) {
            this.f1823d.i(hVar, -1.0f);
            this.f1823d.i(hVar2, 1.0f);
            this.f1823d.i(hVar3, -1.0f);
        } else {
            this.f1823d.i(hVar, 1.0f);
            this.f1823d.i(hVar2, -1.0f);
            this.f1823d.i(hVar3, 1.0f);
        }
        return this;
    }

    public b g(h hVar, h hVar2, h hVar3, h hVar4, float f2) {
        this.f1823d.i(hVar3, 0.5f);
        this.f1823d.i(hVar4, 0.5f);
        this.f1823d.i(hVar, -0.5f);
        this.f1823d.i(hVar2, -0.5f);
        this.f1821b = -f2;
        return this;
    }

    public final boolean h(h hVar) {
        return hVar.m <= 1;
    }

    public final h i(boolean[] zArr, h hVar) {
        int i;
        int a2 = this.f1823d.a();
        h hVar2 = null;
        float f2 = 0.0f;
        for (int i2 = 0; i2 < a2; i2++) {
            float d2 = this.f1823d.d(i2);
            if (d2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                h b2 = this.f1823d.b(i2);
                if ((zArr == null || !zArr[b2.f1846c]) && b2 != hVar && (((i = b2.j) == 3 || i == 4) && d2 < f2)) {
                    f2 = d2;
                    hVar2 = b2;
                }
            }
        }
        return hVar2;
    }

    @Override // b.h.b.d.a
    public boolean isEmpty() {
        return this.f1820a == null && this.f1821b == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && this.f1823d.a() == 0;
    }

    public void j(h hVar) {
        h hVar2 = this.f1820a;
        if (hVar2 != null) {
            this.f1823d.i(hVar2, -1.0f);
            this.f1820a.f1847d = -1;
            this.f1820a = null;
        }
        float j = this.f1823d.j(hVar, true) * (-1.0f);
        this.f1820a = hVar;
        if (j == 1.0f) {
            return;
        }
        this.f1821b /= j;
        this.f1823d.k(j);
    }

    public void k(d dVar, h hVar, boolean z) {
        if (hVar.f1850g) {
            float f2 = this.f1823d.f(hVar);
            this.f1821b = (hVar.f1849f * f2) + this.f1821b;
            this.f1823d.j(hVar, z);
            if (z) {
                hVar.b(this);
            }
            if (this.f1823d.a() == 0) {
                this.f1824e = true;
                dVar.f1832d = true;
            }
        }
    }

    public void l(d dVar, b bVar, boolean z) {
        float h2 = this.f1823d.h(bVar, z);
        this.f1821b = (bVar.f1821b * h2) + this.f1821b;
        if (z) {
            bVar.f1820a.b(this);
        }
        if (this.f1820a == null || this.f1823d.a() != 0) {
            return;
        }
        this.f1824e = true;
        dVar.f1832d = true;
    }

    /* JADX WARN: Removed duplicated region for block: B:29:0x007a  */
    /* JADX WARN: Removed duplicated region for block: B:30:0x007f  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public String toString() {
        String sb;
        boolean z;
        float d2;
        int i;
        if (this.f1820a == null) {
            sb = CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
        } else {
            StringBuilder x = c.b.a.a.a.x("");
            x.append(this.f1820a);
            sb = x.toString();
        }
        String q = c.b.a.a.a.q(sb, " = ");
        if (this.f1821b != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            StringBuilder x2 = c.b.a.a.a.x(q);
            x2.append(this.f1821b);
            q = x2.toString();
            z = true;
        } else {
            z = false;
        }
        int a2 = this.f1823d.a();
        for (int i2 = 0; i2 < a2; i2++) {
            h b2 = this.f1823d.b(i2);
            if (b2 != null && (this.f1823d.d(i2)) != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                String hVar = b2.toString();
                if (!z) {
                    if (d2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        q = c.b.a.a.a.q(q, "- ");
                        d2 *= -1.0f;
                    }
                    q = d2 == 1.0f ? c.b.a.a.a.q(q, hVar) : q + d2 + " " + hVar;
                    z = true;
                } else if (i > 0) {
                    q = c.b.a.a.a.q(q, " + ");
                    if (d2 == 1.0f) {
                    }
                    z = true;
                } else {
                    q = c.b.a.a.a.q(q, " - ");
                    d2 *= -1.0f;
                    if (d2 == 1.0f) {
                    }
                    z = true;
                }
            }
        }
        return !z ? c.b.a.a.a.q(q, IdManager.DEFAULT_VERSION_NAME) : q;
    }

    public b(c cVar) {
        this.f1823d = new b.h.b.a(this, cVar);
    }
}