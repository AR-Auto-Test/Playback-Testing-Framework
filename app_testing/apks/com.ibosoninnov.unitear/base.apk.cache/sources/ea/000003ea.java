package b.h.b;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;

/* compiled from: SolverVariable.java */
/* loaded from: classes.dex */
public class h {

    /* renamed from: a  reason: collision with root package name */
    public static int f1844a = 1;

    /* renamed from: b  reason: collision with root package name */
    public boolean f1845b;

    /* renamed from: f  reason: collision with root package name */
    public float f1849f;
    public int j;

    /* renamed from: c  reason: collision with root package name */
    public int f1846c = -1;

    /* renamed from: d  reason: collision with root package name */
    public int f1847d = -1;

    /* renamed from: e  reason: collision with root package name */
    public int f1848e = 0;

    /* renamed from: g  reason: collision with root package name */
    public boolean f1850g = false;

    /* renamed from: h  reason: collision with root package name */
    public float[] f1851h = new float[9];
    public float[] i = new float[9];
    public b[] k = new b[16];
    public int l = 0;
    public int m = 0;
    public int n = -1;

    public h(int i) {
        this.j = i;
    }

    public final void a(b bVar) {
        int i = 0;
        while (true) {
            int i2 = this.l;
            if (i < i2) {
                if (this.k[i] == bVar) {
                    return;
                }
                i++;
            } else {
                b[] bVarArr = this.k;
                if (i2 >= bVarArr.length) {
                    this.k = (b[]) Arrays.copyOf(bVarArr, bVarArr.length * 2);
                }
                b[] bVarArr2 = this.k;
                int i3 = this.l;
                bVarArr2[i3] = bVar;
                this.l = i3 + 1;
                return;
            }
        }
    }

    public final void b(b bVar) {
        int i = this.l;
        int i2 = 0;
        while (i2 < i) {
            if (this.k[i2] == bVar) {
                while (i2 < i - 1) {
                    b[] bVarArr = this.k;
                    int i3 = i2 + 1;
                    bVarArr[i2] = bVarArr[i3];
                    i2 = i3;
                }
                this.l--;
                return;
            }
            i2++;
        }
    }

    public void c() {
        this.j = 5;
        this.f1848e = 0;
        this.f1846c = -1;
        this.f1847d = -1;
        this.f1849f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.f1850g = false;
        this.n = -1;
        int i = this.l;
        for (int i2 = 0; i2 < i; i2++) {
            this.k[i2] = null;
        }
        this.l = 0;
        this.m = 0;
        this.f1845b = false;
        Arrays.fill(this.i, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
    }

    public void d(d dVar, float f2) {
        this.f1849f = f2;
        this.f1850g = true;
        this.n = -1;
        int i = this.l;
        this.f1847d = -1;
        for (int i2 = 0; i2 < i; i2++) {
            this.k[i2].k(dVar, this, false);
        }
        this.l = 0;
    }

    public final void e(d dVar, b bVar) {
        int i = this.l;
        for (int i2 = 0; i2 < i; i2++) {
            this.k[i2].l(dVar, bVar, false);
        }
        this.l = 0;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("");
        x.append(this.f1846c);
        return x.toString();
    }
}