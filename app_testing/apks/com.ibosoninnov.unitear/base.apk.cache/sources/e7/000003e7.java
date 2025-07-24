package b.h.b;

import b.h.b.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

/* compiled from: PriorityGoalRow.java */
/* loaded from: classes.dex */
public class g extends b.h.b.b {

    /* renamed from: f  reason: collision with root package name */
    public int f1839f;

    /* renamed from: g  reason: collision with root package name */
    public h[] f1840g;

    /* renamed from: h  reason: collision with root package name */
    public h[] f1841h;
    public int i;
    public b j;

    /* compiled from: PriorityGoalRow.java */
    /* loaded from: classes.dex */
    public class a implements Comparator<h> {
        public a(g gVar) {
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(h hVar, h hVar2) {
            return hVar.f1846c - hVar2.f1846c;
        }
    }

    /* compiled from: PriorityGoalRow.java */
    /* loaded from: classes.dex */
    public class b implements Comparable {

        /* renamed from: b  reason: collision with root package name */
        public h f1842b;

        public b(g gVar) {
        }

        @Override // java.lang.Comparable
        public int compareTo(Object obj) {
            return this.f1842b.f1846c - ((h) obj).f1846c;
        }

        public String toString() {
            String str = "[ ";
            if (this.f1842b != null) {
                for (int i = 0; i < 9; i++) {
                    StringBuilder x = c.b.a.a.a.x(str);
                    x.append(this.f1842b.i[i]);
                    x.append(" ");
                    str = x.toString();
                }
            }
            StringBuilder A = c.b.a.a.a.A(str, "] ");
            A.append(this.f1842b);
            return A.toString();
        }
    }

    public g(c cVar) {
        super(cVar);
        this.f1839f = 128;
        this.f1840g = new h[128];
        this.f1841h = new h[128];
        this.i = 0;
        this.j = new b(this);
    }

    @Override // b.h.b.b, b.h.b.d.a
    public void a(h hVar) {
        this.j.f1842b = hVar;
        Arrays.fill(hVar.i, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        hVar.i[hVar.f1848e] = 1.0f;
        m(hVar);
    }

    /* JADX WARN: Code restructure failed: missing block: B:28:0x0053, code lost:
        if (r8 < r7) goto L33;
     */
    @Override // b.h.b.b, b.h.b.d.a
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public h b(d dVar, boolean[] zArr) {
        int i = -1;
        for (int i2 = 0; i2 < this.i; i2++) {
            h[] hVarArr = this.f1840g;
            h hVar = hVarArr[i2];
            if (!zArr[hVar.f1846c]) {
                b bVar = this.j;
                bVar.f1842b = hVar;
                int i3 = 8;
                boolean z = true;
                if (i == -1) {
                    Objects.requireNonNull(bVar);
                    while (i3 >= 0) {
                        float f2 = bVar.f1842b.i[i3];
                        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            break;
                        } else if (f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                            break;
                        } else {
                            i3--;
                        }
                    }
                    z = false;
                    if (!z) {
                    }
                    i = i2;
                } else {
                    h hVar2 = hVarArr[i];
                    Objects.requireNonNull(bVar);
                    while (true) {
                        if (i3 < 0) {
                            break;
                        }
                        float f3 = hVar2.i[i3];
                        float f4 = bVar.f1842b.i[i3];
                        if (f4 == f3) {
                            i3--;
                        }
                    }
                    z = false;
                    if (!z) {
                    }
                    i = i2;
                }
            }
        }
        if (i == -1) {
            return null;
        }
        return this.f1840g[i];
    }

    @Override // b.h.b.b, b.h.b.d.a
    public void clear() {
        this.i = 0;
        this.f1821b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    @Override // b.h.b.b, b.h.b.d.a
    public boolean isEmpty() {
        return this.i == 0;
    }

    @Override // b.h.b.b
    public void l(d dVar, b.h.b.b bVar, boolean z) {
        h hVar = bVar.f1820a;
        if (hVar == null) {
            return;
        }
        b.a aVar = bVar.f1823d;
        int a2 = aVar.a();
        for (int i = 0; i < a2; i++) {
            h b2 = aVar.b(i);
            float d2 = aVar.d(i);
            b bVar2 = this.j;
            bVar2.f1842b = b2;
            boolean z2 = true;
            if (b2.f1845b) {
                for (int i2 = 0; i2 < 9; i2++) {
                    float[] fArr = bVar2.f1842b.i;
                    fArr[i2] = (hVar.i[i2] * d2) + fArr[i2];
                    if (Math.abs(fArr[i2]) < 1.0E-4f) {
                        bVar2.f1842b.i[i2] = 0.0f;
                    } else {
                        z2 = false;
                    }
                }
                if (z2) {
                    g.this.n(bVar2.f1842b);
                }
                z2 = false;
            } else {
                for (int i3 = 0; i3 < 9; i3++) {
                    float f2 = hVar.i[i3];
                    if (f2 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        float f3 = f2 * d2;
                        if (Math.abs(f3) < 1.0E-4f) {
                            f3 = 0.0f;
                        }
                        bVar2.f1842b.i[i3] = f3;
                    } else {
                        bVar2.f1842b.i[i3] = 0.0f;
                    }
                }
            }
            if (z2) {
                m(b2);
            }
            this.f1821b = (bVar.f1821b * d2) + this.f1821b;
        }
        n(hVar);
    }

    public final void m(h hVar) {
        int i;
        int i2 = this.i + 1;
        h[] hVarArr = this.f1840g;
        if (i2 > hVarArr.length) {
            h[] hVarArr2 = (h[]) Arrays.copyOf(hVarArr, hVarArr.length * 2);
            this.f1840g = hVarArr2;
            this.f1841h = (h[]) Arrays.copyOf(hVarArr2, hVarArr2.length * 2);
        }
        h[] hVarArr3 = this.f1840g;
        int i3 = this.i;
        hVarArr3[i3] = hVar;
        int i4 = i3 + 1;
        this.i = i4;
        if (i4 > 1 && hVarArr3[i4 - 1].f1846c > hVar.f1846c) {
            int i5 = 0;
            while (true) {
                i = this.i;
                if (i5 >= i) {
                    break;
                }
                this.f1841h[i5] = this.f1840g[i5];
                i5++;
            }
            Arrays.sort(this.f1841h, 0, i, new a(this));
            for (int i6 = 0; i6 < this.i; i6++) {
                this.f1840g[i6] = this.f1841h[i6];
            }
        }
        hVar.f1845b = true;
        hVar.a(this);
    }

    public final void n(h hVar) {
        int i = 0;
        while (i < this.i) {
            if (this.f1840g[i] == hVar) {
                while (true) {
                    int i2 = this.i;
                    if (i < i2 - 1) {
                        h[] hVarArr = this.f1840g;
                        int i3 = i + 1;
                        hVarArr[i] = hVarArr[i3];
                        i = i3;
                    } else {
                        this.i = i2 - 1;
                        hVar.f1845b = false;
                        return;
                    }
                }
            } else {
                i++;
            }
        }
    }

    @Override // b.h.b.b
    public String toString() {
        StringBuilder A = c.b.a.a.a.A("", " goal -> (");
        A.append(this.f1821b);
        A.append(") : ");
        String sb = A.toString();
        for (int i = 0; i < this.i; i++) {
            this.j.f1842b = this.f1840g[i];
            StringBuilder x = c.b.a.a.a.x(sb);
            x.append(this.j);
            x.append(" ");
            sb = x.toString();
        }
        return sb;
    }
}