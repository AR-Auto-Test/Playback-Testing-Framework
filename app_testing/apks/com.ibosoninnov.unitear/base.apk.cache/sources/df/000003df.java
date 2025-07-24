package b.h.b;

import b.h.b.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;

/* compiled from: ArrayLinkedVariables.java */
/* loaded from: classes.dex */
public class a implements b.a {

    /* renamed from: b  reason: collision with root package name */
    public final b f1813b;

    /* renamed from: c  reason: collision with root package name */
    public final c f1814c;

    /* renamed from: a  reason: collision with root package name */
    public int f1812a = 0;

    /* renamed from: d  reason: collision with root package name */
    public int f1815d = 8;

    /* renamed from: e  reason: collision with root package name */
    public int[] f1816e = new int[8];

    /* renamed from: f  reason: collision with root package name */
    public int[] f1817f = new int[8];

    /* renamed from: g  reason: collision with root package name */
    public float[] f1818g = new float[8];

    /* renamed from: h  reason: collision with root package name */
    public int f1819h = -1;
    public int i = -1;
    public boolean j = false;

    public a(b bVar, c cVar) {
        this.f1813b = bVar;
        this.f1814c = cVar;
    }

    @Override // b.h.b.b.a
    public int a() {
        return this.f1812a;
    }

    @Override // b.h.b.b.a
    public h b(int i) {
        int i2 = this.f1819h;
        for (int i3 = 0; i2 != -1 && i3 < this.f1812a; i3++) {
            if (i3 == i) {
                return this.f1814c.f1828d[this.f1816e[i2]];
            }
            i2 = this.f1817f[i2];
        }
        return null;
    }

    @Override // b.h.b.b.a
    public void c() {
        int i = this.f1819h;
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            float[] fArr = this.f1818g;
            fArr[i] = fArr[i] * (-1.0f);
            i = this.f1817f[i];
        }
    }

    @Override // b.h.b.b.a
    public final void clear() {
        int i = this.f1819h;
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            h hVar = this.f1814c.f1828d[this.f1816e[i]];
            if (hVar != null) {
                hVar.b(this.f1813b);
            }
            i = this.f1817f[i];
        }
        this.f1819h = -1;
        this.i = -1;
        this.j = false;
        this.f1812a = 0;
    }

    @Override // b.h.b.b.a
    public float d(int i) {
        int i2 = this.f1819h;
        for (int i3 = 0; i2 != -1 && i3 < this.f1812a; i3++) {
            if (i3 == i) {
                return this.f1818g[i2];
            }
            i2 = this.f1817f[i2];
        }
        return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    @Override // b.h.b.b.a
    public void e(h hVar, float f2, boolean z) {
        if (f2 <= -0.001f || f2 >= 0.001f) {
            int i = this.f1819h;
            if (i == -1) {
                this.f1819h = 0;
                this.f1818g[0] = f2;
                this.f1816e[0] = hVar.f1846c;
                this.f1817f[0] = -1;
                hVar.m++;
                hVar.a(this.f1813b);
                this.f1812a++;
                if (this.j) {
                    return;
                }
                int i2 = this.i + 1;
                this.i = i2;
                int[] iArr = this.f1816e;
                if (i2 >= iArr.length) {
                    this.j = true;
                    this.i = iArr.length - 1;
                    return;
                }
                return;
            }
            int i3 = -1;
            for (int i4 = 0; i != -1 && i4 < this.f1812a; i4++) {
                int[] iArr2 = this.f1816e;
                int i5 = iArr2[i];
                int i6 = hVar.f1846c;
                if (i5 == i6) {
                    float[] fArr = this.f1818g;
                    float f3 = fArr[i] + f2;
                    if (f3 > -0.001f && f3 < 0.001f) {
                        f3 = 0.0f;
                    }
                    fArr[i] = f3;
                    if (f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        if (i == this.f1819h) {
                            this.f1819h = this.f1817f[i];
                        } else {
                            int[] iArr3 = this.f1817f;
                            iArr3[i3] = iArr3[i];
                        }
                        if (z) {
                            hVar.b(this.f1813b);
                        }
                        if (this.j) {
                            this.i = i;
                        }
                        hVar.m--;
                        this.f1812a--;
                        return;
                    }
                    return;
                }
                if (iArr2[i] < i6) {
                    i3 = i;
                }
                i = this.f1817f[i];
            }
            int i7 = this.i;
            int i8 = i7 + 1;
            if (this.j) {
                int[] iArr4 = this.f1816e;
                if (iArr4[i7] != -1) {
                    i7 = iArr4.length;
                }
            } else {
                i7 = i8;
            }
            int[] iArr5 = this.f1816e;
            if (i7 >= iArr5.length && this.f1812a < iArr5.length) {
                int i9 = 0;
                while (true) {
                    int[] iArr6 = this.f1816e;
                    if (i9 >= iArr6.length) {
                        break;
                    } else if (iArr6[i9] == -1) {
                        i7 = i9;
                        break;
                    } else {
                        i9++;
                    }
                }
            }
            int[] iArr7 = this.f1816e;
            if (i7 >= iArr7.length) {
                i7 = iArr7.length;
                int i10 = this.f1815d * 2;
                this.f1815d = i10;
                this.j = false;
                this.i = i7 - 1;
                this.f1818g = Arrays.copyOf(this.f1818g, i10);
                this.f1816e = Arrays.copyOf(this.f1816e, this.f1815d);
                this.f1817f = Arrays.copyOf(this.f1817f, this.f1815d);
            }
            this.f1816e[i7] = hVar.f1846c;
            this.f1818g[i7] = f2;
            if (i3 != -1) {
                int[] iArr8 = this.f1817f;
                iArr8[i7] = iArr8[i3];
                iArr8[i3] = i7;
            } else {
                this.f1817f[i7] = this.f1819h;
                this.f1819h = i7;
            }
            hVar.m++;
            hVar.a(this.f1813b);
            this.f1812a++;
            if (!this.j) {
                this.i++;
            }
            int i11 = this.i;
            int[] iArr9 = this.f1816e;
            if (i11 >= iArr9.length) {
                this.j = true;
                this.i = iArr9.length - 1;
            }
        }
    }

    @Override // b.h.b.b.a
    public final float f(h hVar) {
        int i = this.f1819h;
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            if (this.f1816e[i] == hVar.f1846c) {
                return this.f1818g[i];
            }
            i = this.f1817f[i];
        }
        return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    @Override // b.h.b.b.a
    public boolean g(h hVar) {
        int i = this.f1819h;
        if (i == -1) {
            return false;
        }
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            if (this.f1816e[i] == hVar.f1846c) {
                return true;
            }
            i = this.f1817f[i];
        }
        return false;
    }

    @Override // b.h.b.b.a
    public float h(b bVar, boolean z) {
        float f2 = f(bVar.f1820a);
        j(bVar.f1820a, z);
        b.a aVar = bVar.f1823d;
        int a2 = aVar.a();
        for (int i = 0; i < a2; i++) {
            h b2 = aVar.b(i);
            e(b2, aVar.f(b2) * f2, z);
        }
        return f2;
    }

    @Override // b.h.b.b.a
    public final void i(h hVar, float f2) {
        if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            j(hVar, true);
            return;
        }
        int i = this.f1819h;
        if (i == -1) {
            this.f1819h = 0;
            this.f1818g[0] = f2;
            this.f1816e[0] = hVar.f1846c;
            this.f1817f[0] = -1;
            hVar.m++;
            hVar.a(this.f1813b);
            this.f1812a++;
            if (this.j) {
                return;
            }
            int i2 = this.i + 1;
            this.i = i2;
            int[] iArr = this.f1816e;
            if (i2 >= iArr.length) {
                this.j = true;
                this.i = iArr.length - 1;
                return;
            }
            return;
        }
        int i3 = -1;
        for (int i4 = 0; i != -1 && i4 < this.f1812a; i4++) {
            int[] iArr2 = this.f1816e;
            int i5 = iArr2[i];
            int i6 = hVar.f1846c;
            if (i5 == i6) {
                this.f1818g[i] = f2;
                return;
            }
            if (iArr2[i] < i6) {
                i3 = i;
            }
            i = this.f1817f[i];
        }
        int i7 = this.i;
        int i8 = i7 + 1;
        if (this.j) {
            int[] iArr3 = this.f1816e;
            if (iArr3[i7] != -1) {
                i7 = iArr3.length;
            }
        } else {
            i7 = i8;
        }
        int[] iArr4 = this.f1816e;
        if (i7 >= iArr4.length && this.f1812a < iArr4.length) {
            int i9 = 0;
            while (true) {
                int[] iArr5 = this.f1816e;
                if (i9 >= iArr5.length) {
                    break;
                } else if (iArr5[i9] == -1) {
                    i7 = i9;
                    break;
                } else {
                    i9++;
                }
            }
        }
        int[] iArr6 = this.f1816e;
        if (i7 >= iArr6.length) {
            i7 = iArr6.length;
            int i10 = this.f1815d * 2;
            this.f1815d = i10;
            this.j = false;
            this.i = i7 - 1;
            this.f1818g = Arrays.copyOf(this.f1818g, i10);
            this.f1816e = Arrays.copyOf(this.f1816e, this.f1815d);
            this.f1817f = Arrays.copyOf(this.f1817f, this.f1815d);
        }
        this.f1816e[i7] = hVar.f1846c;
        this.f1818g[i7] = f2;
        if (i3 != -1) {
            int[] iArr7 = this.f1817f;
            iArr7[i7] = iArr7[i3];
            iArr7[i3] = i7;
        } else {
            this.f1817f[i7] = this.f1819h;
            this.f1819h = i7;
        }
        hVar.m++;
        hVar.a(this.f1813b);
        int i11 = this.f1812a + 1;
        this.f1812a = i11;
        if (!this.j) {
            this.i++;
        }
        int[] iArr8 = this.f1816e;
        if (i11 >= iArr8.length) {
            this.j = true;
        }
        if (this.i >= iArr8.length) {
            this.j = true;
            this.i = iArr8.length - 1;
        }
    }

    @Override // b.h.b.b.a
    public final float j(h hVar, boolean z) {
        int i = this.f1819h;
        if (i == -1) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        int i2 = 0;
        int i3 = -1;
        while (i != -1 && i2 < this.f1812a) {
            if (this.f1816e[i] == hVar.f1846c) {
                if (i == this.f1819h) {
                    this.f1819h = this.f1817f[i];
                } else {
                    int[] iArr = this.f1817f;
                    iArr[i3] = iArr[i];
                }
                if (z) {
                    hVar.b(this.f1813b);
                }
                hVar.m--;
                this.f1812a--;
                this.f1816e[i] = -1;
                if (this.j) {
                    this.i = i;
                }
                return this.f1818g[i];
            }
            i2++;
            i3 = i;
            i = this.f1817f[i];
        }
        return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    @Override // b.h.b.b.a
    public void k(float f2) {
        int i = this.f1819h;
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            float[] fArr = this.f1818g;
            fArr[i] = fArr[i] / f2;
            i = this.f1817f[i];
        }
    }

    public String toString() {
        int i = this.f1819h;
        String str = "";
        for (int i2 = 0; i != -1 && i2 < this.f1812a; i2++) {
            StringBuilder x = c.b.a.a.a.x(c.b.a.a.a.q(str, " -> "));
            x.append(this.f1818g[i]);
            x.append(" : ");
            StringBuilder x2 = c.b.a.a.a.x(x.toString());
            x2.append(this.f1814c.f1828d[this.f1816e[i]]);
            str = x2.toString();
            i = this.f1817f[i];
        }
        return str;
    }
}