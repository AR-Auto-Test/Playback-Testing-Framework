package g;

import java.util.Arrays;

/* compiled from: SegmentedByteString.java */
/* loaded from: classes2.dex */
public final class v extends h {

    /* renamed from: g  reason: collision with root package name */
    public final transient byte[][] f6218g;

    /* renamed from: h  reason: collision with root package name */
    public final transient int[] f6219h;

    public v(e eVar, int i) {
        super(null);
        z.b(eVar.f6176d, 0L, i);
        t tVar = eVar.f6175c;
        int i2 = 0;
        int i3 = 0;
        int i4 = 0;
        while (i3 < i) {
            int i5 = tVar.f6211c;
            int i6 = tVar.f6210b;
            if (i5 != i6) {
                i3 += i5 - i6;
                i4++;
                tVar = tVar.f6214f;
            } else {
                throw new AssertionError("s.limit == s.pos");
            }
        }
        this.f6218g = new byte[i4];
        this.f6219h = new int[i4 * 2];
        t tVar2 = eVar.f6175c;
        int i7 = 0;
        while (i2 < i) {
            byte[][] bArr = this.f6218g;
            bArr[i7] = tVar2.f6209a;
            int i8 = tVar2.f6211c;
            int i9 = tVar2.f6210b;
            int i10 = (i8 - i9) + i2;
            i2 = i10 > i ? i : i10;
            int[] iArr = this.f6219h;
            iArr[i7] = i2;
            iArr[bArr.length + i7] = i9;
            tVar2.f6212d = true;
            i7++;
            tVar2 = tVar2.f6214f;
        }
    }

    @Override // g.h
    public String a() {
        return s().a();
    }

    @Override // g.h
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof h) {
            h hVar = (h) obj;
            if (hVar.l() == l() && j(0, hVar, 0, l())) {
                return true;
            }
        }
        return false;
    }

    @Override // g.h
    public byte f(int i) {
        z.b(this.f6219h[this.f6218g.length - 1], i, 1L);
        int r = r(i);
        int i2 = r == 0 ? 0 : this.f6219h[r - 1];
        int[] iArr = this.f6219h;
        byte[][] bArr = this.f6218g;
        return bArr[r][(i - i2) + iArr[bArr.length + r]];
    }

    @Override // g.h
    public String g() {
        return s().g();
    }

    @Override // g.h
    public byte[] h() {
        return o();
    }

    @Override // g.h
    public int hashCode() {
        int i = this.f6181e;
        if (i != 0) {
            return i;
        }
        int length = this.f6218g.length;
        int i2 = 0;
        int i3 = 1;
        int i4 = 0;
        while (i2 < length) {
            byte[] bArr = this.f6218g[i2];
            int[] iArr = this.f6219h;
            int i5 = iArr[length + i2];
            int i6 = iArr[i2];
            int i7 = (i6 - i4) + i5;
            while (i5 < i7) {
                i3 = (i3 * 31) + bArr[i5];
                i5++;
            }
            i2++;
            i4 = i6;
        }
        this.f6181e = i3;
        return i3;
    }

    @Override // g.h
    public boolean j(int i, h hVar, int i2, int i3) {
        if (i < 0 || i > l() - i3) {
            return false;
        }
        int r = r(i);
        while (i3 > 0) {
            int i4 = r == 0 ? 0 : this.f6219h[r - 1];
            int min = Math.min(i3, ((this.f6219h[r] - i4) + i4) - i);
            int[] iArr = this.f6219h;
            byte[][] bArr = this.f6218g;
            if (!hVar.k(i2, bArr[r], (i - i4) + iArr[bArr.length + r], min)) {
                return false;
            }
            i += min;
            i2 += min;
            i3 -= min;
            r++;
        }
        return true;
    }

    @Override // g.h
    public boolean k(int i, byte[] bArr, int i2, int i3) {
        if (i < 0 || i > l() - i3 || i2 < 0 || i2 > bArr.length - i3) {
            return false;
        }
        int r = r(i);
        while (i3 > 0) {
            int i4 = r == 0 ? 0 : this.f6219h[r - 1];
            int min = Math.min(i3, ((this.f6219h[r] - i4) + i4) - i);
            int[] iArr = this.f6219h;
            byte[][] bArr2 = this.f6218g;
            if (!z.a(bArr2[r], (i - i4) + iArr[bArr2.length + r], bArr, i2, min)) {
                return false;
            }
            i += min;
            i2 += min;
            i3 -= min;
            r++;
        }
        return true;
    }

    @Override // g.h
    public int l() {
        return this.f6219h[this.f6218g.length - 1];
    }

    @Override // g.h
    public h m(int i, int i2) {
        return s().m(i, i2);
    }

    @Override // g.h
    public h n() {
        return s().n();
    }

    @Override // g.h
    public byte[] o() {
        int[] iArr = this.f6219h;
        byte[][] bArr = this.f6218g;
        byte[] bArr2 = new byte[iArr[bArr.length - 1]];
        int length = bArr.length;
        int i = 0;
        int i2 = 0;
        while (i < length) {
            int[] iArr2 = this.f6219h;
            int i3 = iArr2[length + i];
            int i4 = iArr2[i];
            System.arraycopy(this.f6218g[i], i3, bArr2, i2, i4 - i2);
            i++;
            i2 = i4;
        }
        return bArr2;
    }

    @Override // g.h
    public String p() {
        return s().p();
    }

    @Override // g.h
    public void q(e eVar) {
        int length = this.f6218g.length;
        int i = 0;
        int i2 = 0;
        while (i < length) {
            int[] iArr = this.f6219h;
            int i3 = iArr[length + i];
            int i4 = iArr[i];
            t tVar = new t(this.f6218g[i], i3, (i3 + i4) - i2, true, false);
            t tVar2 = eVar.f6175c;
            if (tVar2 == null) {
                tVar.f6215g = tVar;
                tVar.f6214f = tVar;
                eVar.f6175c = tVar;
            } else {
                tVar2.f6215g.b(tVar);
            }
            i++;
            i2 = i4;
        }
        eVar.f6176d += i2;
    }

    public final int r(int i) {
        int binarySearch = Arrays.binarySearch(this.f6219h, 0, this.f6218g.length, i + 1);
        return binarySearch >= 0 ? binarySearch : ~binarySearch;
    }

    public final h s() {
        return new h(o());
    }

    @Override // g.h
    public String toString() {
        return s().toString();
    }
}