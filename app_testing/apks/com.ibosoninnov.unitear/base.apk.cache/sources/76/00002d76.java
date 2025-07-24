package g;

import com.google.common.primitives.UnsignedBytes;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

/* compiled from: ByteString.java */
/* loaded from: classes2.dex */
public class h implements Serializable, Comparable<h> {

    /* renamed from: b  reason: collision with root package name */
    public static final char[] f6178b = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    /* renamed from: c  reason: collision with root package name */
    public static final h f6179c = i(new byte[0]);

    /* renamed from: d  reason: collision with root package name */
    public final byte[] f6180d;

    /* renamed from: e  reason: collision with root package name */
    public transient int f6181e;

    /* renamed from: f  reason: collision with root package name */
    public transient String f6182f;

    public h(byte[] bArr) {
        this.f6180d = bArr;
    }

    public static h b(String str) {
        if (str.length() % 2 == 0) {
            int length = str.length() / 2;
            byte[] bArr = new byte[length];
            for (int i = 0; i < length; i++) {
                int i2 = i * 2;
                bArr[i] = (byte) (c(str.charAt(i2 + 1)) + (c(str.charAt(i2)) << 4));
            }
            return i(bArr);
        }
        throw new IllegalArgumentException(c.b.a.a.a.q("Unexpected hex string: ", str));
    }

    public static int c(char c2) {
        if (c2 < '0' || c2 > '9') {
            char c3 = 'a';
            if (c2 < 'a' || c2 > 'f') {
                c3 = 'A';
                if (c2 < 'A' || c2 > 'F') {
                    throw new IllegalArgumentException("Unexpected hex digit: " + c2);
                }
            }
            return (c2 - c3) + 10;
        }
        return c2 - '0';
    }

    public static h e(String str) {
        if (str != null) {
            h hVar = new h(str.getBytes(z.f6224a));
            hVar.f6182f = str;
            return hVar;
        }
        throw new IllegalArgumentException("s == null");
    }

    public static h i(byte... bArr) {
        if (bArr != null) {
            return new h((byte[]) bArr.clone());
        }
        throw new IllegalArgumentException("data == null");
    }

    public String a() {
        byte[] bArr = this.f6180d;
        byte[] bArr2 = d.f6173a;
        byte[] bArr3 = new byte[((bArr.length + 2) / 3) * 4];
        int length = bArr.length - (bArr.length % 3);
        int i = 0;
        for (int i2 = 0; i2 < length; i2 += 3) {
            int i3 = i + 1;
            bArr3[i] = bArr2[(bArr[i2] & UnsignedBytes.MAX_VALUE) >> 2];
            int i4 = i3 + 1;
            int i5 = i2 + 1;
            bArr3[i3] = bArr2[((bArr[i2] & 3) << 4) | ((bArr[i5] & UnsignedBytes.MAX_VALUE) >> 4)];
            int i6 = i4 + 1;
            int i7 = i2 + 2;
            bArr3[i4] = bArr2[((bArr[i5] & 15) << 2) | ((bArr[i7] & UnsignedBytes.MAX_VALUE) >> 6)];
            i = i6 + 1;
            bArr3[i6] = bArr2[bArr[i7] & 63];
        }
        int length2 = bArr.length % 3;
        if (length2 == 1) {
            int i8 = i + 1;
            bArr3[i] = bArr2[(bArr[length] & UnsignedBytes.MAX_VALUE) >> 2];
            int i9 = i8 + 1;
            bArr3[i8] = bArr2[(bArr[length] & 3) << 4];
            bArr3[i9] = 61;
            bArr3[i9 + 1] = 61;
        } else if (length2 == 2) {
            int i10 = i + 1;
            bArr3[i] = bArr2[(bArr[length] & UnsignedBytes.MAX_VALUE) >> 2];
            int i11 = i10 + 1;
            int i12 = length + 1;
            bArr3[i10] = bArr2[((bArr[i12] & UnsignedBytes.MAX_VALUE) >> 4) | ((bArr[length] & 3) << 4)];
            bArr3[i11] = bArr2[(bArr[i12] & 15) << 2];
            bArr3[i11 + 1] = 61;
        }
        try {
            return new String(bArr3, "US-ASCII");
        } catch (UnsupportedEncodingException e2) {
            throw new AssertionError(e2);
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    /* JADX WARN: Code restructure failed: missing block: B:13:0x002e, code lost:
        if (r0 < r1) goto L9;
     */
    /* JADX WARN: Code restructure failed: missing block: B:15:0x0031, code lost:
        return -1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:18:?, code lost:
        return 1;
     */
    /* JADX WARN: Code restructure failed: missing block: B:8:0x0025, code lost:
        if (r7 < r8) goto L9;
     */
    @Override // java.lang.Comparable
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public int compareTo(h hVar) {
        h hVar2 = hVar;
        int l = l();
        int l2 = hVar2.l();
        int min = Math.min(l, l2);
        for (int i = 0; i < min; i++) {
            int f2 = f(i) & UnsignedBytes.MAX_VALUE;
            int f3 = hVar2.f(i) & UnsignedBytes.MAX_VALUE;
            if (f2 == f3) {
            }
        }
        if (l == l2) {
            return 0;
        }
    }

    public final h d(String str) {
        try {
            return i(MessageDigest.getInstance(str).digest(this.f6180d));
        } catch (NoSuchAlgorithmException e2) {
            throw new AssertionError(e2);
        }
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof h) {
            h hVar = (h) obj;
            int l = hVar.l();
            byte[] bArr = this.f6180d;
            if (l == bArr.length && hVar.k(0, bArr, 0, bArr.length)) {
                return true;
            }
        }
        return false;
    }

    public byte f(int i) {
        return this.f6180d[i];
    }

    public String g() {
        byte[] bArr = this.f6180d;
        char[] cArr = new char[bArr.length * 2];
        int i = 0;
        for (byte b2 : bArr) {
            int i2 = i + 1;
            char[] cArr2 = f6178b;
            cArr[i] = cArr2[(b2 >> 4) & 15];
            i = i2 + 1;
            cArr[i2] = cArr2[b2 & 15];
        }
        return new String(cArr);
    }

    public byte[] h() {
        return this.f6180d;
    }

    public int hashCode() {
        int i = this.f6181e;
        if (i != 0) {
            return i;
        }
        int hashCode = Arrays.hashCode(this.f6180d);
        this.f6181e = hashCode;
        return hashCode;
    }

    public boolean j(int i, h hVar, int i2, int i3) {
        return hVar.k(i2, this.f6180d, i, i3);
    }

    public boolean k(int i, byte[] bArr, int i2, int i3) {
        if (i >= 0) {
            byte[] bArr2 = this.f6180d;
            if (i <= bArr2.length - i3 && i2 >= 0 && i2 <= bArr.length - i3 && z.a(bArr2, i, bArr, i2, i3)) {
                return true;
            }
        }
        return false;
    }

    public int l() {
        return this.f6180d.length;
    }

    public h m(int i, int i2) {
        if (i >= 0) {
            byte[] bArr = this.f6180d;
            if (i2 <= bArr.length) {
                int i3 = i2 - i;
                if (i3 >= 0) {
                    if (i == 0 && i2 == bArr.length) {
                        return this;
                    }
                    byte[] bArr2 = new byte[i3];
                    System.arraycopy(bArr, i, bArr2, 0, i3);
                    return new h(bArr2);
                }
                throw new IllegalArgumentException("endIndex < beginIndex");
            }
            throw new IllegalArgumentException(c.b.a.a.a.s(c.b.a.a.a.x("endIndex > length("), this.f6180d.length, ")"));
        }
        throw new IllegalArgumentException("beginIndex < 0");
    }

    public h n() {
        int i = 0;
        while (true) {
            byte[] bArr = this.f6180d;
            if (i >= bArr.length) {
                return this;
            }
            byte b2 = bArr[i];
            if (b2 >= 65 && b2 <= 90) {
                byte[] bArr2 = (byte[]) bArr.clone();
                bArr2[i] = (byte) (b2 + 32);
                for (int i2 = i + 1; i2 < bArr2.length; i2++) {
                    byte b3 = bArr2[i2];
                    if (b3 >= 65 && b3 <= 90) {
                        bArr2[i2] = (byte) (b3 + 32);
                    }
                }
                return new h(bArr2);
            }
            i++;
        }
    }

    public byte[] o() {
        return (byte[]) this.f6180d.clone();
    }

    public String p() {
        String str = this.f6182f;
        if (str != null) {
            return str;
        }
        String str2 = new String(this.f6180d, z.f6224a);
        this.f6182f = str2;
        return str2;
    }

    public void q(e eVar) {
        byte[] bArr = this.f6180d;
        eVar.R(bArr, 0, bArr.length);
    }

    public String toString() {
        if (this.f6180d.length == 0) {
            return "[size=0]";
        }
        String p = p();
        int length = p.length();
        int i = 0;
        int i2 = 0;
        while (true) {
            if (i >= length) {
                i = p.length();
                break;
            } else if (i2 == 64) {
                break;
            } else {
                int codePointAt = p.codePointAt(i);
                if ((!Character.isISOControl(codePointAt) || codePointAt == 10 || codePointAt == 13) && codePointAt != 65533) {
                    i2++;
                    i += Character.charCount(codePointAt);
                }
            }
        }
        i = -1;
        if (i == -1) {
            if (this.f6180d.length <= 64) {
                StringBuilder x = c.b.a.a.a.x("[hex=");
                x.append(g());
                x.append("]");
                return x.toString();
            }
            StringBuilder x2 = c.b.a.a.a.x("[size=");
            x2.append(this.f6180d.length);
            x2.append(" hex=");
            x2.append(m(0, 64).g());
            x2.append("…]");
            return x2.toString();
        }
        String replace = p.substring(0, i).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r");
        if (i < p.length()) {
            StringBuilder x3 = c.b.a.a.a.x("[size=");
            x3.append(this.f6180d.length);
            x3.append(" text=");
            x3.append(replace);
            x3.append("…]");
            return x3.toString();
        }
        return c.b.a.a.a.r("[text=", replace, "]");
    }
}