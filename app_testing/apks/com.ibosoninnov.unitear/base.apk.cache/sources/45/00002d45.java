package f.g0.k;

import com.google.common.primitives.UnsignedBytes;
import f.g0.c;
import g.l;
import g.o;
import g.s;
import java.io.InputStream;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: PublicSuffixDatabase.java */
/* loaded from: classes2.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public static final byte[] f6034a = {42};

    /* renamed from: b  reason: collision with root package name */
    public static final String[] f6035b = new String[0];

    /* renamed from: c  reason: collision with root package name */
    public static final String[] f6036c = {"*"};

    /* renamed from: d  reason: collision with root package name */
    public static final a f6037d = new a();

    /* renamed from: e  reason: collision with root package name */
    public final AtomicBoolean f6038e = new AtomicBoolean(false);

    /* renamed from: f  reason: collision with root package name */
    public final CountDownLatch f6039f = new CountDownLatch(1);

    /* renamed from: g  reason: collision with root package name */
    public byte[] f6040g;

    /* renamed from: h  reason: collision with root package name */
    public byte[] f6041h;

    public static String a(byte[] bArr, byte[][] bArr2, int i) {
        int i2;
        boolean z;
        int i3;
        int i4;
        int length = bArr.length;
        int i5 = 0;
        while (i5 < length) {
            int i6 = (i5 + length) / 2;
            while (i6 > -1 && bArr[i6] != 10) {
                i6--;
            }
            int i7 = i6 + 1;
            int i8 = 1;
            while (true) {
                i2 = i7 + i8;
                if (bArr[i2] == 10) {
                    break;
                }
                i8++;
            }
            int i9 = i2 - i7;
            int i10 = i;
            boolean z2 = false;
            int i11 = 0;
            int i12 = 0;
            while (true) {
                if (z2) {
                    i3 = 46;
                    z = false;
                } else {
                    z = z2;
                    i3 = bArr2[i10][i11] & UnsignedBytes.MAX_VALUE;
                }
                i4 = i3 - (bArr[i7 + i12] & UnsignedBytes.MAX_VALUE);
                if (i4 == 0) {
                    i12++;
                    i11++;
                    if (i12 == i9) {
                        break;
                    } else if (bArr2[i10].length != i11) {
                        z2 = z;
                    } else if (i10 == bArr2.length - 1) {
                        break;
                    } else {
                        i10++;
                        i11 = -1;
                        z2 = true;
                    }
                } else {
                    break;
                }
            }
            if (i4 >= 0) {
                if (i4 <= 0) {
                    int i13 = i9 - i12;
                    int length2 = bArr2[i10].length - i11;
                    while (true) {
                        i10++;
                        if (i10 >= bArr2.length) {
                            break;
                        }
                        length2 += bArr2[i10].length;
                    }
                    if (length2 >= i13) {
                        if (length2 <= i13) {
                            return new String(bArr, i7, i9, c.i);
                        }
                    }
                }
                i5 = i2 + 1;
            }
            length = i7 - 1;
        }
        return null;
    }

    public final void b() {
        InputStream resourceAsStream = a.class.getResourceAsStream("publicsuffixes.gz");
        if (resourceAsStream == null) {
            return;
        }
        s sVar = new s(new l(o.c(resourceAsStream)));
        try {
            byte[] bArr = new byte[sVar.readInt()];
            sVar.C(bArr);
            byte[] bArr2 = new byte[sVar.readInt()];
            sVar.C(bArr2);
            synchronized (this) {
                this.f6040g = bArr;
                this.f6041h = bArr2;
            }
            this.f6039f.countDown();
        } finally {
            c.f(sVar);
        }
    }
}