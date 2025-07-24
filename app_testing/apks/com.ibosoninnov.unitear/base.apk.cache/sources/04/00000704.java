package c.c.a.l;

import android.util.Log;
import com.google.common.primitives.UnsignedBytes;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.util.Objects;

/* compiled from: GifHeaderParser.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: b  reason: collision with root package name */
    public ByteBuffer f3505b;

    /* renamed from: c  reason: collision with root package name */
    public c f3506c;

    /* renamed from: a  reason: collision with root package name */
    public final byte[] f3504a = new byte[256];

    /* renamed from: d  reason: collision with root package name */
    public int f3507d = 0;

    public final boolean a() {
        return this.f3506c.f3497b != 0;
    }

    public c b() {
        if (this.f3505b != null) {
            if (a()) {
                return this.f3506c;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 6; i++) {
                sb.append((char) c());
            }
            if (!sb.toString().startsWith("GIF")) {
                this.f3506c.f3497b = 1;
            } else {
                this.f3506c.f3501f = f();
                this.f3506c.f3502g = f();
                int c2 = c();
                c cVar = this.f3506c;
                cVar.f3503h = (c2 & 128) != 0;
                cVar.i = (int) Math.pow(2.0d, (c2 & 7) + 1);
                this.f3506c.j = c();
                c cVar2 = this.f3506c;
                c();
                Objects.requireNonNull(cVar2);
                if (this.f3506c.f3503h && !a()) {
                    c cVar3 = this.f3506c;
                    cVar3.f3496a = e(cVar3.i);
                    c cVar4 = this.f3506c;
                    cVar4.k = cVar4.f3496a[cVar4.j];
                }
            }
            if (!a()) {
                boolean z = false;
                while (!z && !a() && this.f3506c.f3498c <= Integer.MAX_VALUE) {
                    int c3 = c();
                    if (c3 == 33) {
                        int c4 = c();
                        if (c4 == 1) {
                            g();
                        } else if (c4 == 249) {
                            this.f3506c.f3499d = new b();
                            c();
                            int c5 = c();
                            b bVar = this.f3506c.f3499d;
                            int i2 = (c5 & 28) >> 2;
                            bVar.f3494g = i2;
                            if (i2 == 0) {
                                bVar.f3494g = 1;
                            }
                            bVar.f3493f = (c5 & 1) != 0;
                            int f2 = f();
                            if (f2 < 2) {
                                f2 = 10;
                            }
                            b bVar2 = this.f3506c.f3499d;
                            bVar2.i = f2 * 10;
                            bVar2.f3495h = c();
                            c();
                        } else if (c4 == 254) {
                            g();
                        } else if (c4 != 255) {
                            g();
                        } else {
                            d();
                            StringBuilder sb2 = new StringBuilder();
                            for (int i3 = 0; i3 < 11; i3++) {
                                sb2.append((char) this.f3504a[i3]);
                            }
                            if (sb2.toString().equals("NETSCAPE2.0")) {
                                do {
                                    d();
                                    byte[] bArr = this.f3504a;
                                    if (bArr[0] == 1) {
                                        byte b2 = bArr[1];
                                        byte b3 = bArr[2];
                                        Objects.requireNonNull(this.f3506c);
                                    }
                                    if (this.f3507d > 0) {
                                    }
                                } while (!a());
                            } else {
                                g();
                            }
                        }
                    } else if (c3 == 44) {
                        c cVar5 = this.f3506c;
                        if (cVar5.f3499d == null) {
                            cVar5.f3499d = new b();
                        }
                        cVar5.f3499d.f3488a = f();
                        this.f3506c.f3499d.f3489b = f();
                        this.f3506c.f3499d.f3490c = f();
                        this.f3506c.f3499d.f3491d = f();
                        int c6 = c();
                        boolean z2 = (c6 & 128) != 0;
                        int pow = (int) Math.pow(2.0d, (c6 & 7) + 1);
                        b bVar3 = this.f3506c.f3499d;
                        bVar3.f3492e = (c6 & 64) != 0;
                        if (z2) {
                            bVar3.k = e(pow);
                        } else {
                            bVar3.k = null;
                        }
                        this.f3506c.f3499d.j = this.f3505b.position();
                        c();
                        g();
                        if (!a()) {
                            c cVar6 = this.f3506c;
                            cVar6.f3498c++;
                            cVar6.f3500e.add(cVar6.f3499d);
                        }
                    } else if (c3 != 59) {
                        this.f3506c.f3497b = 1;
                    } else {
                        z = true;
                    }
                }
                c cVar7 = this.f3506c;
                if (cVar7.f3498c < 0) {
                    cVar7.f3497b = 1;
                }
            }
            return this.f3506c;
        }
        throw new IllegalStateException("You must call setData() before parseHeader()");
    }

    public final int c() {
        try {
            return this.f3505b.get() & UnsignedBytes.MAX_VALUE;
        } catch (Exception unused) {
            this.f3506c.f3497b = 1;
            return 0;
        }
    }

    public final void d() {
        int c2 = c();
        this.f3507d = c2;
        if (c2 <= 0) {
            return;
        }
        int i = 0;
        int i2 = 0;
        while (true) {
            try {
                i2 = this.f3507d;
                if (i >= i2) {
                    return;
                }
                i2 -= i;
                this.f3505b.get(this.f3504a, i, i2);
                i += i2;
            } catch (Exception e2) {
                if (Log.isLoggable("GifHeaderParser", 3)) {
                    StringBuilder z = c.b.a.a.a.z("Error Reading Block n: ", i, " count: ", i2, " blockSize: ");
                    z.append(this.f3507d);
                    Log.d("GifHeaderParser", z.toString(), e2);
                }
                this.f3506c.f3497b = 1;
                return;
            }
        }
    }

    public final int[] e(int i) {
        byte[] bArr = new byte[i * 3];
        int[] iArr = null;
        try {
            this.f3505b.get(bArr);
            iArr = new int[256];
            int i2 = 0;
            int i3 = 0;
            while (i2 < i) {
                int i4 = i3 + 1;
                int i5 = i4 + 1;
                int i6 = i5 + 1;
                int i7 = i2 + 1;
                iArr[i2] = ((bArr[i3] & UnsignedBytes.MAX_VALUE) << 16) | (-16777216) | ((bArr[i4] & UnsignedBytes.MAX_VALUE) << 8) | (bArr[i5] & UnsignedBytes.MAX_VALUE);
                i3 = i6;
                i2 = i7;
            }
        } catch (BufferUnderflowException e2) {
            if (Log.isLoggable("GifHeaderParser", 3)) {
                Log.d("GifHeaderParser", "Format Error Reading Color Table", e2);
            }
            this.f3506c.f3497b = 1;
        }
        return iArr;
    }

    public final int f() {
        return this.f3505b.getShort();
    }

    public final void g() {
        int c2;
        do {
            c2 = c();
            this.f3505b.position(Math.min(this.f3505b.position() + c2, this.f3505b.limit()));
        } while (c2 > 0);
    }
}