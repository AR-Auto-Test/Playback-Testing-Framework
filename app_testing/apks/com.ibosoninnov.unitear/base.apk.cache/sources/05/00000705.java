package c.c.a.l;

import android.graphics.Bitmap;
import android.util.Log;
import c.c.a.l.a;
import com.google.common.primitives.UnsignedBytes;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Iterator;

/* compiled from: StandardGifDecoder.java */
/* loaded from: classes.dex */
public class e implements a {

    /* renamed from: a  reason: collision with root package name */
    public static final String f3508a = "e";

    /* renamed from: b  reason: collision with root package name */
    public int[] f3509b;

    /* renamed from: d  reason: collision with root package name */
    public final a.InterfaceC0064a f3511d;

    /* renamed from: e  reason: collision with root package name */
    public ByteBuffer f3512e;

    /* renamed from: f  reason: collision with root package name */
    public byte[] f3513f;

    /* renamed from: g  reason: collision with root package name */
    public short[] f3514g;

    /* renamed from: h  reason: collision with root package name */
    public byte[] f3515h;
    public byte[] i;
    public byte[] j;
    public int[] k;
    public int l;
    public c m;
    public Bitmap n;
    public boolean o;
    public int p;
    public int q;
    public int r;
    public int s;
    public Boolean t;

    /* renamed from: c  reason: collision with root package name */
    public final int[] f3510c = new int[256];
    public Bitmap.Config u = Bitmap.Config.ARGB_8888;

    public e(a.InterfaceC0064a interfaceC0064a, c cVar, ByteBuffer byteBuffer, int i) {
        int[] iArr;
        this.f3511d = interfaceC0064a;
        this.m = new c();
        synchronized (this) {
            if (i > 0) {
                int highestOneBit = Integer.highestOneBit(i);
                this.p = 0;
                this.m = cVar;
                this.l = -1;
                ByteBuffer asReadOnlyBuffer = byteBuffer.asReadOnlyBuffer();
                this.f3512e = asReadOnlyBuffer;
                asReadOnlyBuffer.position(0);
                this.f3512e.order(ByteOrder.LITTLE_ENDIAN);
                this.o = false;
                Iterator<b> it = cVar.f3500e.iterator();
                while (true) {
                    if (!it.hasNext()) {
                        break;
                    } else if (it.next().f3494g == 3) {
                        this.o = true;
                        break;
                    }
                }
                this.q = highestOneBit;
                int i2 = cVar.f3501f;
                this.s = i2 / highestOneBit;
                int i3 = cVar.f3502g;
                this.r = i3 / highestOneBit;
                this.j = ((c.c.a.m.x.g.b) this.f3511d).a(i2 * i3);
                a.InterfaceC0064a interfaceC0064a2 = this.f3511d;
                int i4 = this.s * this.r;
                c.c.a.m.v.c0.b bVar = ((c.c.a.m.x.g.b) interfaceC0064a2).f4035b;
                if (bVar == null) {
                    iArr = new int[i4];
                } else {
                    iArr = (int[]) bVar.d(i4, int[].class);
                }
                this.k = iArr;
            } else {
                throw new IllegalArgumentException("Sample size must be >=0, not: " + i);
            }
        }
    }

    @Override // c.c.a.l.a
    public synchronized Bitmap a() {
        if (this.m.f3498c <= 0 || this.l < 0) {
            String str = f3508a;
            if (Log.isLoggable(str, 3)) {
                Log.d(str, "Unable to decode frame, frameCount=" + this.m.f3498c + ", framePointer=" + this.l);
            }
            this.p = 1;
        }
        int i = this.p;
        if (i != 1 && i != 2) {
            this.p = 0;
            if (this.f3513f == null) {
                this.f3513f = ((c.c.a.m.x.g.b) this.f3511d).a(255);
            }
            b bVar = this.m.f3500e.get(this.l);
            int i2 = this.l - 1;
            b bVar2 = i2 >= 0 ? this.m.f3500e.get(i2) : null;
            int[] iArr = bVar.k;
            if (iArr == null) {
                iArr = this.m.f3496a;
            }
            this.f3509b = iArr;
            if (iArr == null) {
                String str2 = f3508a;
                if (Log.isLoggable(str2, 3)) {
                    Log.d(str2, "No valid color table found for frame #" + this.l);
                }
                this.p = 1;
                return null;
            }
            if (bVar.f3493f) {
                System.arraycopy(iArr, 0, this.f3510c, 0, iArr.length);
                int[] iArr2 = this.f3510c;
                this.f3509b = iArr2;
                iArr2[bVar.f3495h] = 0;
                if (bVar.f3494g == 2 && this.l == 0) {
                    this.t = Boolean.TRUE;
                }
            }
            return j(bVar, bVar2);
        }
        String str3 = f3508a;
        if (Log.isLoggable(str3, 3)) {
            Log.d(str3, "Unable to decode frame, status=" + this.p);
        }
        return null;
    }

    @Override // c.c.a.l.a
    public void b() {
        this.l = (this.l + 1) % this.m.f3498c;
    }

    @Override // c.c.a.l.a
    public int c() {
        return this.m.f3498c;
    }

    @Override // c.c.a.l.a
    public void clear() {
        c.c.a.m.v.c0.b bVar;
        c.c.a.m.v.c0.b bVar2;
        c.c.a.m.v.c0.b bVar3;
        this.m = null;
        byte[] bArr = this.j;
        if (bArr != null && (bVar3 = ((c.c.a.m.x.g.b) this.f3511d).f4035b) != null) {
            bVar3.put(bArr);
        }
        int[] iArr = this.k;
        if (iArr != null && (bVar2 = ((c.c.a.m.x.g.b) this.f3511d).f4035b) != null) {
            bVar2.put(iArr);
        }
        Bitmap bitmap = this.n;
        if (bitmap != null) {
            ((c.c.a.m.x.g.b) this.f3511d).f4034a.d(bitmap);
        }
        this.n = null;
        this.f3512e = null;
        this.t = null;
        byte[] bArr2 = this.f3513f;
        if (bArr2 == null || (bVar = ((c.c.a.m.x.g.b) this.f3511d).f4035b) == null) {
            return;
        }
        bVar.put(bArr2);
    }

    @Override // c.c.a.l.a
    public int d() {
        int i;
        c cVar = this.m;
        int i2 = cVar.f3498c;
        if (i2 <= 0 || (i = this.l) < 0) {
            return 0;
        }
        if (i < 0 || i >= i2) {
            return -1;
        }
        return cVar.f3500e.get(i).i;
    }

    @Override // c.c.a.l.a
    public ByteBuffer e() {
        return this.f3512e;
    }

    @Override // c.c.a.l.a
    public int f() {
        return this.l;
    }

    @Override // c.c.a.l.a
    public int g() {
        return (this.k.length * 4) + this.f3512e.limit() + this.j.length;
    }

    public final Bitmap h() {
        Boolean bool = this.t;
        Bitmap.Config config = (bool == null || bool.booleanValue()) ? Bitmap.Config.ARGB_8888 : this.u;
        Bitmap c2 = ((c.c.a.m.x.g.b) this.f3511d).f4034a.c(this.s, this.r, config);
        c2.setHasAlpha(true);
        return c2;
    }

    public void i(Bitmap.Config config) {
        if (config != Bitmap.Config.ARGB_8888 && config != Bitmap.Config.RGB_565) {
            throw new IllegalArgumentException("Unsupported format: " + config + ", must be one of " + Bitmap.Config.ARGB_8888 + " or " + Bitmap.Config.RGB_565);
        }
        this.u = config;
    }

    /* JADX WARN: Code restructure failed: missing block: B:25:0x0045, code lost:
        if (r3.j == r36.f3495h) goto L33;
     */
    /* JADX WARN: Removed duplicated region for block: B:29:0x005e  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Bitmap j(b bVar, b bVar2) {
        int[] iArr;
        byte b2;
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        int i10;
        int i11;
        int i12;
        int i13;
        short s;
        int i14;
        short s2;
        short s3;
        int i15;
        Bitmap bitmap;
        int i16;
        int i17;
        int i18;
        int[] iArr2 = this.k;
        byte b3 = 0;
        if (bVar2 == null) {
            Bitmap bitmap2 = this.n;
            if (bitmap2 != null) {
                ((c.c.a.m.x.g.b) this.f3511d).f4034a.d(bitmap2);
            }
            this.n = null;
            Arrays.fill(iArr2, 0);
        }
        if (bVar2 != null && bVar2.f3494g == 3 && this.n == null) {
            Arrays.fill(iArr2, 0);
        }
        if (bVar2 != null && (i15 = bVar2.f3494g) > 0) {
            if (i15 == 2) {
                if (!bVar.f3493f) {
                    c cVar = this.m;
                    i16 = cVar.k;
                    if (bVar.k != null) {
                    }
                    int i19 = bVar2.f3491d;
                    int i20 = this.q;
                    int i21 = i19 / i20;
                    int i22 = bVar2.f3489b / i20;
                    int i23 = bVar2.f3490c / i20;
                    int i24 = bVar2.f3488a / i20;
                    int i25 = this.s;
                    i17 = (i22 * i25) + i24;
                    i18 = (i21 * i25) + i17;
                    while (i17 < i18) {
                        int i26 = i17 + i23;
                        for (int i27 = i17; i27 < i26; i27++) {
                            iArr2[i27] = i16;
                        }
                        i17 += this.s;
                    }
                }
                i16 = 0;
                int i192 = bVar2.f3491d;
                int i202 = this.q;
                int i212 = i192 / i202;
                int i222 = bVar2.f3489b / i202;
                int i232 = bVar2.f3490c / i202;
                int i242 = bVar2.f3488a / i202;
                int i252 = this.s;
                i17 = (i222 * i252) + i242;
                i18 = (i212 * i252) + i17;
                while (i17 < i18) {
                }
            } else if (i15 == 3 && (bitmap = this.n) != null) {
                int i28 = this.s;
                bitmap.getPixels(iArr2, 0, i28, 0, 0, i28, this.r);
            }
        }
        this.f3512e.position(bVar.j);
        int i29 = bVar.f3490c * bVar.f3491d;
        byte[] bArr = this.j;
        if (bArr == null || bArr.length < i29) {
            this.j = ((c.c.a.m.x.g.b) this.f3511d).a(i29);
        }
        byte[] bArr2 = this.j;
        if (this.f3514g == null) {
            this.f3514g = new short[4096];
        }
        short[] sArr = this.f3514g;
        if (this.f3515h == null) {
            this.f3515h = new byte[4096];
        }
        byte[] bArr3 = this.f3515h;
        if (this.i == null) {
            this.i = new byte[4097];
        }
        byte[] bArr4 = this.i;
        int i30 = this.f3512e.get() & UnsignedBytes.MAX_VALUE;
        int i31 = 1 << i30;
        int i32 = i31 + 1;
        int i33 = i31 + 2;
        int i34 = i30 + 1;
        int i35 = (1 << i34) - 1;
        for (int i36 = 0; i36 < i31; i36++) {
            sArr[i36] = 0;
            bArr3[i36] = (byte) i36;
        }
        byte[] bArr5 = this.f3513f;
        e eVar = this;
        int i37 = i34;
        int i38 = 0;
        int i39 = 0;
        int i40 = 0;
        int i41 = 0;
        int i42 = 0;
        int i43 = 0;
        short s4 = 0;
        int i44 = 0;
        int i45 = i33;
        int i46 = i35;
        short s5 = -1;
        while (true) {
            if (i38 >= i29) {
                iArr = iArr2;
                b2 = b3;
                i = i39;
                break;
            }
            if (i40 == 0) {
                int i47 = eVar.f3512e.get() & UnsignedBytes.MAX_VALUE;
                if (i47 <= 0) {
                    i12 = i34;
                    i13 = i38;
                    iArr = iArr2;
                    s = s5;
                } else {
                    i12 = i34;
                    ByteBuffer byteBuffer = eVar.f3512e;
                    i13 = i38;
                    s = s5;
                    iArr = iArr2;
                    byteBuffer.get(eVar.f3513f, 0, Math.min(i47, byteBuffer.remaining()));
                }
                if (i47 <= 0) {
                    eVar.p = 3;
                    i = i39;
                    b2 = 0;
                    break;
                }
                i40 = i47;
                i41 = 0;
            } else {
                i12 = i34;
                i13 = i38;
                iArr = iArr2;
                s = s5;
            }
            i43 += (bArr5[i41] & UnsignedBytes.MAX_VALUE) << i42;
            i41++;
            i40--;
            int i48 = i42 + 8;
            int i49 = i45;
            int i50 = i37;
            i38 = i13;
            s5 = s;
            byte[] bArr6 = bArr5;
            short s6 = s4;
            while (true) {
                if (i48 < i50) {
                    s4 = s6;
                    eVar = this;
                    break;
                }
                e eVar2 = eVar;
                int i51 = i43 & i46;
                i43 >>= i50;
                i48 -= i50;
                if (i51 == i31) {
                    i14 = i48;
                    i49 = i33;
                    i46 = i35;
                    eVar = eVar2;
                    i50 = i12;
                    s5 = -1;
                    s2 = s6;
                } else if (i51 == i32) {
                    s4 = s6;
                    eVar = eVar2;
                    break;
                } else {
                    i14 = i48;
                    if (s5 == -1) {
                        bArr2[i39] = bArr3[i51];
                        i39++;
                        i38++;
                        s6 = i51;
                        s5 = s6;
                        i48 = i14;
                        eVar = this;
                    } else {
                        if (i51 >= i49) {
                            bArr4[i44] = (byte) s6;
                            i44++;
                            s3 = s5;
                        } else {
                            s3 = i51;
                        }
                        while (s3 >= i31) {
                            bArr4[i44] = bArr3[s3];
                            i44++;
                            s3 = sArr[s3];
                        }
                        int i52 = bArr3[s3] & 255;
                        byte b4 = i52 == 1 ? (byte) 1 : (byte) 0;
                        bArr2[i39] = b4;
                        while (true) {
                            i39++;
                            i38++;
                            if (i44 <= 0) {
                                break;
                            }
                            i44--;
                            bArr2[i39] = bArr4[i44];
                        }
                        s2 = i52 == 1 ? 1 : 0;
                        if (i49 < 4096) {
                            sArr[i49] = s5;
                            bArr3[i49] = b4;
                            i49++;
                            if ((i49 & i46) == 0 && i49 < 4096) {
                                i50++;
                                i46 += i49;
                            }
                        }
                        s5 = i51;
                        eVar = this;
                    }
                }
                s6 = s2;
                i48 = i14;
            }
            i37 = i50;
            i45 = i49;
            bArr5 = bArr6;
            i34 = i12;
            b3 = 0;
            i42 = i48;
            iArr2 = iArr;
        }
        Arrays.fill(bArr2, i, i29, b2);
        if (!bVar.f3492e && this.q == 1) {
            int[] iArr3 = this.k;
            int i53 = bVar.f3491d;
            int i54 = bVar.f3489b;
            int i55 = bVar.f3490c;
            int i56 = bVar.f3488a;
            byte b5 = this.l == 0 ? (byte) 1 : b2;
            int i57 = this.s;
            byte[] bArr7 = this.j;
            int[] iArr4 = this.f3509b;
            int i58 = -1;
            for (int i59 = b2; i59 < i53; i59++) {
                int i60 = (i59 + i54) * i57;
                int i61 = i60 + i56;
                int i62 = i61 + i55;
                int i63 = i60 + i57;
                if (i63 < i62) {
                    i62 = i63;
                }
                int i64 = bVar.f3490c * i59;
                while (i61 < i62) {
                    int i65 = i53;
                    int i66 = bArr7[i64];
                    int i67 = i54;
                    int i68 = i66 & 255;
                    if (i68 != i58) {
                        int i69 = iArr4[i68];
                        if (i69 != 0) {
                            iArr3[i61] = i69;
                        } else {
                            i58 = i66;
                        }
                    }
                    i64++;
                    i61++;
                    i53 = i65;
                    i54 = i67;
                }
            }
            Boolean bool = this.t;
            this.t = Boolean.valueOf((bool != null && bool.booleanValue()) || !(this.t != null || b5 == 0 || i58 == -1));
        } else {
            int[] iArr5 = this.k;
            int i70 = bVar.f3491d;
            int i71 = this.q;
            int i72 = i70 / i71;
            int i73 = bVar.f3489b / i71;
            int i74 = bVar.f3490c / i71;
            int i75 = bVar.f3488a / i71;
            boolean z = this.l == 0;
            int i76 = this.s;
            int i77 = this.r;
            byte[] bArr8 = this.j;
            int[] iArr6 = this.f3509b;
            Boolean bool2 = this.t;
            int i78 = 8;
            int i79 = 0;
            int i80 = 0;
            int i81 = 1;
            while (i79 < i72) {
                Boolean bool3 = bool2;
                if (bVar.f3492e) {
                    if (i80 >= i72) {
                        int i82 = i81 + 1;
                        i2 = i72;
                        if (i82 == 2) {
                            i80 = 4;
                            i81 = i82;
                        } else if (i82 != 3) {
                            i81 = i82;
                            if (i82 == 4) {
                                i80 = 1;
                                i78 = 2;
                            }
                        } else {
                            i78 = 4;
                            i81 = i82;
                            i80 = 2;
                        }
                    } else {
                        i2 = i72;
                    }
                    i3 = i80 + i78;
                } else {
                    i2 = i72;
                    i3 = i80;
                    i80 = i79;
                }
                int i83 = i80 + i73;
                boolean z2 = i71 == 1;
                if (i83 < i77) {
                    int i84 = i83 * i76;
                    int i85 = i84 + i75;
                    i4 = i3;
                    int i86 = i85 + i74;
                    int i87 = i84 + i76;
                    if (i87 < i86) {
                        i86 = i87;
                    }
                    i5 = i73;
                    int i88 = i79 * i71 * bVar.f3490c;
                    if (z2) {
                        bool2 = bool3;
                        int i89 = i85;
                        while (true) {
                            i6 = i74;
                            if (i89 >= i86) {
                                break;
                            }
                            int i90 = iArr6[bArr8[i88] & 255];
                            if (i90 != 0) {
                                iArr5[i89] = i90;
                            } else if (z && bool2 == null) {
                                bool2 = Boolean.TRUE;
                            }
                            i88 += i71;
                            i89++;
                            i74 = i6;
                        }
                    } else {
                        i6 = i74;
                        int i91 = ((i86 - i85) * i71) + i88;
                        bool2 = bool3;
                        int i92 = i85;
                        while (i92 < i86) {
                            int i93 = i86;
                            int i94 = bVar.f3490c;
                            int i95 = i75;
                            int i96 = i76;
                            int i97 = i88;
                            int i98 = 0;
                            int i99 = 0;
                            int i100 = 0;
                            int i101 = 0;
                            int i102 = 0;
                            while (true) {
                                if (i97 >= this.q + i88) {
                                    i10 = i77;
                                    break;
                                }
                                byte[] bArr9 = this.j;
                                i10 = i77;
                                if (i97 >= bArr9.length || i97 >= i91) {
                                    break;
                                }
                                int i103 = this.f3509b[bArr9[i97] & 255];
                                if (i103 != 0) {
                                    i98 += (i103 >> 24) & 255;
                                    i99 += (i103 >> 16) & 255;
                                    i100 += (i103 >> 8) & 255;
                                    i101 += i103 & 255;
                                    i102++;
                                }
                                i97++;
                                i77 = i10;
                            }
                            int i104 = i94 + i88;
                            for (int i105 = i104; i105 < this.q + i104; i105++) {
                                byte[] bArr10 = this.j;
                                if (i105 >= bArr10.length || i105 >= i91) {
                                    break;
                                }
                                int i106 = this.f3509b[bArr10[i105] & 255];
                                if (i106 != 0) {
                                    i98 += (i106 >> 24) & 255;
                                    i99 += (i106 >> 16) & 255;
                                    i100 += (i106 >> 8) & 255;
                                    i101 += i106 & 255;
                                    i102++;
                                }
                            }
                            int i107 = i102 == 0 ? 0 : ((i98 / i102) << 24) | ((i99 / i102) << 16) | ((i100 / i102) << 8) | (i101 / i102);
                            if (i107 != 0) {
                                iArr5[i92] = i107;
                            } else if (z && bool2 == null) {
                                bool2 = Boolean.TRUE;
                            }
                            i88 += i71;
                            i92++;
                            i86 = i93;
                            i75 = i95;
                            i76 = i96;
                            i77 = i10;
                        }
                    }
                    i7 = i75;
                    i8 = i76;
                    i9 = i77;
                } else {
                    i4 = i3;
                    i5 = i73;
                    i6 = i74;
                    i7 = i75;
                    i8 = i76;
                    i9 = i77;
                    bool2 = bool3;
                }
                i79++;
                i72 = i2;
                i80 = i4;
                i73 = i5;
                i74 = i6;
                i75 = i7;
                i76 = i8;
                i77 = i9;
            }
            Boolean bool4 = bool2;
            if (this.t == null) {
                this.t = Boolean.valueOf(bool4 == null ? false : bool4.booleanValue());
            }
        }
        if (this.o && ((i11 = bVar.f3494g) == 0 || i11 == 1)) {
            if (this.n == null) {
                this.n = h();
            }
            Bitmap bitmap3 = this.n;
            int i108 = this.s;
            bitmap3.setPixels(iArr, 0, i108, 0, 0, i108, this.r);
        }
        Bitmap h2 = h();
        int i109 = this.s;
        h2.setPixels(iArr, 0, i109, 0, 0, i109, this.r);
        return h2;
    }
}