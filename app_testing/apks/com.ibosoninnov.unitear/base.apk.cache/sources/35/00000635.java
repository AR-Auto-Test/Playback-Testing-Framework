package c.a.a.b0.h0;

import c.a.a.b0.h0.c;
import g.e;
import g.g;
import g.h;
import java.io.EOFException;
import java.util.Objects;

/* compiled from: JsonUtf8Reader.java */
/* loaded from: classes.dex */
public final class d extends c {

    /* renamed from: g  reason: collision with root package name */
    public static final h f2987g = h.e("'\\");

    /* renamed from: h  reason: collision with root package name */
    public static final h f2988h = h.e("\"\\");
    public static final h i = h.e("{}[]:, \n\t\r\f/\\;#=");
    public static final h j = h.e("\n\r");
    public static final h k = h.e("*/");
    public final g l;
    public final e m;
    public int n = 0;
    public long o;
    public int p;
    public String q;

    public d(g gVar) {
        Objects.requireNonNull(gVar, "source == null");
        this.l = gVar;
        this.m = gVar.e();
        N(6);
    }

    @Override // c.a.a.b0.h0.c
    public void B() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 3) {
            N(1);
            this.f2977f[this.f2974c - 1] = 0;
            this.n = 0;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Expected BEGIN_ARRAY but was ");
        x.append(M());
        x.append(" at path ");
        x.append(F());
        throw new a(x.toString());
    }

    @Override // c.a.a.b0.h0.c
    public void C() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 1) {
            N(3);
            this.n = 0;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Expected BEGIN_OBJECT but was ");
        x.append(M());
        x.append(" at path ");
        x.append(F());
        throw new a(x.toString());
    }

    @Override // c.a.a.b0.h0.c
    public void D() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 4) {
            int i3 = this.f2974c - 1;
            this.f2974c = i3;
            int[] iArr = this.f2977f;
            int i4 = i3 - 1;
            iArr[i4] = iArr[i4] + 1;
            this.n = 0;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Expected END_ARRAY but was ");
        x.append(M());
        x.append(" at path ");
        x.append(F());
        throw new a(x.toString());
    }

    @Override // c.a.a.b0.h0.c
    public void E() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 2) {
            int i3 = this.f2974c - 1;
            this.f2974c = i3;
            this.f2976e[i3] = null;
            int[] iArr = this.f2977f;
            int i4 = i3 - 1;
            iArr[i4] = iArr[i4] + 1;
            this.n = 0;
            return;
        }
        StringBuilder x = c.b.a.a.a.x("Expected END_OBJECT but was ");
        x.append(M());
        x.append(" at path ");
        x.append(F());
        throw new a(x.toString());
    }

    @Override // c.a.a.b0.h0.c
    public boolean G() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        return (i2 == 2 || i2 == 4 || i2 == 18) ? false : true;
    }

    @Override // c.a.a.b0.h0.c
    public boolean H() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 5) {
            this.n = 0;
            int[] iArr = this.f2977f;
            int i3 = this.f2974c - 1;
            iArr[i3] = iArr[i3] + 1;
            return true;
        } else if (i2 == 6) {
            this.n = 0;
            int[] iArr2 = this.f2977f;
            int i4 = this.f2974c - 1;
            iArr2[i4] = iArr2[i4] + 1;
            return false;
        } else {
            StringBuilder x = c.b.a.a.a.x("Expected a boolean but was ");
            x.append(M());
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
    }

    @Override // c.a.a.b0.h0.c
    public double I() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 16) {
            this.n = 0;
            int[] iArr = this.f2977f;
            int i3 = this.f2974c - 1;
            iArr[i3] = iArr[i3] + 1;
            return this.o;
        }
        if (i2 == 17) {
            this.q = this.m.L(this.p);
        } else if (i2 == 9) {
            this.q = X(f2988h);
        } else if (i2 == 8) {
            this.q = X(f2987g);
        } else if (i2 == 10) {
            this.q = Y();
        } else if (i2 != 11) {
            StringBuilder x = c.b.a.a.a.x("Expected a double but was ");
            x.append(M());
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
        this.n = 11;
        try {
            double parseDouble = Double.parseDouble(this.q);
            if (!Double.isNaN(parseDouble) && !Double.isInfinite(parseDouble)) {
                this.q = null;
                this.n = 0;
                int[] iArr2 = this.f2977f;
                int i4 = this.f2974c - 1;
                iArr2[i4] = iArr2[i4] + 1;
                return parseDouble;
            }
            throw new b("JSON forbids NaN and infinities: " + parseDouble + " at path " + F());
        } catch (NumberFormatException unused) {
            StringBuilder x2 = c.b.a.a.a.x("Expected a double but was ");
            x2.append(this.q);
            x2.append(" at path ");
            x2.append(F());
            throw new a(x2.toString());
        }
    }

    @Override // c.a.a.b0.h0.c
    public int J() {
        String X;
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 16) {
            long j2 = this.o;
            int i3 = (int) j2;
            if (j2 == i3) {
                this.n = 0;
                int[] iArr = this.f2977f;
                int i4 = this.f2974c - 1;
                iArr[i4] = iArr[i4] + 1;
                return i3;
            }
            StringBuilder x = c.b.a.a.a.x("Expected an int but was ");
            x.append(this.o);
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
        if (i2 == 17) {
            this.q = this.m.L(this.p);
        } else if (i2 == 9 || i2 == 8) {
            if (i2 == 9) {
                X = X(f2988h);
            } else {
                X = X(f2987g);
            }
            this.q = X;
            try {
                int parseInt = Integer.parseInt(X);
                this.n = 0;
                int[] iArr2 = this.f2977f;
                int i5 = this.f2974c - 1;
                iArr2[i5] = iArr2[i5] + 1;
                return parseInt;
            } catch (NumberFormatException unused) {
            }
        } else if (i2 != 11) {
            StringBuilder x2 = c.b.a.a.a.x("Expected an int but was ");
            x2.append(M());
            x2.append(" at path ");
            x2.append(F());
            throw new a(x2.toString());
        }
        this.n = 11;
        try {
            double parseDouble = Double.parseDouble(this.q);
            int i6 = (int) parseDouble;
            if (i6 == parseDouble) {
                this.q = null;
                this.n = 0;
                int[] iArr3 = this.f2977f;
                int i7 = this.f2974c - 1;
                iArr3[i7] = iArr3[i7] + 1;
                return i6;
            }
            StringBuilder x3 = c.b.a.a.a.x("Expected an int but was ");
            x3.append(this.q);
            x3.append(" at path ");
            x3.append(F());
            throw new a(x3.toString());
        } catch (NumberFormatException unused2) {
            StringBuilder x4 = c.b.a.a.a.x("Expected an int but was ");
            x4.append(this.q);
            x4.append(" at path ");
            x4.append(F());
            throw new a(x4.toString());
        }
    }

    @Override // c.a.a.b0.h0.c
    public String K() {
        String str;
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 14) {
            str = Y();
        } else if (i2 == 13) {
            str = X(f2988h);
        } else if (i2 == 12) {
            str = X(f2987g);
        } else if (i2 == 15) {
            str = this.q;
        } else {
            StringBuilder x = c.b.a.a.a.x("Expected a name but was ");
            x.append(M());
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
        this.n = 0;
        this.f2976e[this.f2974c - 1] = str;
        return str;
    }

    @Override // c.a.a.b0.h0.c
    public String L() {
        String L;
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 10) {
            L = Y();
        } else if (i2 == 9) {
            L = X(f2988h);
        } else if (i2 == 8) {
            L = X(f2987g);
        } else if (i2 == 11) {
            L = this.q;
            this.q = null;
        } else if (i2 == 16) {
            L = Long.toString(this.o);
        } else if (i2 == 17) {
            L = this.m.L(this.p);
        } else {
            StringBuilder x = c.b.a.a.a.x("Expected a string but was ");
            x.append(M());
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
        this.n = 0;
        int[] iArr = this.f2977f;
        int i3 = this.f2974c - 1;
        iArr[i3] = iArr[i3] + 1;
        return L;
    }

    @Override // c.a.a.b0.h0.c
    public c.b M() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        switch (i2) {
            case 1:
                return c.b.BEGIN_OBJECT;
            case 2:
                return c.b.END_OBJECT;
            case 3:
                return c.b.BEGIN_ARRAY;
            case 4:
                return c.b.END_ARRAY;
            case 5:
            case 6:
                return c.b.BOOLEAN;
            case 7:
                return c.b.NULL;
            case 8:
            case 9:
            case 10:
            case 11:
                return c.b.STRING;
            case 12:
            case 13:
            case 14:
            case 15:
                return c.b.NAME;
            case 16:
            case 17:
                return c.b.NUMBER;
            case 18:
                return c.b.END_DOCUMENT;
            default:
                throw new AssertionError();
        }
    }

    @Override // c.a.a.b0.h0.c
    public int O(c.a aVar) {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 < 12 || i2 > 15) {
            return -1;
        }
        if (i2 == 15) {
            return U(this.q, aVar);
        }
        int A = this.l.A(aVar.f2979b);
        if (A != -1) {
            this.n = 0;
            this.f2976e[this.f2974c - 1] = aVar.f2978a[A];
            return A;
        }
        String str = this.f2976e[this.f2974c - 1];
        String K = K();
        int U = U(K, aVar);
        if (U == -1) {
            this.n = 15;
            this.q = K;
            this.f2976e[this.f2974c - 1] = str;
        }
        return U;
    }

    @Override // c.a.a.b0.h0.c
    public void P() {
        int i2 = this.n;
        if (i2 == 0) {
            i2 = T();
        }
        if (i2 == 14) {
            b0();
        } else if (i2 == 13) {
            a0(f2988h);
        } else if (i2 == 12) {
            a0(f2987g);
        } else if (i2 != 15) {
            StringBuilder x = c.b.a.a.a.x("Expected a name but was ");
            x.append(M());
            x.append(" at path ");
            x.append(F());
            throw new a(x.toString());
        }
        this.n = 0;
        this.f2976e[this.f2974c - 1] = "null";
    }

    @Override // c.a.a.b0.h0.c
    public void Q() {
        int i2 = 0;
        do {
            int i3 = this.n;
            if (i3 == 0) {
                i3 = T();
            }
            if (i3 == 3) {
                N(1);
            } else if (i3 == 1) {
                N(3);
            } else {
                if (i3 == 4) {
                    i2--;
                    if (i2 >= 0) {
                        this.f2974c--;
                    } else {
                        StringBuilder x = c.b.a.a.a.x("Expected a value but was ");
                        x.append(M());
                        x.append(" at path ");
                        x.append(F());
                        throw new a(x.toString());
                    }
                } else if (i3 == 2) {
                    i2--;
                    if (i2 >= 0) {
                        this.f2974c--;
                    } else {
                        StringBuilder x2 = c.b.a.a.a.x("Expected a value but was ");
                        x2.append(M());
                        x2.append(" at path ");
                        x2.append(F());
                        throw new a(x2.toString());
                    }
                } else if (i3 == 14 || i3 == 10) {
                    b0();
                } else if (i3 == 9 || i3 == 13) {
                    a0(f2988h);
                } else if (i3 == 8 || i3 == 12) {
                    a0(f2987g);
                } else if (i3 == 17) {
                    this.m.c(this.p);
                } else if (i3 == 18) {
                    StringBuilder x3 = c.b.a.a.a.x("Expected a value but was ");
                    x3.append(M());
                    x3.append(" at path ");
                    x3.append(F());
                    throw new a(x3.toString());
                }
                this.n = 0;
            }
            i2++;
            this.n = 0;
        } while (i2 != 0);
        int[] iArr = this.f2977f;
        int i4 = this.f2974c;
        int i5 = i4 - 1;
        iArr[i5] = iArr[i5] + 1;
        this.f2976e[i4 - 1] = "null";
    }

    public final void S() {
        R("Use JsonReader.setLenient(true) to accept malformed JSON");
        throw null;
    }

    /* JADX WARN: Code restructure failed: missing block: B:134:0x01ab, code lost:
        if (V(r2) != false) goto L66;
     */
    /* JADX WARN: Code restructure failed: missing block: B:135:0x01ad, code lost:
        r2 = 2;
     */
    /* JADX WARN: Code restructure failed: missing block: B:136:0x01ae, code lost:
        if (r5 != r2) goto L101;
     */
    /* JADX WARN: Code restructure failed: missing block: B:137:0x01b0, code lost:
        if (r6 == false) goto L100;
     */
    /* JADX WARN: Code restructure failed: missing block: B:139:0x01b6, code lost:
        if (r7 != Long.MIN_VALUE) goto L94;
     */
    /* JADX WARN: Code restructure failed: missing block: B:140:0x01b8, code lost:
        if (r9 == false) goto L100;
     */
    /* JADX WARN: Code restructure failed: missing block: B:142:0x01be, code lost:
        if (r7 != 0) goto L97;
     */
    /* JADX WARN: Code restructure failed: missing block: B:143:0x01c0, code lost:
        if (r9 != false) goto L100;
     */
    /* JADX WARN: Code restructure failed: missing block: B:144:0x01c2, code lost:
        if (r9 == false) goto L98;
     */
    /* JADX WARN: Code restructure failed: missing block: B:146:0x01c5, code lost:
        r7 = -r7;
     */
    /* JADX WARN: Code restructure failed: missing block: B:147:0x01c6, code lost:
        r17.o = r7;
        r17.m.c(r1);
        r14 = 16;
        r17.n = 16;
     */
    /* JADX WARN: Code restructure failed: missing block: B:148:0x01d3, code lost:
        r2 = 2;
     */
    /* JADX WARN: Code restructure failed: missing block: B:149:0x01d4, code lost:
        if (r5 == r2) goto L105;
     */
    /* JADX WARN: Code restructure failed: missing block: B:150:0x01d6, code lost:
        if (r5 == 4) goto L105;
     */
    /* JADX WARN: Code restructure failed: missing block: B:152:0x01d9, code lost:
        if (r5 != 7) goto L66;
     */
    /* JADX WARN: Code restructure failed: missing block: B:153:0x01db, code lost:
        r17.p = r1;
        r14 = 17;
        r17.n = 17;
     */
    /* JADX WARN: Removed duplicated region for block: B:174:0x020f A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:175:0x0210  */
    /* JADX WARN: Removed duplicated region for block: B:87:0x0124 A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:88:0x0125  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final int T() {
        String str;
        String str2;
        int i2;
        byte D;
        int i3;
        char c2;
        char c3;
        int[] iArr = this.f2975d;
        int i4 = this.f2974c;
        int i5 = iArr[i4 - 1];
        char c4 = 2;
        if (i5 == 1) {
            iArr[i4 - 1] = 2;
        } else if (i5 == 2) {
            int W = W(true);
            this.m.readByte();
            if (W != 44) {
                if (W == 59) {
                    S();
                    throw null;
                } else if (W == 93) {
                    this.n = 4;
                    return 4;
                } else {
                    R("Unterminated array");
                    throw null;
                }
            }
        } else if (i5 == 3 || i5 == 5) {
            iArr[i4 - 1] = 4;
            if (i5 == 5) {
                int W2 = W(true);
                this.m.readByte();
                if (W2 != 44) {
                    if (W2 == 59) {
                        S();
                        throw null;
                    } else if (W2 == 125) {
                        this.n = 2;
                        return 2;
                    } else {
                        R("Unterminated object");
                        throw null;
                    }
                }
            }
            int W3 = W(true);
            if (W3 == 34) {
                this.m.readByte();
                this.n = 13;
                return 13;
            } else if (W3 == 39) {
                this.m.readByte();
                S();
                throw null;
            } else if (W3 != 125) {
                S();
                throw null;
            } else if (i5 != 5) {
                this.m.readByte();
                this.n = 2;
                return 2;
            } else {
                R("Expected name");
                throw null;
            }
        } else if (i5 == 4) {
            iArr[i4 - 1] = 5;
            int W4 = W(true);
            this.m.readByte();
            if (W4 != 58) {
                if (W4 != 61) {
                    R("Expected ':'");
                    throw null;
                }
                S();
                throw null;
            }
        } else if (i5 == 6) {
            iArr[i4 - 1] = 7;
        } else if (i5 == 7) {
            if (W(false) == -1) {
                this.n = 18;
                return 18;
            }
            S();
            throw null;
        } else if (i5 == 8) {
            throw new IllegalStateException("JsonReader is closed");
        }
        int W5 = W(true);
        if (W5 == 34) {
            this.m.readByte();
            this.n = 9;
            return 9;
        } else if (W5 == 39) {
            S();
            throw null;
        } else {
            if (W5 != 44 && W5 != 59) {
                if (W5 == 91) {
                    this.m.readByte();
                    this.n = 3;
                    return 3;
                } else if (W5 != 93) {
                    if (W5 != 123) {
                        byte D2 = this.m.D(0L);
                        if (D2 == 116 || D2 == 84) {
                            str = "true";
                            str2 = "TRUE";
                            i2 = 5;
                        } else if (D2 == 102 || D2 == 70) {
                            str = "false";
                            str2 = "FALSE";
                            i2 = 6;
                        } else if (D2 == 110 || D2 == 78) {
                            str = "null";
                            str2 = "NULL";
                            i2 = 7;
                        } else {
                            i2 = 0;
                            if (i2 == 0) {
                                return i2;
                            }
                            long j2 = 0;
                            boolean z = true;
                            int i6 = 0;
                            char c5 = 0;
                            boolean z2 = false;
                            while (true) {
                                int i7 = i6 + 1;
                                if (!this.l.o(i7)) {
                                    char c6 = c4;
                                    break;
                                }
                                byte D3 = this.m.D(i6);
                                if (D3 != 43) {
                                    if (D3 == 69 || D3 == 101) {
                                        if (c5 != 2 && c5 != 4) {
                                            break;
                                        }
                                        c5 = 5;
                                    } else if (D3 == 45) {
                                        c2 = 6;
                                        if (c5 == 0) {
                                            c5 = 1;
                                            z2 = true;
                                        } else {
                                            if (c5 != 5) {
                                                break;
                                            }
                                            c3 = c2;
                                            c5 = c3;
                                        }
                                    } else if (D3 == 46) {
                                        if (c5 != 2) {
                                            break;
                                        }
                                        c3 = 3;
                                        c5 = c3;
                                    } else if (D3 < 48 || D3 > 57) {
                                        break;
                                    } else if (c5 == 1 || c5 == 0) {
                                        j2 = -(D3 - 48);
                                        c5 = 2;
                                    } else if (c5 == 2) {
                                        if (j2 == 0) {
                                            break;
                                        }
                                        long j3 = (10 * j2) - (D3 - 48);
                                        int i8 = (j2 > (-922337203685477580L) ? 1 : (j2 == (-922337203685477580L) ? 0 : -1));
                                        z = (i8 > 0 || (i8 == 0 && j3 < j2)) & z;
                                        j2 = j3;
                                    } else if (c5 == 3) {
                                        c5 = 4;
                                    } else if (c5 == 5 || c5 == 6) {
                                        c5 = 7;
                                    }
                                    if (i3 == 0) {
                                        return i3;
                                    }
                                    if (!V(this.m.D(0L))) {
                                        R("Expected value");
                                        throw null;
                                    }
                                    S();
                                    throw null;
                                }
                                c2 = 6;
                                if (c5 != 5) {
                                    break;
                                }
                                c3 = c2;
                                c5 = c3;
                                i6 = i7;
                                c4 = 2;
                            }
                            i3 = 0;
                            if (i3 == 0) {
                            }
                        }
                        int length = str.length();
                        int i9 = 1;
                        while (true) {
                            if (i9 < length) {
                                int i10 = i9 + 1;
                                if (!this.l.o(i10) || ((D = this.m.D(i9)) != str.charAt(i9) && D != str2.charAt(i9))) {
                                    break;
                                }
                                i9 = i10;
                            } else if (!this.l.o(length + 1) || !V(this.m.D(length))) {
                                this.m.c(length);
                                this.n = i2;
                            }
                        }
                        i2 = 0;
                        if (i2 == 0) {
                        }
                    } else {
                        this.m.readByte();
                        this.n = 1;
                        return 1;
                    }
                } else if (i5 == 1) {
                    this.m.readByte();
                    this.n = 4;
                    return 4;
                }
            }
            if (i5 != 1 && i5 != 2) {
                R("Unexpected value");
                throw null;
            }
            S();
            throw null;
        }
    }

    public final int U(String str, c.a aVar) {
        int length = aVar.f2978a.length;
        for (int i2 = 0; i2 < length; i2++) {
            if (str.equals(aVar.f2978a[i2])) {
                this.n = 0;
                this.f2976e[this.f2974c - 1] = str;
                return i2;
            }
        }
        return -1;
    }

    public final boolean V(int i2) {
        if (i2 == 9 || i2 == 10 || i2 == 12 || i2 == 13 || i2 == 32) {
            return false;
        }
        if (i2 != 35) {
            if (i2 == 44) {
                return false;
            }
            if (i2 != 47 && i2 != 61) {
                if (i2 == 123 || i2 == 125 || i2 == 58) {
                    return false;
                }
                if (i2 != 59) {
                    switch (i2) {
                        case 91:
                        case 93:
                            return false;
                        case 92:
                            break;
                        default:
                            return true;
                    }
                }
            }
        }
        S();
        throw null;
    }

    public final int W(boolean z) {
        int i2 = 0;
        while (true) {
            int i3 = i2 + 1;
            if (!this.l.o(i3)) {
                if (z) {
                    throw new EOFException("End of input");
                }
                return -1;
            }
            byte D = this.m.D(i2);
            if (D != 10 && D != 32 && D != 13 && D != 9) {
                this.m.c(i3 - 1);
                if (D == 47) {
                    if (this.l.o(2L)) {
                        S();
                        throw null;
                    }
                    return D;
                } else if (D != 35) {
                    return D;
                } else {
                    S();
                    throw null;
                }
            }
            i2 = i3;
        }
    }

    public final String X(h hVar) {
        StringBuilder sb = null;
        while (true) {
            long g2 = this.l.g(hVar);
            if (g2 != -1) {
                if (this.m.D(g2) != 92) {
                    if (sb == null) {
                        String L = this.m.L(g2);
                        this.m.readByte();
                        return L;
                    }
                    sb.append(this.m.L(g2));
                    this.m.readByte();
                    return sb.toString();
                }
                if (sb == null) {
                    sb = new StringBuilder();
                }
                sb.append(this.m.L(g2));
                this.m.readByte();
                sb.append(Z());
            } else {
                R("Unterminated string");
                throw null;
            }
        }
    }

    public final String Y() {
        long g2 = this.l.g(i);
        return g2 != -1 ? this.m.L(g2) : this.m.K();
    }

    public final char Z() {
        int i2;
        int i3;
        if (this.l.o(1L)) {
            byte readByte = this.m.readByte();
            if (readByte == 10 || readByte == 34 || readByte == 39 || readByte == 47 || readByte == 92) {
                return (char) readByte;
            }
            if (readByte != 98) {
                if (readByte != 102) {
                    if (readByte != 110) {
                        if (readByte != 114) {
                            if (readByte != 116) {
                                if (readByte == 117) {
                                    if (this.l.o(4L)) {
                                        char c2 = 0;
                                        for (int i4 = 0; i4 < 4; i4++) {
                                            byte D = this.m.D(i4);
                                            char c3 = (char) (c2 << 4);
                                            if (D < 48 || D > 57) {
                                                if (D >= 97 && D <= 102) {
                                                    i2 = D - 97;
                                                } else if (D < 65 || D > 70) {
                                                    StringBuilder x = c.b.a.a.a.x("\\u");
                                                    x.append(this.m.L(4L));
                                                    R(x.toString());
                                                    throw null;
                                                } else {
                                                    i2 = D - 65;
                                                }
                                                i3 = i2 + 10;
                                            } else {
                                                i3 = D - 48;
                                            }
                                            c2 = (char) (i3 + c3);
                                        }
                                        this.m.c(4L);
                                        return c2;
                                    }
                                    StringBuilder x2 = c.b.a.a.a.x("Unterminated escape sequence at path ");
                                    x2.append(F());
                                    throw new EOFException(x2.toString());
                                }
                                StringBuilder x3 = c.b.a.a.a.x("Invalid escape sequence: \\");
                                x3.append((char) readByte);
                                R(x3.toString());
                                throw null;
                            }
                            return '\t';
                        }
                        return '\r';
                    }
                    return '\n';
                }
                return '\f';
            }
            return '\b';
        }
        R("Unterminated escape sequence");
        throw null;
    }

    public final void a0(h hVar) {
        while (true) {
            long g2 = this.l.g(hVar);
            if (g2 != -1) {
                if (this.m.D(g2) == 92) {
                    this.m.c(g2 + 1);
                    Z();
                } else {
                    this.m.c(g2 + 1);
                    return;
                }
            } else {
                R("Unterminated string");
                throw null;
            }
        }
    }

    public final void b0() {
        long g2 = this.l.g(i);
        e eVar = this.m;
        if (g2 == -1) {
            g2 = eVar.f6176d;
        }
        eVar.c(g2);
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.n = 0;
        this.f2975d[0] = 8;
        this.f2974c = 1;
        this.m.B();
        this.l.close();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("JsonReader(");
        x.append(this.l);
        x.append(")");
        return x.toString();
    }
}