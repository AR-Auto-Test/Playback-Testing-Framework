package c.a.a.b0.h0;

import g.e;
import g.h;
import g.q;
import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;

/* compiled from: JsonReader.java */
/* loaded from: classes.dex */
public abstract class c implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public static final String[] f2973b = new String[128];

    /* renamed from: c  reason: collision with root package name */
    public int f2974c;

    /* renamed from: d  reason: collision with root package name */
    public int[] f2975d = new int[32];

    /* renamed from: e  reason: collision with root package name */
    public String[] f2976e = new String[32];

    /* renamed from: f  reason: collision with root package name */
    public int[] f2977f = new int[32];

    /* compiled from: JsonReader.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final String[] f2978a;

        /* renamed from: b  reason: collision with root package name */
        public final q f2979b;

        public a(String[] strArr, q qVar) {
            this.f2978a = strArr;
            this.f2979b = qVar;
        }

        /* JADX WARN: Removed duplicated region for block: B:19:0x003a A[Catch: IOException -> 0x0069, TryCatch #0 {IOException -> 0x0069, blocks: (B:2:0x0000, B:3:0x000a, B:5:0x000d, B:7:0x001e, B:9:0x0026, B:21:0x0042, B:19:0x003a, B:20:0x003d, B:23:0x0047, B:24:0x004a, B:25:0x0059), top: B:30:0x0000 }] */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public static a a(String... strArr) {
            String str;
            try {
                h[] hVarArr = new h[strArr.length];
                e eVar = new e();
                for (int i = 0; i < strArr.length; i++) {
                    String str2 = strArr[i];
                    String[] strArr2 = c.f2973b;
                    eVar.T(34);
                    int length = str2.length();
                    int i2 = 0;
                    for (int i3 = 0; i3 < length; i3++) {
                        char charAt = str2.charAt(i3);
                        if (charAt < 128) {
                            str = strArr2[charAt];
                            if (str == null) {
                            }
                            if (i2 < i3) {
                                eVar.Z(str2, i2, i3);
                            }
                            eVar.Y(str);
                            i2 = i3 + 1;
                        } else {
                            if (charAt == 8232) {
                                str = "\\u2028";
                            } else if (charAt == 8233) {
                                str = "\\u2029";
                            }
                            if (i2 < i3) {
                            }
                            eVar.Y(str);
                            i2 = i3 + 1;
                        }
                    }
                    if (i2 < length) {
                        eVar.Z(str2, i2, length);
                    }
                    eVar.T(34);
                    eVar.readByte();
                    hVarArr[i] = eVar.H();
                }
                return new a((String[]) strArr.clone(), q.b(hVarArr));
            } catch (IOException e2) {
                throw new AssertionError(e2);
            }
        }
    }

    /* compiled from: JsonReader.java */
    /* loaded from: classes.dex */
    public enum b {
        BEGIN_ARRAY,
        END_ARRAY,
        BEGIN_OBJECT,
        END_OBJECT,
        NAME,
        STRING,
        NUMBER,
        BOOLEAN,
        NULL,
        END_DOCUMENT
    }

    static {
        for (int i = 0; i <= 31; i++) {
            f2973b[i] = String.format("\\u%04x", Integer.valueOf(i));
        }
        String[] strArr = f2973b;
        strArr[34] = "\\\"";
        strArr[92] = "\\\\";
        strArr[9] = "\\t";
        strArr[8] = "\\b";
        strArr[10] = "\\n";
        strArr[13] = "\\r";
        strArr[12] = "\\f";
    }

    public abstract void B();

    public abstract void C();

    public abstract void D();

    public abstract void E();

    public final String F() {
        int i = this.f2974c;
        int[] iArr = this.f2975d;
        String[] strArr = this.f2976e;
        int[] iArr2 = this.f2977f;
        StringBuilder sb = new StringBuilder();
        sb.append('$');
        for (int i2 = 0; i2 < i; i2++) {
            int i3 = iArr[i2];
            if (i3 == 1 || i3 == 2) {
                sb.append('[');
                sb.append(iArr2[i2]);
                sb.append(']');
            } else if (i3 == 3 || i3 == 4 || i3 == 5) {
                sb.append('.');
                if (strArr[i2] != null) {
                    sb.append(strArr[i2]);
                }
            }
        }
        return sb.toString();
    }

    public abstract boolean G();

    public abstract boolean H();

    public abstract double I();

    public abstract int J();

    public abstract String K();

    public abstract String L();

    public abstract b M();

    public final void N(int i) {
        int i2 = this.f2974c;
        int[] iArr = this.f2975d;
        if (i2 == iArr.length) {
            if (i2 != 256) {
                this.f2975d = Arrays.copyOf(iArr, iArr.length * 2);
                String[] strArr = this.f2976e;
                this.f2976e = (String[]) Arrays.copyOf(strArr, strArr.length * 2);
                int[] iArr2 = this.f2977f;
                this.f2977f = Arrays.copyOf(iArr2, iArr2.length * 2);
            } else {
                StringBuilder x = c.b.a.a.a.x("Nesting too deep at ");
                x.append(F());
                throw new c.a.a.b0.h0.a(x.toString());
            }
        }
        int[] iArr3 = this.f2975d;
        int i3 = this.f2974c;
        this.f2974c = i3 + 1;
        iArr3[i3] = i;
    }

    public abstract int O(a aVar);

    public abstract void P();

    public abstract void Q();

    public final c.a.a.b0.h0.b R(String str) {
        StringBuilder A = c.b.a.a.a.A(str, " at path ");
        A.append(F());
        throw new c.a.a.b0.h0.b(A.toString());
    }
}