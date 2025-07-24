package f;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import javax.net.ssl.SSLSocket;

/* compiled from: ConnectionSpec.java */
/* loaded from: classes2.dex */
public final class i {

    /* renamed from: a  reason: collision with root package name */
    public static final g[] f6053a;

    /* renamed from: b  reason: collision with root package name */
    public static final i f6054b;

    /* renamed from: c  reason: collision with root package name */
    public static final i f6055c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f6056d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f6057e;

    /* renamed from: f  reason: collision with root package name */
    public final String[] f6058f;

    /* renamed from: g  reason: collision with root package name */
    public final String[] f6059g;

    static {
        g[] gVarArr = {g.j, g.l, g.k, g.m, g.o, g.n, g.f5770h, g.i, g.f5768f, g.f5769g, g.f5766d, g.f5767e, g.f5765c};
        f6053a = gVarArr;
        a aVar = new a(true);
        String[] strArr = new String[13];
        for (int i = 0; i < 13; i++) {
            strArr[i] = gVarArr[i].p;
        }
        aVar.a(strArr);
        f0 f0Var = f0.TLS_1_0;
        aVar.c(f0.TLS_1_3, f0.TLS_1_2, f0.TLS_1_1, f0Var);
        if (aVar.f6060a) {
            aVar.f6063d = true;
            i iVar = new i(aVar);
            f6054b = iVar;
            a aVar2 = new a(iVar);
            aVar2.c(f0Var);
            if (aVar2.f6060a) {
                aVar2.f6063d = true;
                f6055c = new i(new a(false));
                return;
            }
            throw new IllegalStateException("no TLS extensions for cleartext connections");
        }
        throw new IllegalStateException("no TLS extensions for cleartext connections");
    }

    public i(a aVar) {
        this.f6056d = aVar.f6060a;
        this.f6058f = aVar.f6061b;
        this.f6059g = aVar.f6062c;
        this.f6057e = aVar.f6063d;
    }

    public boolean a(SSLSocket sSLSocket) {
        if (this.f6056d) {
            String[] strArr = this.f6059g;
            if (strArr == null || f.g0.c.u(f.g0.c.p, strArr, sSLSocket.getEnabledProtocols())) {
                String[] strArr2 = this.f6058f;
                return strArr2 == null || f.g0.c.u(g.f5763a, strArr2, sSLSocket.getEnabledCipherSuites());
            }
            return false;
        }
        return false;
    }

    public boolean equals(Object obj) {
        if (obj instanceof i) {
            if (obj == this) {
                return true;
            }
            i iVar = (i) obj;
            boolean z = this.f6056d;
            if (z != iVar.f6056d) {
                return false;
            }
            return !z || (Arrays.equals(this.f6058f, iVar.f6058f) && Arrays.equals(this.f6059g, iVar.f6059g) && this.f6057e == iVar.f6057e);
        }
        return false;
    }

    public int hashCode() {
        if (this.f6056d) {
            return ((((527 + Arrays.hashCode(this.f6058f)) * 31) + Arrays.hashCode(this.f6059g)) * 31) + (!this.f6057e ? 1 : 0);
        }
        return 17;
    }

    public String toString() {
        String str;
        List list;
        if (this.f6056d) {
            String[] strArr = this.f6058f;
            List list2 = null;
            String str2 = "[all enabled]";
            if (strArr != null) {
                if (strArr != null) {
                    ArrayList arrayList = new ArrayList(strArr.length);
                    for (String str3 : strArr) {
                        arrayList.add(g.a(str3));
                    }
                    list = Collections.unmodifiableList(arrayList);
                } else {
                    list = null;
                }
                str = list.toString();
            } else {
                str = "[all enabled]";
            }
            String[] strArr2 = this.f6059g;
            if (strArr2 != null) {
                if (strArr2 != null) {
                    ArrayList arrayList2 = new ArrayList(strArr2.length);
                    for (String str4 : strArr2) {
                        arrayList2.add(f0.a(str4));
                    }
                    list2 = Collections.unmodifiableList(arrayList2);
                }
                str2 = list2.toString();
            }
            return "ConnectionSpec(cipherSuites=" + str + ", tlsVersions=" + str2 + ", supportsTlsExtensions=" + this.f6057e + ")";
        }
        return "ConnectionSpec()";
    }

    /* compiled from: ConnectionSpec.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public boolean f6060a;

        /* renamed from: b  reason: collision with root package name */
        public String[] f6061b;

        /* renamed from: c  reason: collision with root package name */
        public String[] f6062c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f6063d;

        public a(boolean z) {
            this.f6060a = z;
        }

        public a a(String... strArr) {
            if (this.f6060a) {
                if (strArr.length != 0) {
                    this.f6061b = (String[]) strArr.clone();
                    return this;
                }
                throw new IllegalArgumentException("At least one cipher suite is required");
            }
            throw new IllegalStateException("no cipher suites for cleartext connections");
        }

        public a b(String... strArr) {
            if (this.f6060a) {
                if (strArr.length != 0) {
                    this.f6062c = (String[]) strArr.clone();
                    return this;
                }
                throw new IllegalArgumentException("At least one TLS version is required");
            }
            throw new IllegalStateException("no TLS versions for cleartext connections");
        }

        public a c(f0... f0VarArr) {
            if (this.f6060a) {
                String[] strArr = new String[f0VarArr.length];
                for (int i = 0; i < f0VarArr.length; i++) {
                    strArr[i] = f0VarArr[i].f5762h;
                }
                b(strArr);
                return this;
            }
            throw new IllegalStateException("no TLS versions for cleartext connections");
        }

        public a(i iVar) {
            this.f6060a = iVar.f6056d;
            this.f6061b = iVar.f6058f;
            this.f6062c = iVar.f6059g;
            this.f6063d = iVar.f6057e;
        }
    }
}