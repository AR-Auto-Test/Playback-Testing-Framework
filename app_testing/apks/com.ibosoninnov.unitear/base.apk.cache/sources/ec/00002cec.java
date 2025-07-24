package f;

import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.net.ssl.SSLPeerUnverifiedException;

/* compiled from: CertificatePinner.java */
/* loaded from: classes2.dex */
public final class f {

    /* renamed from: a  reason: collision with root package name */
    public static final f f5753a = new f(new LinkedHashSet(new ArrayList()), null);

    /* renamed from: b  reason: collision with root package name */
    public final Set<a> f5754b;

    /* renamed from: c  reason: collision with root package name */
    public final f.g0.l.c f5755c;

    /* compiled from: CertificatePinner.java */
    /* loaded from: classes2.dex */
    public static final class a {
        public boolean equals(Object obj) {
            if (obj instanceof a) {
                Objects.requireNonNull((a) obj);
                throw null;
            }
            return false;
        }

        public int hashCode() {
            throw null;
        }

        public String toString() {
            new StringBuilder().append((String) null);
            throw null;
        }
    }

    public f(Set<a> set, f.g0.l.c cVar) {
        this.f5754b = set;
        this.f5755c = cVar;
    }

    public static String b(Certificate certificate) {
        if (certificate instanceof X509Certificate) {
            StringBuilder x = c.b.a.a.a.x("sha256/");
            x.append(g.h.i(((X509Certificate) certificate).getPublicKey().getEncoded()).d("SHA-256").a());
            return x.toString();
        }
        throw new IllegalArgumentException("Certificate pinning requires X509 certificates");
    }

    public void a(String str, List<Certificate> list) {
        List emptyList = Collections.emptyList();
        Iterator<a> it = this.f5754b.iterator();
        if (!it.hasNext()) {
            if (emptyList.isEmpty()) {
                return;
            }
            f.g0.l.c cVar = this.f5755c;
            if (cVar != null) {
                list = cVar.a(list, str);
            }
            int size = list.size();
            for (int i = 0; i < size; i++) {
                X509Certificate x509Certificate = (X509Certificate) list.get(i);
                if (emptyList.size() > 0) {
                    Objects.requireNonNull((a) emptyList.get(0));
                    throw null;
                }
            }
            StringBuilder A = c.b.a.a.a.A("Certificate pinning failure!", "\n  Peer certificate chain:");
            int size2 = list.size();
            for (int i2 = 0; i2 < size2; i2++) {
                X509Certificate x509Certificate2 = (X509Certificate) list.get(i2);
                A.append("\n    ");
                A.append(b(x509Certificate2));
                A.append(": ");
                A.append(x509Certificate2.getSubjectDN().getName());
            }
            A.append("\n  Pinned certificates for ");
            A.append(str);
            A.append(":");
            int size3 = emptyList.size();
            for (int i3 = 0; i3 < size3; i3++) {
                A.append("\n    ");
                A.append((a) emptyList.get(i3));
            }
            throw new SSLPeerUnverifiedException(A.toString());
        }
        Objects.requireNonNull(it.next());
        throw null;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof f) {
            f fVar = (f) obj;
            if (f.g0.c.m(this.f5755c, fVar.f5755c) && this.f5754b.equals(fVar.f5754b)) {
                return true;
            }
        }
        return false;
    }

    public int hashCode() {
        f.g0.l.c cVar = this.f5755c;
        return this.f5754b.hashCode() + ((cVar != null ? cVar.hashCode() : 0) * 31);
    }
}