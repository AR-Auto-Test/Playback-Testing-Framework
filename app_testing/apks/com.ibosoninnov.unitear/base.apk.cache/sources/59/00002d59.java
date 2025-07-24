package f;

import java.security.cert.Certificate;
import java.util.Collections;
import java.util.List;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLSession;

/* compiled from: Handshake.java */
/* loaded from: classes2.dex */
public final class p {

    /* renamed from: a  reason: collision with root package name */
    public final f0 f6080a;

    /* renamed from: b  reason: collision with root package name */
    public final g f6081b;

    /* renamed from: c  reason: collision with root package name */
    public final List<Certificate> f6082c;

    /* renamed from: d  reason: collision with root package name */
    public final List<Certificate> f6083d;

    public p(f0 f0Var, g gVar, List<Certificate> list, List<Certificate> list2) {
        this.f6080a = f0Var;
        this.f6081b = gVar;
        this.f6082c = list;
        this.f6083d = list2;
    }

    public static p a(SSLSession sSLSession) {
        Certificate[] certificateArr;
        List emptyList;
        List emptyList2;
        String cipherSuite = sSLSession.getCipherSuite();
        if (cipherSuite != null) {
            g a2 = g.a(cipherSuite);
            String protocol = sSLSession.getProtocol();
            if (protocol != null) {
                f0 a3 = f0.a(protocol);
                try {
                    certificateArr = sSLSession.getPeerCertificates();
                } catch (SSLPeerUnverifiedException unused) {
                    certificateArr = null;
                }
                if (certificateArr != null) {
                    emptyList = f.g0.c.q(certificateArr);
                } else {
                    emptyList = Collections.emptyList();
                }
                Certificate[] localCertificates = sSLSession.getLocalCertificates();
                if (localCertificates != null) {
                    emptyList2 = f.g0.c.q(localCertificates);
                } else {
                    emptyList2 = Collections.emptyList();
                }
                return new p(a3, a2, emptyList, emptyList2);
            }
            throw new IllegalStateException("tlsVersion == null");
        }
        throw new IllegalStateException("cipherSuite == null");
    }

    public boolean equals(Object obj) {
        if (obj instanceof p) {
            p pVar = (p) obj;
            return this.f6080a.equals(pVar.f6080a) && this.f6081b.equals(pVar.f6081b) && this.f6082c.equals(pVar.f6082c) && this.f6083d.equals(pVar.f6083d);
        }
        return false;
    }

    public int hashCode() {
        int hashCode = this.f6081b.hashCode();
        int hashCode2 = this.f6082c.hashCode();
        return this.f6083d.hashCode() + ((hashCode2 + ((hashCode + ((this.f6080a.hashCode() + 527) * 31)) * 31)) * 31);
    }
}