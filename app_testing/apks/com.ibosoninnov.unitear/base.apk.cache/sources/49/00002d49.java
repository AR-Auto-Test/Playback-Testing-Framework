package f.g0.l;

import java.security.cert.CertificateParsingException;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLException;
import javax.net.ssl.SSLSession;

/* compiled from: OkHostnameVerifier.java */
/* loaded from: classes2.dex */
public final class d implements HostnameVerifier {

    /* renamed from: a  reason: collision with root package name */
    public static final d f6044a = new d();

    public static List<String> a(X509Certificate x509Certificate) {
        List<String> b2 = b(x509Certificate, 7);
        List<String> b3 = b(x509Certificate, 2);
        ArrayList arrayList = new ArrayList(b3.size() + b2.size());
        arrayList.addAll(b2);
        arrayList.addAll(b3);
        return arrayList;
    }

    public static List<String> b(X509Certificate x509Certificate, int i) {
        Integer num;
        String str;
        ArrayList arrayList = new ArrayList();
        try {
            Collection<List<?>> subjectAlternativeNames = x509Certificate.getSubjectAlternativeNames();
            if (subjectAlternativeNames == null) {
                return Collections.emptyList();
            }
            for (List<?> list : subjectAlternativeNames) {
                if (list != null && list.size() >= 2 && (num = (Integer) list.get(0)) != null && num.intValue() == i && (str = (String) list.get(1)) != null) {
                    arrayList.add(str);
                }
            }
            return arrayList;
        } catch (CertificateParsingException unused) {
            return Collections.emptyList();
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:67:0x0101 A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean c(String str, X509Certificate x509Certificate) {
        boolean z;
        int length;
        if (f.g0.c.q.matcher(str).matches()) {
            List<String> b2 = b(x509Certificate, 7);
            int size = b2.size();
            for (int i = 0; i < size; i++) {
                if (str.equalsIgnoreCase(b2.get(i))) {
                    return true;
                }
            }
        } else {
            String lowerCase = str.toLowerCase(Locale.US);
            for (String str2 : b(x509Certificate, 2)) {
                if (lowerCase != null && lowerCase.length() != 0 && !lowerCase.startsWith(".") && !lowerCase.endsWith("..") && str2 != null && str2.length() != 0 && !str2.startsWith(".") && !str2.endsWith("..")) {
                    String str3 = lowerCase.endsWith(".") ? lowerCase : lowerCase + '.';
                    if (!str2.endsWith(".")) {
                        str2 = str2 + '.';
                    }
                    String lowerCase2 = str2.toLowerCase(Locale.US);
                    if (!lowerCase2.contains("*")) {
                        z = str3.equals(lowerCase2);
                        continue;
                    } else if (lowerCase2.startsWith("*.") && lowerCase2.indexOf(42, 1) == -1 && str3.length() >= lowerCase2.length() && !"*.".equals(lowerCase2)) {
                        String substring = lowerCase2.substring(1);
                        if (str3.endsWith(substring) && ((length = str3.length() - substring.length()) <= 0 || str3.lastIndexOf(46, length - 1) == -1)) {
                            z = true;
                            continue;
                        }
                    }
                    if (z) {
                        return true;
                    }
                }
                z = false;
                continue;
                if (z) {
                }
            }
        }
        return false;
    }

    @Override // javax.net.ssl.HostnameVerifier
    public boolean verify(String str, SSLSession sSLSession) {
        try {
            return c(str, (X509Certificate) sSLSession.getPeerCertificates()[0]);
        } catch (SSLException unused) {
            return false;
        }
    }
}