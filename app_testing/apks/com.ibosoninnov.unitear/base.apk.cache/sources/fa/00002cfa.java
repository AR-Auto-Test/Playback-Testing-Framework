package f.g0.f;

import f.g;
import f.i;
import f.v;
import java.net.UnknownServiceException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import javax.net.ssl.SSLSocket;

/* compiled from: ConnectionSpecSelector.java */
/* loaded from: classes2.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public final List<i> f5786a;

    /* renamed from: b  reason: collision with root package name */
    public int f5787b = 0;

    /* renamed from: c  reason: collision with root package name */
    public boolean f5788c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f5789d;

    public b(List<i> list) {
        this.f5786a = list;
    }

    public i a(SSLSocket sSLSocket) {
        i iVar;
        boolean z;
        String[] enabledCipherSuites;
        String[] enabledProtocols;
        int i = this.f5787b;
        int size = this.f5786a.size();
        while (true) {
            if (i >= size) {
                iVar = null;
                break;
            }
            iVar = this.f5786a.get(i);
            if (iVar.a(sSLSocket)) {
                this.f5787b = i + 1;
                break;
            }
            i++;
        }
        if (iVar != null) {
            int i2 = this.f5787b;
            while (true) {
                if (i2 >= this.f5786a.size()) {
                    z = false;
                    break;
                } else if (this.f5786a.get(i2).a(sSLSocket)) {
                    z = true;
                    break;
                } else {
                    i2++;
                }
            }
            this.f5788c = z;
            f.g0.a aVar = f.g0.a.f5771a;
            boolean z2 = this.f5789d;
            Objects.requireNonNull((v.a) aVar);
            if (iVar.f6058f != null) {
                enabledCipherSuites = f.g0.c.s(f.g.f5763a, sSLSocket.getEnabledCipherSuites(), iVar.f6058f);
            } else {
                enabledCipherSuites = sSLSocket.getEnabledCipherSuites();
            }
            if (iVar.f6059g != null) {
                enabledProtocols = f.g0.c.s(f.g0.c.p, sSLSocket.getEnabledProtocols(), iVar.f6059g);
            } else {
                enabledProtocols = sSLSocket.getEnabledProtocols();
            }
            String[] supportedCipherSuites = sSLSocket.getSupportedCipherSuites();
            Comparator<String> comparator = f.g.f5763a;
            byte[] bArr = f.g0.c.f5773a;
            int length = supportedCipherSuites.length;
            int i3 = 0;
            while (true) {
                if (i3 >= length) {
                    i3 = -1;
                    break;
                }
                if (((g.a) comparator).compare(supportedCipherSuites[i3], "TLS_FALLBACK_SCSV") == 0) {
                    break;
                }
                i3++;
            }
            if (z2 && i3 != -1) {
                String str = supportedCipherSuites[i3];
                int length2 = enabledCipherSuites.length + 1;
                String[] strArr = new String[length2];
                System.arraycopy(enabledCipherSuites, 0, strArr, 0, enabledCipherSuites.length);
                strArr[length2 - 1] = str;
                enabledCipherSuites = strArr;
            }
            boolean z3 = iVar.f6056d;
            if (z3) {
                if (enabledCipherSuites.length != 0) {
                    String[] strArr2 = (String[]) enabledCipherSuites.clone();
                    if (z3) {
                        if (enabledProtocols.length != 0) {
                            sSLSocket.setEnabledProtocols((String[]) enabledProtocols.clone());
                            sSLSocket.setEnabledCipherSuites(strArr2);
                            return iVar;
                        }
                        throw new IllegalArgumentException("At least one TLS version is required");
                    }
                    throw new IllegalStateException("no TLS versions for cleartext connections");
                }
                throw new IllegalArgumentException("At least one cipher suite is required");
            }
            throw new IllegalStateException("no cipher suites for cleartext connections");
        }
        StringBuilder x = c.b.a.a.a.x("Unable to find acceptable protocols. isFallback=");
        x.append(this.f5789d);
        x.append(", modes=");
        x.append(this.f5786a);
        x.append(", supported protocols=");
        x.append(Arrays.toString(sSLSocket.getEnabledProtocols()));
        throw new UnknownServiceException(x.toString());
    }
}