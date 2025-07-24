package f;

import f.r;
import java.net.Proxy;
import java.net.ProxySelector;
import java.util.List;
import java.util.Objects;
import javax.net.SocketFactory;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLSocketFactory;

/* compiled from: Address.java */
/* loaded from: classes2.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public final r f5715a;

    /* renamed from: b  reason: collision with root package name */
    public final m f5716b;

    /* renamed from: c  reason: collision with root package name */
    public final SocketFactory f5717c;

    /* renamed from: d  reason: collision with root package name */
    public final b f5718d;

    /* renamed from: e  reason: collision with root package name */
    public final List<w> f5719e;

    /* renamed from: f  reason: collision with root package name */
    public final List<i> f5720f;

    /* renamed from: g  reason: collision with root package name */
    public final ProxySelector f5721g;

    /* renamed from: h  reason: collision with root package name */
    public final Proxy f5722h;
    public final SSLSocketFactory i;
    public final HostnameVerifier j;
    public final f k;

    public a(String str, int i, m mVar, SocketFactory socketFactory, SSLSocketFactory sSLSocketFactory, HostnameVerifier hostnameVerifier, f fVar, b bVar, Proxy proxy, List<w> list, List<i> list2, ProxySelector proxySelector) {
        r.a aVar = new r.a();
        String str2 = sSLSocketFactory != null ? "https" : "http";
        if (str2.equalsIgnoreCase("http")) {
            aVar.f6094a = "http";
        } else if (str2.equalsIgnoreCase("https")) {
            aVar.f6094a = "https";
        } else {
            throw new IllegalArgumentException(c.b.a.a.a.q("unexpected scheme: ", str2));
        }
        Objects.requireNonNull(str, "host == null");
        String c2 = f.g0.c.c(r.j(str, 0, str.length(), false));
        if (c2 != null) {
            aVar.f6097d = c2;
            if (i > 0 && i <= 65535) {
                aVar.f6098e = i;
                this.f5715a = aVar.a();
                Objects.requireNonNull(mVar, "dns == null");
                this.f5716b = mVar;
                Objects.requireNonNull(socketFactory, "socketFactory == null");
                this.f5717c = socketFactory;
                Objects.requireNonNull(bVar, "proxyAuthenticator == null");
                this.f5718d = bVar;
                Objects.requireNonNull(list, "protocols == null");
                this.f5719e = f.g0.c.p(list);
                Objects.requireNonNull(list2, "connectionSpecs == null");
                this.f5720f = f.g0.c.p(list2);
                Objects.requireNonNull(proxySelector, "proxySelector == null");
                this.f5721g = proxySelector;
                this.f5722h = null;
                this.i = sSLSocketFactory;
                this.j = hostnameVerifier;
                this.k = fVar;
                return;
            }
            throw new IllegalArgumentException(c.b.a.a.a.j("unexpected port: ", i));
        }
        throw new IllegalArgumentException(c.b.a.a.a.q("unexpected host: ", str));
    }

    public boolean a(a aVar) {
        return this.f5716b.equals(aVar.f5716b) && this.f5718d.equals(aVar.f5718d) && this.f5719e.equals(aVar.f5719e) && this.f5720f.equals(aVar.f5720f) && this.f5721g.equals(aVar.f5721g) && f.g0.c.m(this.f5722h, aVar.f5722h) && f.g0.c.m(this.i, aVar.i) && f.g0.c.m(this.j, aVar.j) && f.g0.c.m(this.k, aVar.k) && this.f5715a.f6091f == aVar.f5715a.f6091f;
    }

    public boolean equals(Object obj) {
        if (obj instanceof a) {
            a aVar = (a) obj;
            if (this.f5715a.equals(aVar.f5715a) && a(aVar)) {
                return true;
            }
        }
        return false;
    }

    public int hashCode() {
        int hashCode = this.f5716b.hashCode();
        int hashCode2 = this.f5718d.hashCode();
        int hashCode3 = this.f5719e.hashCode();
        int hashCode4 = (this.f5721g.hashCode() + ((this.f5720f.hashCode() + ((hashCode3 + ((hashCode2 + ((hashCode + ((this.f5715a.hashCode() + 527) * 31)) * 31)) * 31)) * 31)) * 31)) * 31;
        Proxy proxy = this.f5722h;
        int hashCode5 = (hashCode4 + (proxy != null ? proxy.hashCode() : 0)) * 31;
        SSLSocketFactory sSLSocketFactory = this.i;
        int hashCode6 = (hashCode5 + (sSLSocketFactory != null ? sSLSocketFactory.hashCode() : 0)) * 31;
        HostnameVerifier hostnameVerifier = this.j;
        int hashCode7 = (hashCode6 + (hostnameVerifier != null ? hostnameVerifier.hashCode() : 0)) * 31;
        f fVar = this.k;
        return hashCode7 + (fVar != null ? fVar.hashCode() : 0);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Address{");
        x.append(this.f5715a.f6090e);
        x.append(":");
        x.append(this.f5715a.f6091f);
        if (this.f5722h != null) {
            x.append(", proxy=");
            x.append(this.f5722h);
        } else {
            x.append(", proxySelector=");
            x.append(this.f5721g);
        }
        x.append("}");
        return x.toString();
    }
}