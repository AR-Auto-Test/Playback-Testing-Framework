package f;

import java.net.InetSocketAddress;
import java.net.Proxy;
import java.util.Objects;

/* compiled from: Route.java */
/* loaded from: classes2.dex */
public final class e0 {

    /* renamed from: a  reason: collision with root package name */
    public final a f5750a;

    /* renamed from: b  reason: collision with root package name */
    public final Proxy f5751b;

    /* renamed from: c  reason: collision with root package name */
    public final InetSocketAddress f5752c;

    public e0(a aVar, Proxy proxy, InetSocketAddress inetSocketAddress) {
        Objects.requireNonNull(aVar, "address == null");
        Objects.requireNonNull(inetSocketAddress, "inetSocketAddress == null");
        this.f5750a = aVar;
        this.f5751b = proxy;
        this.f5752c = inetSocketAddress;
    }

    public boolean a() {
        return this.f5750a.i != null && this.f5751b.type() == Proxy.Type.HTTP;
    }

    public boolean equals(Object obj) {
        if (obj instanceof e0) {
            e0 e0Var = (e0) obj;
            if (e0Var.f5750a.equals(this.f5750a) && e0Var.f5751b.equals(this.f5751b) && e0Var.f5752c.equals(this.f5752c)) {
                return true;
            }
        }
        return false;
    }

    public int hashCode() {
        int hashCode = this.f5751b.hashCode();
        return this.f5752c.hashCode() + ((hashCode + ((this.f5750a.hashCode() + 527) * 31)) * 31);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Route{");
        x.append(this.f5752c);
        x.append("}");
        return x.toString();
    }
}