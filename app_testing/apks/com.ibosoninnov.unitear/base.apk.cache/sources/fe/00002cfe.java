package f.g0.f;

import f.e0;
import f.n;
import f.r;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.ProxySelector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/* compiled from: RouteSelector.java */
/* loaded from: classes2.dex */
public final class f {

    /* renamed from: a  reason: collision with root package name */
    public final f.a f5800a;

    /* renamed from: b  reason: collision with root package name */
    public final d f5801b;

    /* renamed from: c  reason: collision with root package name */
    public final f.d f5802c;

    /* renamed from: d  reason: collision with root package name */
    public final n f5803d;

    /* renamed from: e  reason: collision with root package name */
    public List<Proxy> f5804e;

    /* renamed from: f  reason: collision with root package name */
    public int f5805f;

    /* renamed from: g  reason: collision with root package name */
    public List<InetSocketAddress> f5806g = Collections.emptyList();

    /* renamed from: h  reason: collision with root package name */
    public final List<e0> f5807h = new ArrayList();

    /* compiled from: RouteSelector.java */
    /* loaded from: classes2.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final List<e0> f5808a;

        /* renamed from: b  reason: collision with root package name */
        public int f5809b = 0;

        public a(List<e0> list) {
            this.f5808a = list;
        }

        public boolean a() {
            return this.f5809b < this.f5808a.size();
        }
    }

    public f(f.a aVar, d dVar, f.d dVar2, n nVar) {
        this.f5804e = Collections.emptyList();
        this.f5800a = aVar;
        this.f5801b = dVar;
        this.f5802c = dVar2;
        this.f5803d = nVar;
        r rVar = aVar.f5715a;
        Proxy proxy = aVar.f5722h;
        if (proxy != null) {
            this.f5804e = Collections.singletonList(proxy);
        } else {
            List<Proxy> select = aVar.f5721g.select(rVar.o());
            this.f5804e = (select == null || select.isEmpty()) ? f.g0.c.q(Proxy.NO_PROXY) : f.g0.c.p(select);
        }
        this.f5805f = 0;
    }

    public void a(e0 e0Var, IOException iOException) {
        f.a aVar;
        ProxySelector proxySelector;
        if (e0Var.f5751b.type() != Proxy.Type.DIRECT && (proxySelector = (aVar = this.f5800a).f5721g) != null) {
            proxySelector.connectFailed(aVar.f5715a.o(), e0Var.f5751b.address(), iOException);
        }
        d dVar = this.f5801b;
        synchronized (dVar) {
            dVar.f5797a.add(e0Var);
        }
    }

    public boolean b() {
        return c() || !this.f5807h.isEmpty();
    }

    public final boolean c() {
        return this.f5805f < this.f5804e.size();
    }
}