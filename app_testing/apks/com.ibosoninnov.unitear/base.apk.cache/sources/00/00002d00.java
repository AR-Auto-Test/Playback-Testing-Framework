package f.g0.f;

import f.e0;
import f.g0.f.f;
import f.g0.i.u;
import f.h;
import f.m;
import f.n;
import f.r;
import f.v;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.Socket;
import java.net.SocketAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;

/* compiled from: StreamAllocation.java */
/* loaded from: classes2.dex */
public final class g {

    /* renamed from: a  reason: collision with root package name */
    public final f.a f5810a;

    /* renamed from: b  reason: collision with root package name */
    public f.a f5811b;

    /* renamed from: c  reason: collision with root package name */
    public e0 f5812c;

    /* renamed from: d  reason: collision with root package name */
    public final h f5813d;

    /* renamed from: e  reason: collision with root package name */
    public final f.d f5814e;

    /* renamed from: f  reason: collision with root package name */
    public final n f5815f;

    /* renamed from: g  reason: collision with root package name */
    public final Object f5816g;

    /* renamed from: h  reason: collision with root package name */
    public final f f5817h;
    public int i;
    public c j;
    public boolean k;
    public boolean l;
    public boolean m;
    public f.g0.g.c n;

    /* compiled from: StreamAllocation.java */
    /* loaded from: classes2.dex */
    public static final class a extends WeakReference<g> {

        /* renamed from: a  reason: collision with root package name */
        public final Object f5818a;

        public a(g gVar, Object obj) {
            super(gVar);
            this.f5818a = obj;
        }
    }

    public g(h hVar, f.a aVar, f.d dVar, n nVar, Object obj) {
        this.f5813d = hVar;
        this.f5810a = aVar;
        this.f5814e = dVar;
        this.f5815f = nVar;
        Objects.requireNonNull((v.a) f.g0.a.f5771a);
        this.f5817h = new f(aVar, hVar.f6050f, dVar, nVar);
        this.f5816g = obj;
    }

    public void a(c cVar, boolean z) {
        if (this.j == null) {
            this.j = cVar;
            this.k = z;
            cVar.n.add(new a(this, this.f5816g));
            return;
        }
        throw new IllegalStateException();
    }

    public synchronized c b() {
        return this.j;
    }

    public final Socket c(boolean z, boolean z2, boolean z3) {
        Socket socket;
        if (z3) {
            this.n = null;
        }
        boolean z4 = true;
        if (z2) {
            this.l = true;
        }
        c cVar = this.j;
        if (cVar != null) {
            if (z) {
                cVar.k = true;
            }
            if (this.n == null) {
                if (this.l || cVar.k) {
                    int size = cVar.n.size();
                    for (int i = 0; i < size; i++) {
                        if (cVar.n.get(i).get() == this) {
                            cVar.n.remove(i);
                            if (this.j.n.isEmpty()) {
                                this.j.o = System.nanoTime();
                                f.g0.a aVar = f.g0.a.f5771a;
                                h hVar = this.f5813d;
                                c cVar2 = this.j;
                                Objects.requireNonNull((v.a) aVar);
                                Objects.requireNonNull(hVar);
                                if (!cVar2.k && hVar.f6046b != 0) {
                                    hVar.notifyAll();
                                    z4 = false;
                                } else {
                                    hVar.f6049e.remove(cVar2);
                                }
                                if (z4) {
                                    socket = this.j.f5793e;
                                    this.j = null;
                                    return socket;
                                }
                            }
                            socket = null;
                            this.j = null;
                            return socket;
                        }
                    }
                    throw new IllegalStateException();
                }
                return null;
            }
            return null;
        }
        return null;
    }

    public final c d(int i, int i2, int i3, int i4, boolean z) {
        c cVar;
        e0 e0Var;
        Socket c2;
        c cVar2;
        boolean z2;
        boolean z3;
        Socket socket;
        f.a aVar;
        String str;
        int i5;
        boolean contains;
        synchronized (this.f5813d) {
            if (!this.l) {
                if (this.n == null) {
                    if (!this.m) {
                        cVar = this.j;
                        e0Var = null;
                        c2 = (cVar == null || !cVar.k) ? null : c(false, false, true);
                        cVar2 = this.j;
                        if (cVar2 != null) {
                            cVar = null;
                        } else {
                            cVar2 = null;
                        }
                        if (!this.k) {
                            cVar = null;
                        }
                        if (cVar2 == null) {
                            f.g0.a.f5771a.c(this.f5813d, this.f5810a, this, null);
                            c cVar3 = this.j;
                            if (cVar3 != null) {
                                cVar2 = cVar3;
                                z2 = true;
                            } else {
                                e0Var = this.f5812c;
                            }
                        }
                        z2 = false;
                    } else {
                        throw new IOException("Canceled");
                    }
                } else {
                    throw new IllegalStateException("codec != null");
                }
            } else {
                throw new IllegalStateException("released");
            }
        }
        f.g0.c.g(c2);
        if (cVar != null) {
            Objects.requireNonNull(this.f5815f);
        }
        if (z2) {
            Objects.requireNonNull(this.f5815f);
        }
        if (cVar2 != null) {
            return cVar2;
        }
        if (e0Var != null || ((aVar = this.f5811b) != null && aVar.a())) {
            z3 = false;
        } else {
            f fVar = this.f5817h;
            if (fVar.b()) {
                ArrayList arrayList = new ArrayList();
                while (fVar.c()) {
                    if (fVar.c()) {
                        List<Proxy> list = fVar.f5804e;
                        int i6 = fVar.f5805f;
                        fVar.f5805f = i6 + 1;
                        Proxy proxy = list.get(i6);
                        fVar.f5806g = new ArrayList();
                        if (proxy.type() != Proxy.Type.DIRECT && proxy.type() != Proxy.Type.SOCKS) {
                            SocketAddress address = proxy.address();
                            if (address instanceof InetSocketAddress) {
                                InetSocketAddress inetSocketAddress = (InetSocketAddress) address;
                                InetAddress address2 = inetSocketAddress.getAddress();
                                if (address2 == null) {
                                    str = inetSocketAddress.getHostName();
                                } else {
                                    str = address2.getHostAddress();
                                }
                                i5 = inetSocketAddress.getPort();
                            } else {
                                StringBuilder x = c.b.a.a.a.x("Proxy.address() is not an InetSocketAddress: ");
                                x.append(address.getClass());
                                throw new IllegalArgumentException(x.toString());
                            }
                        } else {
                            r rVar = fVar.f5800a.f5715a;
                            str = rVar.f6090e;
                            i5 = rVar.f6091f;
                        }
                        if (i5 >= 1 && i5 <= 65535) {
                            if (proxy.type() == Proxy.Type.SOCKS) {
                                fVar.f5806g.add(InetSocketAddress.createUnresolved(str, i5));
                            } else {
                                Objects.requireNonNull(fVar.f5803d);
                                Objects.requireNonNull((m.a) fVar.f5800a.f5716b);
                                if (str != null) {
                                    try {
                                        List asList = Arrays.asList(InetAddress.getAllByName(str));
                                        if (!asList.isEmpty()) {
                                            Objects.requireNonNull(fVar.f5803d);
                                            int size = asList.size();
                                            for (int i7 = 0; i7 < size; i7++) {
                                                fVar.f5806g.add(new InetSocketAddress((InetAddress) asList.get(i7), i5));
                                            }
                                        } else {
                                            throw new UnknownHostException(fVar.f5800a.f5716b + " returned no addresses for " + str);
                                        }
                                    } catch (NullPointerException e2) {
                                        UnknownHostException unknownHostException = new UnknownHostException(c.b.a.a.a.q("Broken system behaviour for dns lookup of ", str));
                                        unknownHostException.initCause(e2);
                                        throw unknownHostException;
                                    }
                                } else {
                                    throw new UnknownHostException("hostname == null");
                                }
                            }
                            int size2 = fVar.f5806g.size();
                            for (int i8 = 0; i8 < size2; i8++) {
                                e0 e0Var2 = new e0(fVar.f5800a, proxy, fVar.f5806g.get(i8));
                                d dVar = fVar.f5801b;
                                synchronized (dVar) {
                                    contains = dVar.f5797a.contains(e0Var2);
                                }
                                if (contains) {
                                    fVar.f5807h.add(e0Var2);
                                } else {
                                    arrayList.add(e0Var2);
                                }
                            }
                            if (!arrayList.isEmpty()) {
                                break;
                            }
                        } else {
                            throw new SocketException("No route to " + str + ":" + i5 + "; port is out of range");
                        }
                    } else {
                        StringBuilder x2 = c.b.a.a.a.x("No route to ");
                        x2.append(fVar.f5800a.f5715a.f6090e);
                        x2.append("; exhausted proxy configurations: ");
                        x2.append(fVar.f5804e);
                        throw new SocketException(x2.toString());
                    }
                }
                if (arrayList.isEmpty()) {
                    arrayList.addAll(fVar.f5807h);
                    fVar.f5807h.clear();
                }
                this.f5811b = new f.a(arrayList);
                z3 = true;
            } else {
                throw new NoSuchElementException();
            }
        }
        synchronized (this.f5813d) {
            if (this.m) {
                throw new IOException("Canceled");
            }
            if (z3) {
                f.a aVar2 = this.f5811b;
                Objects.requireNonNull(aVar2);
                ArrayList arrayList2 = new ArrayList(aVar2.f5808a);
                int size3 = arrayList2.size();
                int i9 = 0;
                while (true) {
                    if (i9 >= size3) {
                        break;
                    }
                    e0 e0Var3 = (e0) arrayList2.get(i9);
                    f.g0.a.f5771a.c(this.f5813d, this.f5810a, this, e0Var3);
                    c cVar4 = this.j;
                    if (cVar4 != null) {
                        this.f5812c = e0Var3;
                        z2 = true;
                        cVar2 = cVar4;
                        break;
                    }
                    i9++;
                }
            }
            if (!z2) {
                if (e0Var == null) {
                    f.a aVar3 = this.f5811b;
                    if (aVar3.a()) {
                        List<e0> list2 = aVar3.f5808a;
                        int i10 = aVar3.f5809b;
                        aVar3.f5809b = i10 + 1;
                        e0Var = list2.get(i10);
                    } else {
                        throw new NoSuchElementException();
                    }
                }
                this.f5812c = e0Var;
                this.i = 0;
                cVar2 = new c(this.f5813d, e0Var);
                a(cVar2, false);
            }
        }
        if (z2) {
            Objects.requireNonNull(this.f5815f);
            return cVar2;
        }
        cVar2.c(i, i2, i3, i4, z, this.f5814e, this.f5815f);
        f.g0.a aVar4 = f.g0.a.f5771a;
        h hVar = this.f5813d;
        Objects.requireNonNull((v.a) aVar4);
        hVar.f6050f.a(cVar2.f5791c);
        synchronized (this.f5813d) {
            this.k = true;
            f.g0.a aVar5 = f.g0.a.f5771a;
            h hVar2 = this.f5813d;
            Objects.requireNonNull((v.a) aVar5);
            if (!hVar2.f6051g) {
                hVar2.f6051g = true;
                h.f6045a.execute(hVar2.f6048d);
            }
            hVar2.f6049e.add(cVar2);
            if (cVar2.h()) {
                socket = f.g0.a.f5771a.b(this.f5813d, this.f5810a, this);
                cVar2 = this.j;
            } else {
                socket = null;
            }
        }
        f.g0.c.g(socket);
        Objects.requireNonNull(this.f5815f);
        return cVar2;
    }

    public final c e(int i, int i2, int i3, int i4, boolean z, boolean z2) {
        boolean z3;
        while (true) {
            c d2 = d(i, i2, i3, i4, z);
            synchronized (this.f5813d) {
                if (d2.l == 0) {
                    return d2;
                }
                boolean z4 = false;
                if (!d2.f5793e.isClosed() && !d2.f5793e.isInputShutdown() && !d2.f5793e.isOutputShutdown()) {
                    f.g0.i.g gVar = d2.f5796h;
                    if (gVar != null) {
                        synchronized (gVar) {
                            z3 = gVar.i;
                        }
                        z4 = !z3;
                    } else {
                        if (z2) {
                            try {
                                int soTimeout = d2.f5793e.getSoTimeout();
                                d2.f5793e.setSoTimeout(1);
                                if (d2.i.f()) {
                                    d2.f5793e.setSoTimeout(soTimeout);
                                } else {
                                    d2.f5793e.setSoTimeout(soTimeout);
                                }
                            } catch (SocketTimeoutException unused) {
                            } catch (IOException unused2) {
                            }
                        }
                        z4 = true;
                    }
                }
                if (z4) {
                    return d2;
                }
                f();
            }
        }
    }

    public void f() {
        c cVar;
        Socket c2;
        synchronized (this.f5813d) {
            cVar = this.j;
            c2 = c(true, false, false);
            if (this.j != null) {
                cVar = null;
            }
        }
        f.g0.c.g(c2);
        if (cVar != null) {
            Objects.requireNonNull(this.f5815f);
        }
    }

    public void g() {
        c cVar;
        Socket c2;
        synchronized (this.f5813d) {
            cVar = this.j;
            c2 = c(false, true, false);
            if (this.j != null) {
                cVar = null;
            }
        }
        f.g0.c.g(c2);
        if (cVar != null) {
            Objects.requireNonNull(this.f5815f);
        }
    }

    public void h(IOException iOException) {
        c cVar;
        boolean z;
        Socket c2;
        synchronized (this.f5813d) {
            cVar = null;
            if (iOException instanceof u) {
                f.g0.i.b bVar = ((u) iOException).f6006b;
                f.g0.i.b bVar2 = f.g0.i.b.REFUSED_STREAM;
                if (bVar == bVar2) {
                    this.i++;
                }
                if (bVar != bVar2 || this.i > 1) {
                    this.f5812c = null;
                    z = true;
                }
                z = false;
            } else {
                c cVar2 = this.j;
                if (cVar2 != null && (!cVar2.h() || (iOException instanceof f.g0.i.a))) {
                    if (this.j.l == 0) {
                        e0 e0Var = this.f5812c;
                        if (e0Var != null && iOException != null) {
                            this.f5817h.a(e0Var, iOException);
                        }
                        this.f5812c = null;
                    }
                    z = true;
                }
                z = false;
            }
            c cVar3 = this.j;
            c2 = c(z, false, true);
            if (this.j == null && this.k) {
                cVar = cVar3;
            }
        }
        f.g0.c.g(c2);
        if (cVar != null) {
            Objects.requireNonNull(this.f5815f);
        }
    }

    public void i(boolean z, f.g0.g.c cVar, long j, IOException iOException) {
        c cVar2;
        Socket c2;
        boolean z2;
        Objects.requireNonNull(this.f5815f);
        synchronized (this.f5813d) {
            if (cVar != null) {
                if (cVar == this.n) {
                    if (!z) {
                        this.j.l++;
                    }
                    cVar2 = this.j;
                    c2 = c(z, false, true);
                    if (this.j != null) {
                        cVar2 = null;
                    }
                    z2 = this.l;
                }
            }
            throw new IllegalStateException("expected " + this.n + " but was " + cVar);
        }
        f.g0.c.g(c2);
        if (cVar2 != null) {
            Objects.requireNonNull(this.f5815f);
        }
        if (iOException != null) {
            Objects.requireNonNull(this.f5815f);
        } else if (z2) {
            Objects.requireNonNull(this.f5815f);
        }
    }

    public String toString() {
        c b2 = b();
        return b2 != null ? b2.toString() : this.f5810a.toString();
    }
}