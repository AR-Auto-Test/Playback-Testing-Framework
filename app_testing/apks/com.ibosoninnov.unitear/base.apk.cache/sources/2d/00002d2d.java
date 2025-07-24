package f.g0.i;

import com.google.common.primitives.UnsignedBytes;
import f.g0.i.d;
import f.g0.i.g;
import f.g0.i.p;
import g.x;
import g.y;
import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.RejectedExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;

/* compiled from: Http2Reader.java */
/* loaded from: classes2.dex */
public final class o implements Closeable {

    /* renamed from: b  reason: collision with root package name */
    public static final Logger f5960b = Logger.getLogger(e.class.getName());

    /* renamed from: c  reason: collision with root package name */
    public final g.g f5961c;

    /* renamed from: d  reason: collision with root package name */
    public final a f5962d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f5963e;

    /* renamed from: f  reason: collision with root package name */
    public final d.a f5964f;

    /* compiled from: Http2Reader.java */
    /* loaded from: classes2.dex */
    public static final class a implements x {

        /* renamed from: b  reason: collision with root package name */
        public final g.g f5965b;

        /* renamed from: c  reason: collision with root package name */
        public int f5966c;

        /* renamed from: d  reason: collision with root package name */
        public byte f5967d;

        /* renamed from: e  reason: collision with root package name */
        public int f5968e;

        /* renamed from: f  reason: collision with root package name */
        public int f5969f;

        /* renamed from: g  reason: collision with root package name */
        public short f5970g;

        public a(g.g gVar) {
            this.f5965b = gVar;
        }

        @Override // g.x
        public y b() {
            return this.f5965b.b();
        }

        @Override // g.x, java.io.Closeable, java.lang.AutoCloseable
        public void close() {
        }

        @Override // g.x
        public long u(g.e eVar, long j) {
            int i;
            int readInt;
            do {
                int i2 = this.f5969f;
                if (i2 == 0) {
                    this.f5965b.c(this.f5970g);
                    this.f5970g = (short) 0;
                    if ((this.f5967d & 4) != 0) {
                        return -1L;
                    }
                    i = this.f5968e;
                    int G = o.G(this.f5965b);
                    this.f5969f = G;
                    this.f5966c = G;
                    byte readByte = (byte) (this.f5965b.readByte() & UnsignedBytes.MAX_VALUE);
                    this.f5967d = (byte) (this.f5965b.readByte() & UnsignedBytes.MAX_VALUE);
                    Logger logger = o.f5960b;
                    if (logger.isLoggable(Level.FINE)) {
                        logger.fine(e.a(true, this.f5968e, this.f5966c, readByte, this.f5967d));
                    }
                    readInt = this.f5965b.readInt() & Integer.MAX_VALUE;
                    this.f5968e = readInt;
                    if (readByte != 9) {
                        e.c("%s != TYPE_CONTINUATION", Byte.valueOf(readByte));
                        throw null;
                    }
                } else {
                    long u = this.f5965b.u(eVar, Math.min(j, i2));
                    if (u == -1) {
                        return -1L;
                    }
                    this.f5969f = (int) (this.f5969f - u);
                    return u;
                }
            } while (readInt == i);
            e.c("TYPE_CONTINUATION streamId changed", new Object[0]);
            throw null;
        }
    }

    /* compiled from: Http2Reader.java */
    /* loaded from: classes2.dex */
    public interface b {
    }

    public o(g.g gVar, boolean z) {
        this.f5961c = gVar;
        this.f5963e = z;
        a aVar = new a(gVar);
        this.f5962d = aVar;
        this.f5964f = new d.a(4096, aVar);
    }

    public static int B(int i, byte b2, short s) {
        if ((b2 & 8) != 0) {
            i--;
        }
        if (s <= i) {
            return (short) (i - s);
        }
        e.c("PROTOCOL_ERROR padding %s > remaining length %s", Short.valueOf(s), Integer.valueOf(i));
        throw null;
    }

    public static int G(g.g gVar) {
        return (gVar.readByte() & UnsignedBytes.MAX_VALUE) | ((gVar.readByte() & UnsignedBytes.MAX_VALUE) << 16) | ((gVar.readByte() & UnsignedBytes.MAX_VALUE) << 8);
    }

    public boolean C(boolean z, b bVar) {
        boolean z2;
        boolean z3;
        boolean z4;
        try {
            this.f5961c.v(9L);
            int G = G(this.f5961c);
            if (G < 0 || G > 16384) {
                e.c("FRAME_SIZE_ERROR: %s", Integer.valueOf(G));
                throw null;
            }
            byte readByte = (byte) (this.f5961c.readByte() & UnsignedBytes.MAX_VALUE);
            if (z && readByte != 4) {
                e.c("Expected a SETTINGS frame but was %s", Byte.valueOf(readByte));
                throw null;
            }
            byte readByte2 = (byte) (this.f5961c.readByte() & UnsignedBytes.MAX_VALUE);
            int readInt = this.f5961c.readInt() & Integer.MAX_VALUE;
            Logger logger = f5960b;
            if (logger.isLoggable(Level.FINE)) {
                logger.fine(e.a(true, readInt, G, readByte, readByte2));
            }
            switch (readByte) {
                case 0:
                    if (readInt == 0) {
                        e.c("PROTOCOL_ERROR: TYPE_DATA streamId == 0", new Object[0]);
                        throw null;
                    }
                    boolean z5 = (readByte2 & 1) != 0;
                    if (!((readByte2 & 32) != 0)) {
                        short readByte3 = (readByte2 & 8) != 0 ? (short) (this.f5961c.readByte() & UnsignedBytes.MAX_VALUE) : (short) 0;
                        int B = B(G, readByte2, readByte3);
                        g.g gVar = this.f5961c;
                        g.f fVar = (g.f) bVar;
                        if (g.this.F(readInt)) {
                            g gVar2 = g.this;
                            Objects.requireNonNull(gVar2);
                            g.e eVar = new g.e();
                            long j = B;
                            gVar.v(j);
                            gVar.u(eVar, j);
                            if (eVar.f6176d == j) {
                                gVar2.k.execute(new j(gVar2, "OkHttp %s Push Data[%s]", new Object[]{gVar2.f5918f, Integer.valueOf(readInt)}, readInt, eVar, B, z5));
                            } else {
                                throw new IOException(eVar.f6176d + " != " + B);
                            }
                        } else {
                            p D = g.this.D(readInt);
                            if (D == null) {
                                g.this.J(readInt, f.g0.i.b.PROTOCOL_ERROR);
                                gVar.c(B);
                            } else {
                                p.b bVar2 = D.f5978h;
                                long j2 = B;
                                Objects.requireNonNull(bVar2);
                                while (true) {
                                    if (j2 > 0) {
                                        synchronized (p.this) {
                                            z2 = bVar2.f5987f;
                                            z3 = bVar2.f5984c.f6176d + j2 > bVar2.f5985d;
                                        }
                                        if (z3) {
                                            gVar.c(j2);
                                            p.this.e(f.g0.i.b.FLOW_CONTROL_ERROR);
                                        } else if (z2) {
                                            gVar.c(j2);
                                        } else {
                                            long u = gVar.u(bVar2.f5983b, j2);
                                            if (u != -1) {
                                                j2 -= u;
                                                synchronized (p.this) {
                                                    g.e eVar2 = bVar2.f5984c;
                                                    boolean z6 = eVar2.f6176d == 0;
                                                    eVar2.S(bVar2.f5983b);
                                                    if (z6) {
                                                        p.this.notifyAll();
                                                    }
                                                }
                                            } else {
                                                throw new EOFException();
                                            }
                                        }
                                    }
                                }
                                if (z5) {
                                    D.i();
                                }
                            }
                        }
                        this.f5961c.c(readByte3);
                        break;
                    } else {
                        e.c("PROTOCOL_ERROR: FLAG_COMPRESSED without SETTINGS_COMPRESS_DATA", new Object[0]);
                        throw null;
                    }
                case 1:
                    if (readInt != 0) {
                        boolean z7 = (readByte2 & 1) != 0;
                        short readByte4 = (readByte2 & 8) != 0 ? (short) (this.f5961c.readByte() & UnsignedBytes.MAX_VALUE) : (short) 0;
                        if ((readByte2 & 32) != 0) {
                            this.f5961c.readInt();
                            this.f5961c.readByte();
                            Objects.requireNonNull((g.f) bVar);
                            G -= 5;
                        }
                        List<c> F = F(B(G, readByte2, readByte4), readByte4, readByte2, readInt);
                        g.f fVar2 = (g.f) bVar;
                        if (g.this.F(readInt)) {
                            g gVar3 = g.this;
                            Objects.requireNonNull(gVar3);
                            try {
                                gVar3.k.execute(new i(gVar3, "OkHttp %s Push Headers[%s]", new Object[]{gVar3.f5918f, Integer.valueOf(readInt)}, readInt, F, z7));
                                break;
                            } catch (RejectedExecutionException unused) {
                                break;
                            }
                        } else {
                            synchronized (g.this) {
                                p D2 = g.this.D(readInt);
                                if (D2 == null) {
                                    g gVar4 = g.this;
                                    if (!gVar4.i) {
                                        if (readInt > gVar4.f5919g) {
                                            if (readInt % 2 != gVar4.f5920h % 2) {
                                                p pVar = new p(readInt, gVar4, false, z7, F);
                                                g gVar5 = g.this;
                                                gVar5.f5919g = readInt;
                                                gVar5.f5917e.put(Integer.valueOf(readInt), pVar);
                                                g.f5914b.execute(new l(fVar2, "OkHttp %s stream %d", new Object[]{g.this.f5918f, Integer.valueOf(readInt)}, pVar));
                                            }
                                        }
                                    }
                                } else {
                                    synchronized (D2) {
                                        D2.f5977g = true;
                                        if (D2.f5976f == null) {
                                            D2.f5976f = F;
                                            z4 = D2.h();
                                            D2.notifyAll();
                                        } else {
                                            ArrayList arrayList = new ArrayList();
                                            arrayList.addAll(D2.f5976f);
                                            arrayList.add(null);
                                            arrayList.addAll(F);
                                            D2.f5976f = arrayList;
                                            z4 = true;
                                        }
                                    }
                                    if (!z4) {
                                        D2.f5974d.G(D2.f5973c);
                                    }
                                    if (z7) {
                                        D2.i();
                                    }
                                }
                            }
                            break;
                        }
                    } else {
                        e.c("PROTOCOL_ERROR: TYPE_HEADERS streamId == 0", new Object[0]);
                        throw null;
                    }
                case 2:
                    if (G != 5) {
                        e.c("TYPE_PRIORITY length: %d != 5", Integer.valueOf(G));
                        throw null;
                    } else if (readInt != 0) {
                        this.f5961c.readInt();
                        this.f5961c.readByte();
                        Objects.requireNonNull((g.f) bVar);
                        break;
                    } else {
                        e.c("TYPE_PRIORITY streamId == 0", new Object[0]);
                        throw null;
                    }
                case 3:
                    J(bVar, G, readInt);
                    break;
                case 4:
                    K(bVar, G, readByte2, readInt);
                    break;
                case 5:
                    I(bVar, G, readByte2, readInt);
                    break;
                case 6:
                    H(bVar, G, readByte2, readInt);
                    break;
                case 7:
                    E(bVar, G, readInt);
                    break;
                case 8:
                    L(bVar, G, readInt);
                    break;
                default:
                    this.f5961c.c(G);
                    break;
            }
            return true;
        } catch (IOException unused2) {
            return false;
        }
    }

    public void D(b bVar) {
        if (this.f5963e) {
            if (C(true, bVar)) {
                return;
            }
            e.c("Required SETTINGS preface not received", new Object[0]);
            throw null;
        }
        g.g gVar = this.f5961c;
        g.h hVar = e.f5899a;
        g.h d2 = gVar.d(hVar.l());
        Logger logger = f5960b;
        if (logger.isLoggable(Level.FINE)) {
            logger.fine(f.g0.c.n("<< CONNECTION %s", d2.g()));
        }
        if (hVar.equals(d2)) {
            return;
        }
        e.c("Expected a connection header but was %s", d2.p());
        throw null;
    }

    public final void E(b bVar, int i, int i2) {
        p[] pVarArr;
        if (i < 8) {
            e.c("TYPE_GOAWAY length < 8: %s", Integer.valueOf(i));
            throw null;
        } else if (i2 == 0) {
            int readInt = this.f5961c.readInt();
            int readInt2 = this.f5961c.readInt();
            int i3 = i - 8;
            if (f.g0.i.b.a(readInt2) == null) {
                e.c("TYPE_GOAWAY unexpected error code: %d", Integer.valueOf(readInt2));
                throw null;
            }
            g.h hVar = g.h.f6179c;
            if (i3 > 0) {
                hVar = this.f5961c.d(i3);
            }
            g.f fVar = (g.f) bVar;
            Objects.requireNonNull(fVar);
            hVar.l();
            synchronized (g.this) {
                pVarArr = (p[]) g.this.f5917e.values().toArray(new p[g.this.f5917e.size()]);
                g.this.i = true;
            }
            for (p pVar : pVarArr) {
                if (pVar.f5973c > readInt && pVar.g()) {
                    f.g0.i.b bVar2 = f.g0.i.b.REFUSED_STREAM;
                    synchronized (pVar) {
                        if (pVar.l == null) {
                            pVar.l = bVar2;
                            pVar.notifyAll();
                        }
                    }
                    g.this.G(pVar.f5973c);
                }
            }
        } else {
            e.c("TYPE_GOAWAY streamId != 0", new Object[0]);
            throw null;
        }
    }

    public final List<c> F(int i, short s, byte b2, int i2) {
        a aVar = this.f5962d;
        aVar.f5969f = i;
        aVar.f5966c = i;
        aVar.f5970g = s;
        aVar.f5967d = b2;
        aVar.f5968e = i2;
        d.a aVar2 = this.f5964f;
        while (!aVar2.f5884b.f()) {
            int readByte = aVar2.f5884b.readByte() & UnsignedBytes.MAX_VALUE;
            if (readByte == 128) {
                throw new IOException("index == 0");
            }
            if ((readByte & 128) == 128) {
                int g2 = aVar2.g(readByte, 127) - 1;
                if (g2 >= 0 && g2 <= d.f5881a.length - 1) {
                    aVar2.f5883a.add(d.f5881a[g2]);
                } else {
                    int b3 = aVar2.b(g2 - d.f5881a.length);
                    if (b3 >= 0) {
                        c[] cVarArr = aVar2.f5887e;
                        if (b3 < cVarArr.length) {
                            aVar2.f5883a.add(cVarArr[b3]);
                        }
                    }
                    StringBuilder x = c.b.a.a.a.x("Header index too large ");
                    x.append(g2 + 1);
                    throw new IOException(x.toString());
                }
            } else if (readByte == 64) {
                g.h f2 = aVar2.f();
                d.a(f2);
                aVar2.e(-1, new c(f2, aVar2.f()));
            } else if ((readByte & 64) == 64) {
                aVar2.e(-1, new c(aVar2.d(aVar2.g(readByte, 63) - 1), aVar2.f()));
            } else if ((readByte & 32) == 32) {
                int g3 = aVar2.g(readByte, 31);
                aVar2.f5886d = g3;
                if (g3 >= 0 && g3 <= aVar2.f5885c) {
                    int i3 = aVar2.f5890h;
                    if (g3 < i3) {
                        if (g3 == 0) {
                            aVar2.a();
                        } else {
                            aVar2.c(i3 - g3);
                        }
                    }
                } else {
                    StringBuilder x2 = c.b.a.a.a.x("Invalid dynamic table size update ");
                    x2.append(aVar2.f5886d);
                    throw new IOException(x2.toString());
                }
            } else if (readByte != 16 && readByte != 0) {
                aVar2.f5883a.add(new c(aVar2.d(aVar2.g(readByte, 15) - 1), aVar2.f()));
            } else {
                g.h f3 = aVar2.f();
                d.a(f3);
                aVar2.f5883a.add(new c(f3, aVar2.f()));
            }
        }
        d.a aVar3 = this.f5964f;
        Objects.requireNonNull(aVar3);
        ArrayList arrayList = new ArrayList(aVar3.f5883a);
        aVar3.f5883a.clear();
        return arrayList;
    }

    public final void H(b bVar, int i, byte b2, int i2) {
        if (i != 8) {
            e.c("TYPE_PING length != 8: %s", Integer.valueOf(i));
            throw null;
        } else if (i2 == 0) {
            int readInt = this.f5961c.readInt();
            int readInt2 = this.f5961c.readInt();
            boolean z = (b2 & 1) != 0;
            g.f fVar = (g.f) bVar;
            Objects.requireNonNull(fVar);
            if (z) {
                synchronized (g.this) {
                    g gVar = g.this;
                    gVar.m = false;
                    gVar.notifyAll();
                }
                return;
            }
            try {
                g gVar2 = g.this;
                gVar2.j.execute(new g.e(true, readInt, readInt2));
            } catch (RejectedExecutionException unused) {
            }
        } else {
            e.c("TYPE_PING streamId != 0", new Object[0]);
            throw null;
        }
    }

    public final void I(b bVar, int i, byte b2, int i2) {
        if (i2 != 0) {
            short readByte = (b2 & 8) != 0 ? (short) (this.f5961c.readByte() & UnsignedBytes.MAX_VALUE) : (short) 0;
            int readInt = this.f5961c.readInt() & Integer.MAX_VALUE;
            List<c> F = F(B(i - 4, b2, readByte), readByte, b2, i2);
            g gVar = g.this;
            synchronized (gVar) {
                if (gVar.v.contains(Integer.valueOf(readInt))) {
                    gVar.J(readInt, f.g0.i.b.PROTOCOL_ERROR);
                    return;
                }
                gVar.v.add(Integer.valueOf(readInt));
                try {
                    gVar.k.execute(new h(gVar, "OkHttp %s Push Request[%s]", new Object[]{gVar.f5918f, Integer.valueOf(readInt)}, readInt, F));
                    return;
                } catch (RejectedExecutionException unused) {
                    return;
                }
            }
        }
        e.c("PROTOCOL_ERROR: TYPE_PUSH_PROMISE streamId == 0", new Object[0]);
        throw null;
    }

    public final void J(b bVar, int i, int i2) {
        if (i != 4) {
            e.c("TYPE_RST_STREAM length: %d != 4", Integer.valueOf(i));
            throw null;
        } else if (i2 != 0) {
            int readInt = this.f5961c.readInt();
            f.g0.i.b a2 = f.g0.i.b.a(readInt);
            if (a2 == null) {
                e.c("TYPE_RST_STREAM unexpected error code: %d", Integer.valueOf(readInt));
                throw null;
            }
            g.f fVar = (g.f) bVar;
            if (g.this.F(i2)) {
                g gVar = g.this;
                gVar.k.execute(new k(gVar, "OkHttp %s Push Reset[%s]", new Object[]{gVar.f5918f, Integer.valueOf(i2)}, i2, a2));
                return;
            }
            p G = g.this.G(i2);
            if (G != null) {
                synchronized (G) {
                    if (G.l == null) {
                        G.l = a2;
                        G.notifyAll();
                    }
                }
            }
        } else {
            e.c("TYPE_RST_STREAM streamId == 0", new Object[0]);
            throw null;
        }
    }

    public final void K(b bVar, int i, byte b2, int i2) {
        long j;
        int i3;
        p[] pVarArr = null;
        if (i2 != 0) {
            e.c("TYPE_SETTINGS streamId != 0", new Object[0]);
            throw null;
        } else if ((b2 & 1) != 0) {
            if (i == 0) {
                Objects.requireNonNull((g.f) bVar);
            } else {
                e.c("FRAME_SIZE_ERROR ack frame should be empty!", new Object[0]);
                throw null;
            }
        } else if (i % 6 != 0) {
            e.c("TYPE_SETTINGS length %% 6 != 0: %s", Integer.valueOf(i));
            throw null;
        } else {
            t tVar = new t();
            for (int i4 = 0; i4 < i; i4 += 6) {
                int readShort = this.f5961c.readShort() & 65535;
                int readInt = this.f5961c.readInt();
                if (readShort != 2) {
                    if (readShort == 3) {
                        readShort = 4;
                    } else if (readShort == 4) {
                        readShort = 7;
                        if (readInt < 0) {
                            e.c("PROTOCOL_ERROR SETTINGS_INITIAL_WINDOW_SIZE > 2^31 - 1", new Object[0]);
                            throw null;
                        }
                    } else if (readShort == 5 && (readInt < 16384 || readInt > 16777215)) {
                        e.c("PROTOCOL_ERROR SETTINGS_MAX_FRAME_SIZE: %s", Integer.valueOf(readInt));
                        throw null;
                    }
                } else if (readInt != 0 && readInt != 1) {
                    e.c("PROTOCOL_ERROR SETTINGS_ENABLE_PUSH != 0 or 1", new Object[0]);
                    throw null;
                }
                tVar.b(readShort, readInt);
            }
            g.f fVar = (g.f) bVar;
            synchronized (g.this) {
                int a2 = g.this.q.a();
                t tVar2 = g.this.q;
                Objects.requireNonNull(tVar2);
                for (int i5 = 0; i5 < 10; i5++) {
                    if (((1 << i5) & tVar.f6004a) != 0) {
                        tVar2.b(i5, tVar.f6005b[i5]);
                    }
                }
                try {
                    g gVar = g.this;
                    gVar.j.execute(new n(fVar, "OkHttp %s ACK Settings", new Object[]{gVar.f5918f}, tVar));
                } catch (RejectedExecutionException unused) {
                }
                int a3 = g.this.q.a();
                if (a3 == -1 || a3 == a2) {
                    j = 0;
                } else {
                    j = a3 - a2;
                    g gVar2 = g.this;
                    if (!gVar2.r) {
                        gVar2.o += j;
                        if (j > 0) {
                            gVar2.notifyAll();
                        }
                        g.this.r = true;
                    }
                    if (!g.this.f5917e.isEmpty()) {
                        pVarArr = (p[]) g.this.f5917e.values().toArray(new p[g.this.f5917e.size()]);
                    }
                }
                g.f5914b.execute(new m(fVar, "OkHttp %s settings", g.this.f5918f));
            }
            if (pVarArr == null || j == 0) {
                return;
            }
            for (p pVar : pVarArr) {
                synchronized (pVar) {
                    pVar.f5972b += j;
                    if (i3 > 0) {
                        pVar.notifyAll();
                    }
                }
            }
        }
    }

    public final void L(b bVar, int i, int i2) {
        if (i != 4) {
            e.c("TYPE_WINDOW_UPDATE length !=4: %s", Integer.valueOf(i));
            throw null;
        }
        long readInt = this.f5961c.readInt() & 2147483647L;
        int i3 = (readInt > 0L ? 1 : (readInt == 0L ? 0 : -1));
        if (i3 == 0) {
            e.c("windowSizeIncrement was 0", Long.valueOf(readInt));
            throw null;
        }
        g.f fVar = (g.f) bVar;
        if (i2 == 0) {
            synchronized (g.this) {
                g gVar = g.this;
                gVar.o += readInt;
                gVar.notifyAll();
            }
            return;
        }
        p D = g.this.D(i2);
        if (D != null) {
            synchronized (D) {
                D.f5972b += readInt;
                if (i3 > 0) {
                    D.notifyAll();
                }
            }
        }
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public void close() {
        this.f5961c.close();
    }
}