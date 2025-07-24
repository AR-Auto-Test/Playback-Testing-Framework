package f.g0.g;

import com.google.common.net.HttpHeaders;
import f.b0;
import f.s;
import f.y;
import g.o;
import g.r;
import g.w;
import java.net.ProtocolException;
import java.util.Objects;
import java.util.logging.Logger;

/* compiled from: CallServerInterceptor.java */
/* loaded from: classes2.dex */
public final class b implements s {

    /* renamed from: a  reason: collision with root package name */
    public final boolean f5820a;

    /* compiled from: CallServerInterceptor.java */
    /* loaded from: classes2.dex */
    public static final class a extends g.i {

        /* renamed from: c  reason: collision with root package name */
        public long f5821c;

        public a(w wVar) {
            super(wVar);
        }

        @Override // g.w
        public void l(g.e eVar, long j) {
            this.f6183b.l(eVar, j);
            this.f5821c += j;
        }
    }

    public b(boolean z) {
        this.f5820a = z;
    }

    /* JADX WARN: Code restructure failed: missing block: B:34:0x010d, code lost:
        if ("close".equalsIgnoreCase(r0 != null ? r0 : null) != false) goto L38;
     */
    @Override // f.s
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public b0 a(s.a aVar) {
        b0.a aVar2;
        b0 a2;
        f fVar = (f) aVar;
        c cVar = fVar.f5827c;
        f.g0.f.g gVar = fVar.f5826b;
        f.g0.f.c cVar2 = fVar.f5828d;
        y yVar = fVar.f5830f;
        long currentTimeMillis = System.currentTimeMillis();
        Objects.requireNonNull(fVar.f5832h);
        cVar.b(yVar);
        Objects.requireNonNull(fVar.f5832h);
        if (!b.v.u.c.w(yVar.f6151b) || yVar.f6153d == null) {
            aVar2 = null;
        } else {
            if ("100-continue".equalsIgnoreCase(yVar.f6152c.a(HttpHeaders.EXPECT))) {
                cVar.e();
                Objects.requireNonNull(fVar.f5832h);
                aVar2 = cVar.d(true);
            } else {
                aVar2 = null;
            }
            if (aVar2 == null) {
                Objects.requireNonNull(fVar.f5832h);
                a aVar3 = new a(cVar.f(yVar, yVar.f6153d.a()));
                Logger logger = o.f6197a;
                r rVar = new r(aVar3);
                yVar.f6153d.c(rVar);
                rVar.close();
                Objects.requireNonNull(fVar.f5832h);
            } else if (!cVar2.h()) {
                gVar.f();
            }
        }
        cVar.a();
        if (aVar2 == null) {
            Objects.requireNonNull(fVar.f5832h);
            aVar2 = cVar.d(false);
        }
        aVar2.f5731a = yVar;
        aVar2.f5735e = gVar.b().f5794f;
        aVar2.k = currentTimeMillis;
        aVar2.l = System.currentTimeMillis();
        b0 a3 = aVar2.a();
        int i = a3.f5726d;
        if (i == 100) {
            b0.a d2 = cVar.d(false);
            d2.f5731a = yVar;
            d2.f5735e = gVar.b().f5794f;
            d2.k = currentTimeMillis;
            d2.l = System.currentTimeMillis();
            a3 = d2.a();
            i = a3.f5726d;
        }
        Objects.requireNonNull(fVar.f5832h);
        if (this.f5820a && i == 101) {
            b0.a aVar4 = new b0.a(a3);
            aVar4.f5737g = f.g0.c.f5775c;
            a2 = aVar4.a();
        } else {
            b0.a aVar5 = new b0.a(a3);
            aVar5.f5737g = cVar.c(a3);
            a2 = aVar5.a();
        }
        if (!"close".equalsIgnoreCase(a2.f5724b.f6152c.a(HttpHeaders.CONNECTION))) {
            String a4 = a2.f5729g.a(HttpHeaders.CONNECTION);
        }
        gVar.f();
        if ((i == 204 || i == 205) && a2.f5730h.C() > 0) {
            StringBuilder y = c.b.a.a.a.y("HTTP ", i, " had non-zero Content-Length: ");
            y.append(a2.f5730h.C());
            throw new ProtocolException(y.toString());
        }
        return a2;
    }
}