package f.g0.g;

import com.google.common.net.HttpHeaders;
import f.a0;
import f.b0;
import f.j;
import f.k;
import f.q;
import f.s;
import f.t;
import f.y;
import g.l;
import g.o;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

/* compiled from: BridgeInterceptor.java */
/* loaded from: classes2.dex */
public final class a implements s {

    /* renamed from: a  reason: collision with root package name */
    public final k f5819a;

    public a(k kVar) {
        this.f5819a = kVar;
    }

    @Override // f.s
    public b0 a(s.a aVar) {
        boolean z;
        f fVar = (f) aVar;
        y yVar = fVar.f5830f;
        y.a aVar2 = new y.a(yVar);
        a0 a0Var = yVar.f6153d;
        if (a0Var != null) {
            t b2 = a0Var.b();
            if (b2 != null) {
                aVar2.b(HttpHeaders.CONTENT_TYPE, b2.f6104c);
            }
            long a2 = a0Var.a();
            if (a2 != -1) {
                aVar2.b(HttpHeaders.CONTENT_LENGTH, Long.toString(a2));
                aVar2.f6158c.c(HttpHeaders.TRANSFER_ENCODING);
            } else {
                q.a aVar3 = aVar2.f6158c;
                aVar3.b(HttpHeaders.TRANSFER_ENCODING, "chunked");
                aVar3.c(HttpHeaders.TRANSFER_ENCODING);
                aVar3.f6085a.add(HttpHeaders.TRANSFER_ENCODING);
                aVar3.f6085a.add("chunked");
                aVar2.f6158c.c(HttpHeaders.CONTENT_LENGTH);
            }
        }
        if (yVar.f6152c.a(HttpHeaders.HOST) == null) {
            aVar2.b(HttpHeaders.HOST, f.g0.c.o(yVar.f6150a, false));
        }
        if (yVar.f6152c.a(HttpHeaders.CONNECTION) == null) {
            q.a aVar4 = aVar2.f6158c;
            aVar4.b(HttpHeaders.CONNECTION, "Keep-Alive");
            aVar4.c(HttpHeaders.CONNECTION);
            aVar4.f6085a.add(HttpHeaders.CONNECTION);
            aVar4.f6085a.add("Keep-Alive");
        }
        if (yVar.f6152c.a(HttpHeaders.ACCEPT_ENCODING) == null && yVar.f6152c.a(HttpHeaders.RANGE) == null) {
            q.a aVar5 = aVar2.f6158c;
            aVar5.b(HttpHeaders.ACCEPT_ENCODING, "gzip");
            aVar5.c(HttpHeaders.ACCEPT_ENCODING);
            aVar5.f6085a.add(HttpHeaders.ACCEPT_ENCODING);
            aVar5.f6085a.add("gzip");
            z = true;
        } else {
            z = false;
        }
        Objects.requireNonNull((k.a) this.f5819a);
        List emptyList = Collections.emptyList();
        if (!emptyList.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            int size = emptyList.size();
            for (int i = 0; i < size; i++) {
                if (i > 0) {
                    sb.append("; ");
                }
                j jVar = (j) emptyList.get(i);
                sb.append(jVar.f6068e);
                sb.append('=');
                sb.append(jVar.f6069f);
            }
            aVar2.b(HttpHeaders.COOKIE, sb.toString());
        }
        if (yVar.f6152c.a("User-Agent") == null) {
            q.a aVar6 = aVar2.f6158c;
            aVar6.b("User-Agent", "okhttp/3.10.0");
            aVar6.c("User-Agent");
            aVar6.f6085a.add("User-Agent");
            aVar6.f6085a.add("okhttp/3.10.0");
        }
        b0 b3 = fVar.b(aVar2.a(), fVar.f5826b, fVar.f5827c, fVar.f5828d);
        e.d(this.f5819a, yVar.f6150a, b3.f5729g);
        b0.a aVar7 = new b0.a(b3);
        aVar7.f5731a = yVar;
        if (z) {
            String a3 = b3.f5729g.a(HttpHeaders.CONTENT_ENCODING);
            if (a3 == null) {
                a3 = null;
            }
            if ("gzip".equalsIgnoreCase(a3) && e.b(b3)) {
                l lVar = new l(b3.f5730h.E());
                q.a c2 = b3.f5729g.c();
                c2.c(HttpHeaders.CONTENT_ENCODING);
                c2.c(HttpHeaders.CONTENT_LENGTH);
                List<String> list = c2.f6085a;
                q.a aVar8 = new q.a();
                Collections.addAll(aVar8.f6085a, (String[]) list.toArray(new String[list.size()]));
                aVar7.f5736f = aVar8;
                String a4 = b3.f5729g.a(HttpHeaders.CONTENT_TYPE);
                String str = a4 != null ? a4 : null;
                Logger logger = o.f6197a;
                aVar7.f5737g = new g(str, -1L, new g.s(lVar));
            }
        }
        return aVar7.a();
    }
}