package f.g0.e;

import com.google.common.net.HttpHeaders;
import f.b0;
import f.g0.g.f;
import f.q;
import f.s;
import f.w;
import f.y;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;

/* compiled from: CacheInterceptor.java */
/* loaded from: classes2.dex */
public final class a implements s {
    public a(c cVar) {
    }

    public static boolean b(String str) {
        return HttpHeaders.CONTENT_LENGTH.equalsIgnoreCase(str) || HttpHeaders.CONTENT_ENCODING.equalsIgnoreCase(str) || HttpHeaders.CONTENT_TYPE.equalsIgnoreCase(str);
    }

    public static boolean c(String str) {
        return (HttpHeaders.CONNECTION.equalsIgnoreCase(str) || "Keep-Alive".equalsIgnoreCase(str) || HttpHeaders.PROXY_AUTHENTICATE.equalsIgnoreCase(str) || HttpHeaders.PROXY_AUTHORIZATION.equalsIgnoreCase(str) || HttpHeaders.TE.equalsIgnoreCase(str) || "Trailers".equalsIgnoreCase(str) || HttpHeaders.TRANSFER_ENCODING.equalsIgnoreCase(str) || HttpHeaders.UPGRADE.equalsIgnoreCase(str)) ? false : true;
    }

    public static b0 d(b0 b0Var) {
        if (b0Var == null || b0Var.f5730h == null) {
            return b0Var;
        }
        b0.a aVar = new b0.a(b0Var);
        aVar.f5737g = null;
        return aVar.a();
    }

    @Override // f.s
    public b0 a(s.a aVar) {
        System.currentTimeMillis();
        f fVar = (f) aVar;
        y yVar = fVar.f5830f;
        b bVar = new b(yVar, null);
        if (yVar != null && yVar.a().j) {
            bVar = new b(null, null);
        }
        y yVar2 = bVar.f5783a;
        b0 b0Var = bVar.f5784b;
        if (yVar2 == null && b0Var == null) {
            b0.a aVar2 = new b0.a();
            aVar2.f5731a = fVar.f5830f;
            aVar2.f5732b = w.HTTP_1_1;
            aVar2.f5733c = 504;
            aVar2.f5734d = "Unsatisfiable Request (only-if-cached)";
            aVar2.f5737g = f.g0.c.f5775c;
            aVar2.k = -1L;
            aVar2.l = System.currentTimeMillis();
            return aVar2.a();
        } else if (yVar2 == null) {
            Objects.requireNonNull(b0Var);
            b0.a aVar3 = new b0.a(b0Var);
            aVar3.b(d(b0Var));
            return aVar3.a();
        } else {
            f fVar2 = (f) aVar;
            b0 b2 = fVar2.b(yVar2, fVar2.f5826b, fVar2.f5827c, fVar2.f5828d);
            if (b0Var != null) {
                if (b2.f5726d == 304) {
                    b0.a aVar4 = new b0.a(b0Var);
                    q qVar = b0Var.f5729g;
                    q qVar2 = b2.f5729g;
                    ArrayList arrayList = new ArrayList(20);
                    int d2 = qVar.d();
                    for (int i = 0; i < d2; i++) {
                        String b3 = qVar.b(i);
                        String e2 = qVar.e(i);
                        if ((!HttpHeaders.WARNING.equalsIgnoreCase(b3) || !e2.startsWith("1")) && (b(b3) || !c(b3) || qVar2.a(b3) == null)) {
                            arrayList.add(b3);
                            arrayList.add(e2.trim());
                        }
                    }
                    int d3 = qVar2.d();
                    for (int i2 = 0; i2 < d3; i2++) {
                        String b4 = qVar2.b(i2);
                        if (!b(b4) && c(b4)) {
                            String e3 = qVar2.e(i2);
                            arrayList.add(b4);
                            arrayList.add(e3.trim());
                        }
                    }
                    q.a aVar5 = new q.a();
                    Collections.addAll(aVar5.f6085a, (String[]) arrayList.toArray(new String[arrayList.size()]));
                    aVar4.f5736f = aVar5;
                    aVar4.k = b2.l;
                    aVar4.l = b2.m;
                    aVar4.b(d(b0Var));
                    b0 d4 = d(b2);
                    if (d4 != null) {
                        aVar4.c("networkResponse", d4);
                    }
                    aVar4.f5738h = d4;
                    aVar4.a();
                    b2.f5730h.close();
                    throw null;
                }
                f.g0.c.f(b0Var.f5730h);
            }
            b0.a aVar6 = new b0.a(b2);
            aVar6.b(d(b0Var));
            b0 d5 = d(b2);
            if (d5 != null) {
                aVar6.c("networkResponse", d5);
            }
            aVar6.f5738h = d5;
            return aVar6.a();
        }
    }
}