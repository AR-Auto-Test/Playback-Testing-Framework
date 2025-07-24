package c.e.b;

import com.google.common.net.HttpHeaders;
import f.s;
import f.y;
import java.nio.charset.Charset;

/* compiled from: BasicAuthInterceptor.java */
/* loaded from: classes2.dex */
public class xb implements f.s {

    /* renamed from: a  reason: collision with root package name */
    public String f5411a;

    public xb(String str, String str2) {
        Charset charset = f.g0.c.j;
        String r = c.b.a.a.a.r(str, ":", str2);
        char[] cArr = g.h.f6178b;
        if (r == null) {
            throw new IllegalArgumentException("s == null");
        }
        if (charset != null) {
            this.f5411a = c.b.a.a.a.q("Basic ", new g.h(r.getBytes(charset)).a());
            return;
        }
        throw new IllegalArgumentException("charset == null");
    }

    @Override // f.s
    public f.b0 a(s.a aVar) {
        f.g0.g.f fVar = (f.g0.g.f) aVar;
        y.a aVar2 = new y.a(fVar.f5830f);
        aVar2.b(HttpHeaders.AUTHORIZATION, this.f5411a);
        return fVar.b(aVar2.a(), fVar.f5826b, fVar.f5827c, fVar.f5828d);
    }
}