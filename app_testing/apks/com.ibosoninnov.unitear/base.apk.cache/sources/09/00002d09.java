package f.g0.g;

import f.b0;
import f.n;
import f.s;
import f.y;
import java.util.List;

/* compiled from: RealInterceptorChain.java */
/* loaded from: classes2.dex */
public final class f implements s.a {

    /* renamed from: a  reason: collision with root package name */
    public final List<s> f5825a;

    /* renamed from: b  reason: collision with root package name */
    public final f.g0.f.g f5826b;

    /* renamed from: c  reason: collision with root package name */
    public final c f5827c;

    /* renamed from: d  reason: collision with root package name */
    public final f.g0.f.c f5828d;

    /* renamed from: e  reason: collision with root package name */
    public final int f5829e;

    /* renamed from: f  reason: collision with root package name */
    public final y f5830f;

    /* renamed from: g  reason: collision with root package name */
    public final f.d f5831g;

    /* renamed from: h  reason: collision with root package name */
    public final n f5832h;
    public final int i;
    public final int j;
    public final int k;
    public int l;

    public f(List<s> list, f.g0.f.g gVar, c cVar, f.g0.f.c cVar2, int i, y yVar, f.d dVar, n nVar, int i2, int i3, int i4) {
        this.f5825a = list;
        this.f5828d = cVar2;
        this.f5826b = gVar;
        this.f5827c = cVar;
        this.f5829e = i;
        this.f5830f = yVar;
        this.f5831g = dVar;
        this.f5832h = nVar;
        this.i = i2;
        this.j = i3;
        this.k = i4;
    }

    public b0 a(y yVar) {
        return b(yVar, this.f5826b, this.f5827c, this.f5828d);
    }

    public b0 b(y yVar, f.g0.f.g gVar, c cVar, f.g0.f.c cVar2) {
        if (this.f5829e < this.f5825a.size()) {
            this.l++;
            if (this.f5827c != null && !this.f5828d.j(yVar.f6150a)) {
                StringBuilder x = c.b.a.a.a.x("network interceptor ");
                x.append(this.f5825a.get(this.f5829e - 1));
                x.append(" must retain the same host and port");
                throw new IllegalStateException(x.toString());
            }
            if (this.f5827c != null && this.l > 1) {
                StringBuilder x2 = c.b.a.a.a.x("network interceptor ");
                x2.append(this.f5825a.get(this.f5829e - 1));
                x2.append(" must call proceed() exactly once");
                throw new IllegalStateException(x2.toString());
            }
            List<s> list = this.f5825a;
            int i = this.f5829e;
            f fVar = new f(list, gVar, cVar, cVar2, i + 1, yVar, this.f5831g, this.f5832h, this.i, this.j, this.k);
            s sVar = list.get(i);
            b0 a2 = sVar.a(fVar);
            if (cVar != null && this.f5829e + 1 < this.f5825a.size() && fVar.l != 1) {
                throw new IllegalStateException("network interceptor " + sVar + " must call proceed() exactly once");
            } else if (a2 != null) {
                if (a2.f5730h != null) {
                    return a2;
                }
                throw new IllegalStateException("interceptor " + sVar + " returned a response with no body");
            } else {
                throw new NullPointerException("interceptor " + sVar + " returned null");
            }
        }
        throw new AssertionError();
    }
}