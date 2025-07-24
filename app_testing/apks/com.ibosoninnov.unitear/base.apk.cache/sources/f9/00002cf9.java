package f.g0.f;

import f.b0;
import f.s;
import f.v;
import f.y;
import java.io.IOException;
import java.util.Objects;

/* compiled from: ConnectInterceptor.java */
/* loaded from: classes2.dex */
public final class a implements s {

    /* renamed from: a  reason: collision with root package name */
    public final v f5785a;

    public a(v vVar) {
        this.f5785a = vVar;
    }

    @Override // f.s
    public b0 a(s.a aVar) {
        f.g0.g.f fVar = (f.g0.g.f) aVar;
        y yVar = fVar.f5830f;
        g gVar = fVar.f5826b;
        boolean z = !yVar.f6151b.equals("GET");
        v vVar = this.f5785a;
        Objects.requireNonNull(gVar);
        int i = fVar.i;
        int i2 = fVar.j;
        int i3 = fVar.k;
        Objects.requireNonNull(vVar);
        try {
            f.g0.g.c i4 = gVar.e(i, i2, i3, 0, vVar.w, z).i(vVar, aVar, gVar);
            synchronized (gVar.f5813d) {
                gVar.n = i4;
            }
            return fVar.b(yVar, gVar, i4, gVar.b());
        } catch (IOException e2) {
            throw new e(e2);
        }
    }
}