package androidx.camera.camera2;

import b.d.a.a;
import b.d.a.b;
import b.d.a.c;
import b.d.b.d1.i0;
import b.d.b.d1.u0;
import b.d.b.d1.w0;
import b.d.b.d1.y;
import b.d.b.o0;

/* loaded from: classes.dex */
public final class Camera2Config$DefaultProvider implements o0.b {
    @Override // b.d.b.o0.b
    public o0 getCameraXConfig() {
        c cVar = c.f1010a;
        b bVar = b.f1009a;
        a aVar = a.f1008a;
        o0.a aVar2 = new o0.a();
        u0 u0Var = aVar2.f1657a;
        i0.a<y.a> aVar3 = o0.q;
        i0.c cVar2 = i0.c.OPTIONAL;
        u0Var.A(aVar3, cVar2, cVar);
        aVar2.f1657a.A(o0.r, cVar2, bVar);
        aVar2.f1657a.A(o0.s, cVar2, aVar);
        return new o0(w0.x(aVar2.f1657a));
    }
}