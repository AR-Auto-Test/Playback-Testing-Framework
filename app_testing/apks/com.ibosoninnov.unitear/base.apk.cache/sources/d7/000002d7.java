package b.d.a.f;

import b.d.b.d1.a1;
import b.d.b.d1.i0;
import b.d.b.d1.u0;
import b.d.b.d1.w0;

/* compiled from: CaptureRequestOptions.java */
/* loaded from: classes.dex */
public class i implements a1 {
    public final i0 q;

    /* compiled from: CaptureRequestOptions.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final u0 f1376a = u0.y();

        public static a b(i0 i0Var) {
            a aVar = new a();
            i0Var.c("camera2.captureRequest.option.", new g(aVar, i0Var));
            return aVar;
        }

        public i a() {
            return new i(w0.x(this.f1376a));
        }
    }

    public i(i0 i0Var) {
        this.q = i0Var;
    }

    @Override // b.d.b.d1.a1
    public i0 k() {
        return this.q;
    }
}