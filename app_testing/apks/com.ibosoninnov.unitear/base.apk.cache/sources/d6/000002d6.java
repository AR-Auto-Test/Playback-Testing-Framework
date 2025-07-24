package b.d.a.f;

import android.hardware.camera2.TotalCaptureResult;
import b.d.a.d.a;
import b.d.a.e.a0;
import b.d.a.e.o0;
import b.d.b.d1.g1;
import b.d.b.d1.i0;
import b.d.b.f0;
import java.util.concurrent.Executor;

/* compiled from: Camera2CameraControl.java */
/* loaded from: classes.dex */
public final class h {

    /* renamed from: c  reason: collision with root package name */
    public final o0 f1370c;

    /* renamed from: d  reason: collision with root package name */
    public final Executor f1371d;

    /* renamed from: g  reason: collision with root package name */
    public b.g.a.b<Void> f1374g;

    /* renamed from: a  reason: collision with root package name */
    public boolean f1368a = false;

    /* renamed from: b  reason: collision with root package name */
    public boolean f1369b = false;

    /* renamed from: e  reason: collision with root package name */
    public final Object f1372e = new Object();

    /* renamed from: f  reason: collision with root package name */
    public a.C0012a f1373f = new a.C0012a();

    /* renamed from: h  reason: collision with root package name */
    public final o0.c f1375h = new o0.c() { // from class: b.d.a.f.c
        /* JADX WARN: Removed duplicated region for block: B:13:0x0039  */
        /* JADX WARN: Removed duplicated region for block: B:16:? A[RETURN, SYNTHETIC] */
        @Override // b.d.a.e.o0.c
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public final boolean a(TotalCaptureResult totalCaptureResult) {
            b.g.a.b<Void> bVar;
            Integer num;
            h hVar = h.this;
            if (hVar.f1374g != null) {
                Object tag = totalCaptureResult.getRequest().getTag();
                if ((tag instanceof g1) && (num = ((g1) tag).f1480b.get("Camera2CameraControl")) != null && num.equals(Integer.valueOf(hVar.f1374g.hashCode()))) {
                    bVar = hVar.f1374g;
                    hVar.f1374g = null;
                    if (bVar == null) {
                        bVar.a(null);
                        return false;
                    }
                    return false;
                }
            }
            bVar = null;
            if (bVar == null) {
            }
        }
    };

    public h(o0 o0Var, Executor executor) {
        this.f1370c = o0Var;
        this.f1371d = executor;
    }

    public b.d.a.d.a a() {
        b.d.a.d.a a2;
        synchronized (this.f1372e) {
            b.g.a.b<Void> bVar = this.f1374g;
            if (bVar != null) {
                this.f1373f.f1011a.A(b.d.a.d.a.w, i0.c.OPTIONAL, Integer.valueOf(bVar.hashCode()));
            }
            a2 = this.f1373f.a();
        }
        return a2;
    }

    public final void b(b.g.a.b<Void> bVar) {
        this.f1369b = true;
        b.g.a.b<Void> bVar2 = this.f1374g;
        if (bVar2 == null) {
            bVar2 = null;
        }
        this.f1374g = bVar;
        if (this.f1368a) {
            o0 o0Var = this.f1370c;
            o0Var.f1112c.execute(new a0(o0Var));
            this.f1369b = false;
        }
        if (bVar2 != null) {
            bVar2.c(new f0.a("Camera2CameraControl was updated with new options."));
        }
    }
}