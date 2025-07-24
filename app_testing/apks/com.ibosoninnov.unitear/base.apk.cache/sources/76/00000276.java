package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import b.d.a.f.i;
import b.d.b.d1.b1;
import b.d.b.d1.f0;
import b.d.b.d1.i0;
import java.util.Objects;

/* compiled from: Camera2SessionOptionUnpacker.java */
/* loaded from: classes.dex */
public final class w0 implements b1.d {

    /* renamed from: a  reason: collision with root package name */
    public static final w0 f1223a = new w0();

    /* JADX DEBUG: Type inference failed for r0v4. Raw type applied. Possible types: b.d.b.d1.i0$a<java.lang.Integer>, b.d.b.d1.i0$a<ValueT> */
    /* JADX DEBUG: Type inference failed for r1v5. Raw type applied. Possible types: b.d.b.d1.i0$a<android.hardware.camera2.CameraDevice$StateCallback>, b.d.b.d1.i0$a<ValueT> */
    /* JADX DEBUG: Type inference failed for r1v6. Raw type applied. Possible types: b.d.b.d1.i0$a<android.hardware.camera2.CameraCaptureSession$StateCallback>, b.d.b.d1.i0$a<ValueT> */
    /* JADX DEBUG: Type inference failed for r1v7. Raw type applied. Possible types: b.d.b.d1.i0$a<android.hardware.camera2.CameraCaptureSession$CaptureCallback>, b.d.b.d1.i0$a<ValueT> */
    /* JADX DEBUG: Type inference failed for r1v9. Raw type applied. Possible types: b.d.b.d1.i0$a<b.d.a.d.c>, b.d.b.d1.i0$a<ValueT> */
    @Override // b.d.b.d1.b1.d
    public void a(b.d.b.d1.i1<?> i1Var, b1.b bVar) {
        b.d.b.d1.b1 m = i1Var.m(null);
        b.d.b.d1.i0 i0Var = b.d.b.d1.w0.q;
        int i = b.d.b.d1.b1.a().f1419f.f1464e;
        if (m != null) {
            i = m.f1419f.f1464e;
            for (CameraDevice.StateCallback stateCallback : m.f1415b) {
                bVar.b(stateCallback);
            }
            for (CameraCaptureSession.StateCallback stateCallback2 : m.f1416c) {
                bVar.c(stateCallback2);
            }
            bVar.f1421b.a(m.f1419f.f1465f);
            i0Var = m.f1419f.f1463d;
        }
        f0.a aVar = bVar.f1421b;
        Objects.requireNonNull(aVar);
        aVar.f1469b = b.d.b.d1.u0.z(i0Var);
        bVar.f1421b.f1470c = ((Integer) i1Var.f(b.d.a.d.a.r, Integer.valueOf(i))).intValue();
        bVar.b((CameraDevice.StateCallback) i1Var.f(b.d.a.d.a.s, new c1()));
        bVar.c((CameraCaptureSession.StateCallback) i1Var.f(b.d.a.d.a.t, new a1()));
        e1 e1Var = new e1((CameraCaptureSession.CaptureCallback) i1Var.f(b.d.a.d.a.u, new t0()));
        bVar.f1421b.b(e1Var);
        bVar.f1425f.add(e1Var);
        b.d.b.d1.u0 y = b.d.b.d1.u0.y();
        i0.a aVar2 = b.d.a.d.a.v;
        y.A(aVar2, i0.c.OPTIONAL, (b.d.a.d.c) i1Var.f(aVar2, b.d.a.d.c.d()));
        bVar.f1421b.c(y);
        bVar.f1421b.c(i.a.b(i1Var).a());
    }
}