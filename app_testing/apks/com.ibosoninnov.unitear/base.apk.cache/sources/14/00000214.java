package b.d.a.d;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import b.d.a.f.i;
import b.d.b.d1.i0;
import b.d.b.d1.n;
import b.d.b.d1.u0;
import b.d.b.d1.w0;

/* compiled from: Camera2ImplConfig.java */
/* loaded from: classes.dex */
public final class a extends i {
    public static final i0.a<Integer> r = new n("camera2.captureRequest.templateType", Integer.TYPE, null);
    public static final i0.a<CameraDevice.StateCallback> s = new n("camera2.cameraDevice.stateCallback", CameraDevice.StateCallback.class, null);
    public static final i0.a<CameraCaptureSession.StateCallback> t = new n("camera2.cameraCaptureSession.stateCallback", CameraCaptureSession.StateCallback.class, null);
    public static final i0.a<CameraCaptureSession.CaptureCallback> u = new n("camera2.cameraCaptureSession.captureCallback", CameraCaptureSession.CaptureCallback.class, null);
    public static final i0.a<c> v = new n("camera2.cameraEvent.callback", c.class, null);
    public static final i0.a<Object> w = new n("camera2.captureRequest.tag", Object.class, null);

    /* compiled from: Camera2ImplConfig.java */
    /* renamed from: b.d.a.d.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static final class C0012a {

        /* renamed from: a  reason: collision with root package name */
        public final u0 f1011a = u0.y();

        public a a() {
            return new a(w0.x(this.f1011a));
        }

        public <ValueT> C0012a b(CaptureRequest.Key<ValueT> key, ValueT valuet) {
            i0.a<Integer> aVar = a.r;
            StringBuilder x = c.b.a.a.a.x("camera2.captureRequest.option.");
            x.append(key.getName());
            this.f1011a.A(new n(x.toString(), Object.class, key), i0.c.OPTIONAL, valuet);
            return this;
        }
    }

    public a(i0 i0Var) {
        super(i0Var);
    }
}