package b.d.a.e.y1;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.os.Handler;
import b.d.a.e.y1.b;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: CameraCaptureSessionCompatBaseImpl.java */
/* loaded from: classes.dex */
public class d implements b.a {

    /* renamed from: a  reason: collision with root package name */
    public final CameraCaptureSession f1293a;

    /* renamed from: b  reason: collision with root package name */
    public final Object f1294b;

    /* compiled from: CameraCaptureSessionCompatBaseImpl.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Handler f1295a;

        public a(Handler handler) {
            this.f1295a = handler;
        }
    }

    public d(CameraCaptureSession cameraCaptureSession, Object obj) {
        Objects.requireNonNull(cameraCaptureSession);
        this.f1293a = cameraCaptureSession;
        this.f1294b = obj;
    }

    @Override // b.d.a.e.y1.b.a
    public int a(CaptureRequest captureRequest, Executor executor, CameraCaptureSession.CaptureCallback captureCallback) {
        return this.f1293a.setRepeatingRequest(captureRequest, new b.C0013b(executor, captureCallback), ((a) this.f1294b).f1295a);
    }

    @Override // b.d.a.e.y1.b.a
    public int b(List<CaptureRequest> list, Executor executor, CameraCaptureSession.CaptureCallback captureCallback) {
        return this.f1293a.captureBurst(list, new b.C0013b(executor, captureCallback), ((a) this.f1294b).f1295a);
    }
}