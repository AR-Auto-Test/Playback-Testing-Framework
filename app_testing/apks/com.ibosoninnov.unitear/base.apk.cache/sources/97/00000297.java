package b.d.a.e.y1;

import android.hardware.camera2.CameraDevice;
import android.os.Build;
import android.os.Handler;
import b.d.a.e.y1.j;
import java.util.concurrent.Executor;

/* compiled from: CameraDeviceCompat.java */
/* loaded from: classes.dex */
public final class f {

    /* renamed from: a  reason: collision with root package name */
    public final a f1298a;

    /* compiled from: CameraDeviceCompat.java */
    /* loaded from: classes.dex */
    public interface a {
        void a(b.d.a.e.y1.o.g gVar);
    }

    /* compiled from: CameraDeviceCompat.java */
    /* loaded from: classes.dex */
    public static final class b extends CameraDevice.StateCallback {

        /* renamed from: a  reason: collision with root package name */
        public final CameraDevice.StateCallback f1299a;

        /* renamed from: b  reason: collision with root package name */
        public final Executor f1300b;

        /* compiled from: CameraDeviceCompat.java */
        /* loaded from: classes.dex */
        public class a implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraDevice f1301b;

            public a(CameraDevice cameraDevice) {
                this.f1301b = cameraDevice;
            }

            @Override // java.lang.Runnable
            public void run() {
                b.this.f1299a.onOpened(this.f1301b);
            }
        }

        /* compiled from: CameraDeviceCompat.java */
        /* renamed from: b.d.a.e.y1.f$b$b  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0017b implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraDevice f1303b;

            public RunnableC0017b(CameraDevice cameraDevice) {
                this.f1303b = cameraDevice;
            }

            @Override // java.lang.Runnable
            public void run() {
                b.this.f1299a.onDisconnected(this.f1303b);
            }
        }

        /* compiled from: CameraDeviceCompat.java */
        /* loaded from: classes.dex */
        public class c implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraDevice f1305b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ int f1306c;

            public c(CameraDevice cameraDevice, int i) {
                this.f1305b = cameraDevice;
                this.f1306c = i;
            }

            @Override // java.lang.Runnable
            public void run() {
                b.this.f1299a.onError(this.f1305b, this.f1306c);
            }
        }

        /* compiled from: CameraDeviceCompat.java */
        /* loaded from: classes.dex */
        public class d implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraDevice f1308b;

            public d(CameraDevice cameraDevice) {
                this.f1308b = cameraDevice;
            }

            @Override // java.lang.Runnable
            public void run() {
                b.this.f1299a.onClosed(this.f1308b);
            }
        }

        public b(Executor executor, CameraDevice.StateCallback stateCallback) {
            this.f1300b = executor;
            this.f1299a = stateCallback;
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onClosed(CameraDevice cameraDevice) {
            this.f1300b.execute(new d(cameraDevice));
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onDisconnected(CameraDevice cameraDevice) {
            this.f1300b.execute(new RunnableC0017b(cameraDevice));
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onError(CameraDevice cameraDevice, int i) {
            this.f1300b.execute(new c(cameraDevice, i));
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onOpened(CameraDevice cameraDevice) {
            this.f1300b.execute(new a(cameraDevice));
        }
    }

    public f(CameraDevice cameraDevice, Handler handler) {
        if (Build.VERSION.SDK_INT >= 28) {
            this.f1298a = new i(cameraDevice);
        } else {
            this.f1298a = new h(cameraDevice, new j.a(handler));
        }
    }
}