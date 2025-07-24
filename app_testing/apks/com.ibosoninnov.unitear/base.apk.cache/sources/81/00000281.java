package b.d.a.e.y1;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.os.Build;
import android.os.Handler;
import android.view.Surface;
import b.d.a.e.y1.d;
import java.util.List;
import java.util.concurrent.Executor;

/* compiled from: CameraCaptureSessionCompat.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public final a f1244a;

    /* compiled from: CameraCaptureSessionCompat.java */
    /* loaded from: classes.dex */
    public interface a {
        int a(CaptureRequest captureRequest, Executor executor, CameraCaptureSession.CaptureCallback captureCallback);

        int b(List<CaptureRequest> list, Executor executor, CameraCaptureSession.CaptureCallback captureCallback);
    }

    /* compiled from: CameraCaptureSessionCompat.java */
    /* renamed from: b.d.a.e.y1.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static final class C0013b extends CameraCaptureSession.CaptureCallback {

        /* renamed from: a  reason: collision with root package name */
        public final CameraCaptureSession.CaptureCallback f1245a;

        /* renamed from: b  reason: collision with root package name */
        public final Executor f1246b;

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$a */
        /* loaded from: classes.dex */
        public class a implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1247b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ CaptureRequest f1248c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ long f1249d;

            /* renamed from: e  reason: collision with root package name */
            public final /* synthetic */ long f1250e;

            public a(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, long j, long j2) {
                this.f1247b = cameraCaptureSession;
                this.f1248c = captureRequest;
                this.f1249d = j;
                this.f1250e = j2;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureStarted(this.f1247b, this.f1248c, this.f1249d, this.f1250e);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$b  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0014b implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1252b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ CaptureRequest f1253c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ CaptureResult f1254d;

            public RunnableC0014b(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, CaptureResult captureResult) {
                this.f1252b = cameraCaptureSession;
                this.f1253c = captureRequest;
                this.f1254d = captureResult;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureProgressed(this.f1252b, this.f1253c, this.f1254d);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$c */
        /* loaded from: classes.dex */
        public class c implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1256b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ CaptureRequest f1257c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ TotalCaptureResult f1258d;

            public c(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, TotalCaptureResult totalCaptureResult) {
                this.f1256b = cameraCaptureSession;
                this.f1257c = captureRequest;
                this.f1258d = totalCaptureResult;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureCompleted(this.f1256b, this.f1257c, this.f1258d);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$d */
        /* loaded from: classes.dex */
        public class d implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1260b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ CaptureRequest f1261c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ CaptureFailure f1262d;

            public d(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, CaptureFailure captureFailure) {
                this.f1260b = cameraCaptureSession;
                this.f1261c = captureRequest;
                this.f1262d = captureFailure;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureFailed(this.f1260b, this.f1261c, this.f1262d);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$e */
        /* loaded from: classes.dex */
        public class e implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1264b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ int f1265c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ long f1266d;

            public e(CameraCaptureSession cameraCaptureSession, int i, long j) {
                this.f1264b = cameraCaptureSession;
                this.f1265c = i;
                this.f1266d = j;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureSequenceCompleted(this.f1264b, this.f1265c, this.f1266d);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$f */
        /* loaded from: classes.dex */
        public class f implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1268b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ int f1269c;

            public f(CameraCaptureSession cameraCaptureSession, int i) {
                this.f1268b = cameraCaptureSession;
                this.f1269c = i;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureSequenceAborted(this.f1268b, this.f1269c);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$b$g */
        /* loaded from: classes.dex */
        public class g implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1271b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ CaptureRequest f1272c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ Surface f1273d;

            /* renamed from: e  reason: collision with root package name */
            public final /* synthetic */ long f1274e;

            public g(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, Surface surface, long j) {
                this.f1271b = cameraCaptureSession;
                this.f1272c = captureRequest;
                this.f1273d = surface;
                this.f1274e = j;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0013b.this.f1245a.onCaptureBufferLost(this.f1271b, this.f1272c, this.f1273d, this.f1274e);
            }
        }

        public C0013b(Executor executor, CameraCaptureSession.CaptureCallback captureCallback) {
            this.f1246b = executor;
            this.f1245a = captureCallback;
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureBufferLost(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, Surface surface, long j) {
            this.f1246b.execute(new g(cameraCaptureSession, captureRequest, surface, j));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureCompleted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, TotalCaptureResult totalCaptureResult) {
            this.f1246b.execute(new c(cameraCaptureSession, captureRequest, totalCaptureResult));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureFailed(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, CaptureFailure captureFailure) {
            this.f1246b.execute(new d(cameraCaptureSession, captureRequest, captureFailure));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureProgressed(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, CaptureResult captureResult) {
            this.f1246b.execute(new RunnableC0014b(cameraCaptureSession, captureRequest, captureResult));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureSequenceAborted(CameraCaptureSession cameraCaptureSession, int i) {
            this.f1246b.execute(new f(cameraCaptureSession, i));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureSequenceCompleted(CameraCaptureSession cameraCaptureSession, int i, long j) {
            this.f1246b.execute(new e(cameraCaptureSession, i, j));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureStarted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, long j, long j2) {
            this.f1246b.execute(new a(cameraCaptureSession, captureRequest, j, j2));
        }
    }

    /* compiled from: CameraCaptureSessionCompat.java */
    /* loaded from: classes.dex */
    public static final class c extends CameraCaptureSession.StateCallback {

        /* renamed from: a  reason: collision with root package name */
        public final CameraCaptureSession.StateCallback f1276a;

        /* renamed from: b  reason: collision with root package name */
        public final Executor f1277b;

        /* compiled from: CameraCaptureSessionCompat.java */
        /* loaded from: classes.dex */
        public class a implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1278b;

            public a(CameraCaptureSession cameraCaptureSession) {
                this.f1278b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onConfigured(this.f1278b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$c$b  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0015b implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1280b;

            public RunnableC0015b(CameraCaptureSession cameraCaptureSession) {
                this.f1280b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onConfigureFailed(this.f1280b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* renamed from: b.d.a.e.y1.b$c$c  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0016c implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1282b;

            public RunnableC0016c(CameraCaptureSession cameraCaptureSession) {
                this.f1282b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onReady(this.f1282b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* loaded from: classes.dex */
        public class d implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1284b;

            public d(CameraCaptureSession cameraCaptureSession) {
                this.f1284b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onActive(this.f1284b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* loaded from: classes.dex */
        public class e implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1286b;

            public e(CameraCaptureSession cameraCaptureSession) {
                this.f1286b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onCaptureQueueEmpty(this.f1286b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* loaded from: classes.dex */
        public class f implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1288b;

            public f(CameraCaptureSession cameraCaptureSession) {
                this.f1288b = cameraCaptureSession;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onClosed(this.f1288b);
            }
        }

        /* compiled from: CameraCaptureSessionCompat.java */
        /* loaded from: classes.dex */
        public class g implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ CameraCaptureSession f1290b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ Surface f1291c;

            public g(CameraCaptureSession cameraCaptureSession, Surface surface) {
                this.f1290b = cameraCaptureSession;
                this.f1291c = surface;
            }

            @Override // java.lang.Runnable
            public void run() {
                c.this.f1276a.onSurfacePrepared(this.f1290b, this.f1291c);
            }
        }

        public c(Executor executor, CameraCaptureSession.StateCallback stateCallback) {
            this.f1277b = executor;
            this.f1276a = stateCallback;
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onActive(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new d(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onCaptureQueueEmpty(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new e(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onClosed(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new f(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new RunnableC0015b(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new a(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onReady(CameraCaptureSession cameraCaptureSession) {
            this.f1277b.execute(new RunnableC0016c(cameraCaptureSession));
        }

        @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
        public void onSurfacePrepared(CameraCaptureSession cameraCaptureSession, Surface surface) {
            this.f1277b.execute(new g(cameraCaptureSession, surface));
        }
    }

    public b(CameraCaptureSession cameraCaptureSession, Handler handler) {
        if (Build.VERSION.SDK_INT >= 28) {
            this.f1244a = new b.d.a.e.y1.c(cameraCaptureSession);
        } else {
            this.f1244a = new d(cameraCaptureSession, new d.a(handler));
        }
    }

    public CameraCaptureSession a() {
        return ((d) this.f1244a).f1293a;
    }
}