package b.d.a.e;

import android.hardware.camera2.CameraDevice;
import android.os.Handler;
import android.view.Surface;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.ScheduledExecutorService;

/* compiled from: SynchronizedCaptureSessionOpener.java */
/* loaded from: classes.dex */
public final class t1 {

    /* renamed from: a  reason: collision with root package name */
    public final b f1196a;

    /* compiled from: SynchronizedCaptureSessionOpener.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Executor f1197a;

        /* renamed from: b  reason: collision with root package name */
        public final ScheduledExecutorService f1198b;

        /* renamed from: c  reason: collision with root package name */
        public final Handler f1199c;

        /* renamed from: d  reason: collision with root package name */
        public final h1 f1200d;

        /* renamed from: e  reason: collision with root package name */
        public final int f1201e;

        /* renamed from: f  reason: collision with root package name */
        public final Set<String> f1202f;

        public a(Executor executor, ScheduledExecutorService scheduledExecutorService, Handler handler, h1 h1Var, int i) {
            HashSet hashSet = new HashSet();
            this.f1202f = hashSet;
            this.f1197a = executor;
            this.f1198b = scheduledExecutorService;
            this.f1199c = handler;
            this.f1200d = h1Var;
            this.f1201e = i;
            if (i == 2) {
                hashSet.add("deferrableSurface_close");
            }
            if (i == 2) {
                hashSet.add("wait_for_request");
            }
        }

        public t1 a() {
            if (this.f1202f.isEmpty()) {
                return new t1(new r1(this.f1200d, this.f1197a, this.f1198b, this.f1199c));
            }
            return new t1(new s1(this.f1202f, this.f1200d, this.f1197a, this.f1198b, this.f1199c));
        }
    }

    /* compiled from: SynchronizedCaptureSessionOpener.java */
    /* loaded from: classes.dex */
    public interface b {
        ListenableFuture<List<Surface>> a(List<b.d.b.d1.j0> list, long j);

        ListenableFuture<Void> i(CameraDevice cameraDevice, b.d.a.e.y1.o.g gVar);

        boolean stop();
    }

    public t1(b bVar) {
        this.f1196a = bVar;
    }

    public boolean a() {
        return this.f1196a.stop();
    }
}