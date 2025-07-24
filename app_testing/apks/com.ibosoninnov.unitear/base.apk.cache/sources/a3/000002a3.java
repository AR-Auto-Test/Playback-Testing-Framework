package b.d.a.e.y1;

import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Handler;
import android.util.ArrayMap;
import b.d.a.e.y1.n;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executor;

/* compiled from: CameraManagerCompat.java */
/* loaded from: classes.dex */
public final class k {

    /* renamed from: a  reason: collision with root package name */
    public final b f1313a;

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, e> f1314b = new ArrayMap(4);

    /* compiled from: CameraManagerCompat.java */
    /* loaded from: classes.dex */
    public static final class a extends CameraManager.AvailabilityCallback {

        /* renamed from: a  reason: collision with root package name */
        public final Executor f1315a;

        /* renamed from: b  reason: collision with root package name */
        public final CameraManager.AvailabilityCallback f1316b;

        /* renamed from: c  reason: collision with root package name */
        public final Object f1317c = new Object();

        /* renamed from: d  reason: collision with root package name */
        public boolean f1318d = false;

        /* compiled from: CameraManagerCompat.java */
        /* renamed from: b.d.a.e.y1.k$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class RunnableC0018a implements Runnable {
            public RunnableC0018a() {
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f1316b.onCameraAccessPrioritiesChanged();
            }
        }

        /* compiled from: CameraManagerCompat.java */
        /* loaded from: classes.dex */
        public class b implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ String f1320b;

            public b(String str) {
                this.f1320b = str;
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f1316b.onCameraAvailable(this.f1320b);
            }
        }

        /* compiled from: CameraManagerCompat.java */
        /* loaded from: classes.dex */
        public class c implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ String f1322b;

            public c(String str) {
                this.f1322b = str;
            }

            @Override // java.lang.Runnable
            public void run() {
                a.this.f1316b.onCameraUnavailable(this.f1322b);
            }
        }

        public a(Executor executor, CameraManager.AvailabilityCallback availabilityCallback) {
            this.f1315a = executor;
            this.f1316b = availabilityCallback;
        }

        @Override // android.hardware.camera2.CameraManager.AvailabilityCallback
        public void onCameraAccessPrioritiesChanged() {
            synchronized (this.f1317c) {
                if (!this.f1318d) {
                    this.f1315a.execute(new RunnableC0018a());
                }
            }
        }

        @Override // android.hardware.camera2.CameraManager.AvailabilityCallback
        public void onCameraAvailable(String str) {
            synchronized (this.f1317c) {
                if (!this.f1318d) {
                    this.f1315a.execute(new b(str));
                }
            }
        }

        @Override // android.hardware.camera2.CameraManager.AvailabilityCallback
        public void onCameraUnavailable(String str) {
            synchronized (this.f1317c) {
                if (!this.f1318d) {
                    this.f1315a.execute(new c(str));
                }
            }
        }
    }

    /* compiled from: CameraManagerCompat.java */
    /* loaded from: classes.dex */
    public interface b {
        void a(Executor executor, CameraManager.AvailabilityCallback availabilityCallback);

        void b(CameraManager.AvailabilityCallback availabilityCallback);

        CameraCharacteristics c(String str);

        void d(String str, Executor executor, CameraDevice.StateCallback stateCallback);
    }

    public k(b bVar) {
        this.f1313a = bVar;
    }

    public static k a(Context context, Handler handler) {
        n nVar;
        int i = Build.VERSION.SDK_INT;
        if (i >= 29) {
            nVar = new m(context);
        } else if (i >= 28) {
            nVar = new l(context);
        } else {
            nVar = new n(context, new n.a(handler));
        }
        return new k(nVar);
    }

    public e b(String str) {
        e eVar;
        synchronized (this.f1314b) {
            eVar = this.f1314b.get(str);
            if (eVar == null) {
                e eVar2 = new e(this.f1313a.c(str));
                this.f1314b.put(str, eVar2);
                eVar = eVar2;
            }
        }
        return eVar;
    }

    public String[] c() {
        n nVar = (n) this.f1313a;
        Objects.requireNonNull(nVar);
        try {
            return nVar.f1324a.getCameraIdList();
        } catch (CameraAccessException e2) {
            Set<Integer> set = b.d.a.e.y1.a.f1242b;
            throw new b.d.a.e.y1.a(e2);
        }
    }
}