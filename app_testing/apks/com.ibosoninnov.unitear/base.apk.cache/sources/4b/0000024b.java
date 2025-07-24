package b.d.a.e;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.MeteringRectangle;
import android.os.Looper;
import android.util.ArrayMap;
import android.util.Range;
import b.d.a.d.a;
import b.d.a.e.a0;
import b.d.a.e.o0;
import b.d.a.f.i;
import b.d.b.d1.b1;
import b.d.b.d1.f0;
import b.d.b.d1.i0;
import b.d.b.d1.k1.c.h;
import b.d.b.d1.w;
import b.d.b.f0;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;

/* compiled from: Camera2CameraControlImpl.java */
/* loaded from: classes.dex */
public class o0 implements b.d.b.d1.w {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f1110a = 0;

    /* renamed from: b  reason: collision with root package name */
    public final b f1111b;

    /* renamed from: c  reason: collision with root package name */
    public final Executor f1112c;

    /* renamed from: d  reason: collision with root package name */
    public final Object f1113d = new Object();

    /* renamed from: e  reason: collision with root package name */
    public final b.d.a.e.y1.e f1114e;

    /* renamed from: f  reason: collision with root package name */
    public final w.a f1115f;

    /* renamed from: g  reason: collision with root package name */
    public final b1.b f1116g;

    /* renamed from: h  reason: collision with root package name */
    public final l1 f1117h;
    public final w1 i;
    public final v1 j;
    public final j1 k;
    public final b.d.a.f.h l;
    public final b.d.a.e.y1.q.a m;
    public int n;
    public volatile boolean o;
    public volatile int p;
    public final b.d.a.e.y1.q.b q;
    public final a r;

    /* compiled from: Camera2CameraControlImpl.java */
    /* loaded from: classes.dex */
    public static final class a extends b.d.b.d1.q {

        /* renamed from: a  reason: collision with root package name */
        public Set<b.d.b.d1.q> f1118a = new HashSet();

        /* renamed from: b  reason: collision with root package name */
        public Map<b.d.b.d1.q, Executor> f1119b = new ArrayMap();

        @Override // b.d.b.d1.q
        public void a() {
            for (final b.d.b.d1.q qVar : this.f1118a) {
                try {
                    this.f1119b.get(qVar).execute(new Runnable() { // from class: b.d.a.e.c
                        @Override // java.lang.Runnable
                        public final void run() {
                            b.d.b.d1.q.this.a();
                        }
                    });
                } catch (RejectedExecutionException e2) {
                    b.d.b.u0.b("Camera2CameraControlImp", "Executor rejected to invoke onCaptureCancelled.", e2);
                }
            }
        }

        @Override // b.d.b.d1.q
        public void b(final b.d.b.d1.t tVar) {
            for (final b.d.b.d1.q qVar : this.f1118a) {
                try {
                    this.f1119b.get(qVar).execute(new Runnable() { // from class: b.d.a.e.b
                        @Override // java.lang.Runnable
                        public final void run() {
                            b.d.b.d1.q.this.b(tVar);
                        }
                    });
                } catch (RejectedExecutionException e2) {
                    b.d.b.u0.b("Camera2CameraControlImp", "Executor rejected to invoke onCaptureCompleted.", e2);
                }
            }
        }

        @Override // b.d.b.d1.q
        public void c(final b.d.b.d1.s sVar) {
            for (final b.d.b.d1.q qVar : this.f1118a) {
                try {
                    this.f1119b.get(qVar).execute(new Runnable() { // from class: b.d.a.e.d
                        @Override // java.lang.Runnable
                        public final void run() {
                            b.d.b.d1.q.this.c(sVar);
                        }
                    });
                } catch (RejectedExecutionException e2) {
                    b.d.b.u0.b("Camera2CameraControlImp", "Executor rejected to invoke onCaptureFailed.", e2);
                }
            }
        }
    }

    /* compiled from: Camera2CameraControlImpl.java */
    /* loaded from: classes.dex */
    public static final class b extends CameraCaptureSession.CaptureCallback {

        /* renamed from: a  reason: collision with root package name */
        public final Set<c> f1120a = new HashSet();

        /* renamed from: b  reason: collision with root package name */
        public final Executor f1121b;

        public b(Executor executor) {
            this.f1121b = executor;
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureCompleted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, final TotalCaptureResult totalCaptureResult) {
            this.f1121b.execute(new Runnable() { // from class: b.d.a.e.e
                @Override // java.lang.Runnable
                public final void run() {
                    o0.b bVar = o0.b.this;
                    TotalCaptureResult totalCaptureResult2 = totalCaptureResult;
                    Objects.requireNonNull(bVar);
                    HashSet hashSet = new HashSet();
                    for (o0.c cVar : bVar.f1120a) {
                        if (cVar.a(totalCaptureResult2)) {
                            hashSet.add(cVar);
                        }
                    }
                    if (hashSet.isEmpty()) {
                        return;
                    }
                    bVar.f1120a.removeAll(hashSet);
                }
            });
        }
    }

    /* compiled from: Camera2CameraControlImpl.java */
    /* loaded from: classes.dex */
    public interface c {
        boolean a(TotalCaptureResult totalCaptureResult);
    }

    public o0(b.d.a.e.y1.e eVar, ScheduledExecutorService scheduledExecutorService, Executor executor, w.a aVar, b.d.b.d1.z0 z0Var) {
        b1.b bVar = new b1.b();
        this.f1116g = bVar;
        this.n = 0;
        this.o = false;
        this.p = 2;
        this.q = new b.d.a.e.y1.q.b();
        a aVar2 = new a();
        this.r = aVar2;
        this.f1114e = eVar;
        this.f1115f = aVar;
        this.f1112c = executor;
        b bVar2 = new b(executor);
        this.f1111b = bVar2;
        bVar.f1421b.f1470c = 1;
        bVar.f1421b.b(new e1(bVar2));
        bVar.f1421b.b(aVar2);
        this.k = new j1(this, eVar, executor);
        this.f1117h = new l1(this, scheduledExecutorService, executor);
        this.i = new w1(this, eVar, executor);
        this.j = new v1(this, eVar, executor);
        this.m = new b.d.a.e.y1.q.a(z0Var);
        this.l = new b.d.a.f.h(this, executor);
        b.d.b.d1.k1.b.d dVar = (b.d.b.d1.k1.b.d) executor;
        dVar.execute(new Runnable() { // from class: b.d.a.e.f
            @Override // java.lang.Runnable
            public final void run() {
                o0 o0Var = o0.this;
                o0Var.e(o0Var.l.f1375h);
            }
        });
        dVar.execute(new a0(this));
    }

    @Override // b.d.b.d1.w
    public void a(b.d.b.d1.i0 i0Var) {
        final b.d.a.f.h hVar = this.l;
        b.d.a.f.i a2 = i.a.b(i0Var).a();
        synchronized (hVar.f1372e) {
            for (i0.a<?> aVar : a2.e()) {
                hVar.f1373f.f1011a.A(aVar, i0.c.OPTIONAL, a2.a(aVar));
            }
        }
        b.d.b.d1.k1.c.g.d(b.e.a.d(new b.g.a.d() { // from class: b.d.a.f.f
            @Override // b.g.a.d
            public final Object a(final b.g.a.b bVar) {
                final h hVar2 = h.this;
                hVar2.f1371d.execute(new Runnable() { // from class: b.d.a.f.b
                    @Override // java.lang.Runnable
                    public final void run() {
                        h.this.b(bVar);
                    }
                });
                return "addCaptureRequestOptions";
            }
        })).addListener(h.f1056b, b.b.a.f());
    }

    @Override // b.d.b.f0
    public ListenableFuture<Void> b(final boolean z) {
        int i;
        ListenableFuture d2;
        synchronized (this.f1113d) {
            i = this.n;
        }
        if (!(i > 0)) {
            return new h.a(new f0.a("Camera is not active."));
        }
        final v1 v1Var = this.j;
        if (!v1Var.f1213c) {
            b.d.b.u0.a("TorchControl", "Unable to enableTorch due to there is no flash unit.", null);
            d2 = new h.a(new IllegalStateException("No flash unit"));
        } else {
            v1Var.a(v1Var.f1212b, Integer.valueOf(z ? 1 : 0));
            d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.a.e.k0
                @Override // b.g.a.d
                public final Object a(final b.g.a.b bVar) {
                    final v1 v1Var2 = v1.this;
                    final boolean z2 = z;
                    v1Var2.f1214d.execute(new Runnable() { // from class: b.d.a.e.j0
                        @Override // java.lang.Runnable
                        public final void run() {
                            v1 v1Var3 = v1.this;
                            b.g.a.b<Void> bVar2 = bVar;
                            boolean z3 = z2;
                            if (!v1Var3.f1215e) {
                                v1Var3.a(v1Var3.f1212b, 0);
                                bVar2.c(new f0.a("Camera is not active."));
                                return;
                            }
                            v1Var3.f1217g = z3;
                            v1Var3.f1211a.g(z3);
                            v1Var3.a(v1Var3.f1212b, Integer.valueOf(z3 ? 1 : 0));
                            b.g.a.b<Void> bVar3 = v1Var3.f1216f;
                            if (bVar3 != null) {
                                bVar3.c(new f0.a("There is a new enableTorch being set"));
                            }
                            v1Var3.f1216f = bVar2;
                        }
                    });
                    return "enableTorch: " + z2;
                }
            });
        }
        return b.d.b.d1.k1.c.g.d(d2);
    }

    @Override // b.d.b.d1.w
    public b.d.b.d1.i0 c() {
        return this.l.a();
    }

    @Override // b.d.b.d1.w
    public void d() {
        final b.d.a.f.h hVar = this.l;
        synchronized (hVar.f1372e) {
            hVar.f1373f = new a.C0012a();
        }
        b.d.b.d1.k1.c.g.d(b.e.a.d(new b.g.a.d() { // from class: b.d.a.f.d
            @Override // b.g.a.d
            public final Object a(final b.g.a.b bVar) {
                final h hVar2 = h.this;
                hVar2.f1371d.execute(new Runnable() { // from class: b.d.a.f.a
                    @Override // java.lang.Runnable
                    public final void run() {
                        h.this.b(bVar);
                    }
                });
                return "clearCaptureRequestOptions";
            }
        })).addListener(g.f1035b, b.b.a.f());
    }

    public void e(c cVar) {
        this.f1111b.f1120a.add(cVar);
    }

    public void f() {
        synchronized (this.f1113d) {
            int i = this.n;
            if (i != 0) {
                this.n = i - 1;
            } else {
                throw new IllegalStateException("Decrementing use count occurs more times than incrementing");
            }
        }
    }

    public void g(boolean z) {
        i0.c cVar = i0.c.OPTIONAL;
        this.o = z;
        if (!z) {
            f0.a aVar = new f0.a();
            aVar.f1470c = 1;
            aVar.f1472e = true;
            b.d.b.d1.u0 y = b.d.b.d1.u0.y();
            CaptureRequest.Key key = CaptureRequest.CONTROL_AE_MODE;
            Integer valueOf = Integer.valueOf(h(1));
            i0.a<Integer> aVar2 = b.d.a.d.a.r;
            StringBuilder x = c.b.a.a.a.x("camera2.captureRequest.option.");
            x.append(key.getName());
            y.A(new b.d.b.d1.n(x.toString(), Object.class, key), cVar, valueOf);
            CaptureRequest.Key key2 = CaptureRequest.FLASH_MODE;
            StringBuilder x2 = c.b.a.a.a.x("camera2.captureRequest.option.");
            x2.append(key2.getName());
            y.A(new b.d.b.d1.n(x2.toString(), Object.class, key2), cVar, 0);
            aVar.c(new b.d.a.d.a(b.d.b.d1.w0.x(y)));
            m(Collections.singletonList(aVar.d()));
        }
        n();
    }

    public final int h(int i) {
        int[] iArr = (int[]) this.f1114e.a(CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES);
        if (iArr == null) {
            return 0;
        }
        return j(i, iArr) ? i : j(1, iArr) ? 1 : 0;
    }

    public int i(int i) {
        int[] iArr = (int[]) this.f1114e.a(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES);
        if (iArr == null) {
            return 0;
        }
        if (j(i, iArr)) {
            return i;
        }
        if (j(4, iArr)) {
            return 4;
        }
        return j(1, iArr) ? 1 : 0;
    }

    public final boolean j(int i, int[] iArr) {
        for (int i2 : iArr) {
            if (i == i2) {
                return true;
            }
        }
        return false;
    }

    public void k(c cVar) {
        this.f1111b.f1120a.remove(cVar);
    }

    public void l(final boolean z) {
        b.d.b.c1 a2;
        l1 l1Var = this.f1117h;
        if (z != l1Var.f1095d) {
            l1Var.f1095d = z;
            if (!l1Var.f1095d) {
                l1Var.f1092a.k(l1Var.f1096e);
                b.g.a.b<Void> bVar = l1Var.l;
                if (bVar != null) {
                    bVar.c(new f0.a("Cancelled by another cancelFocusAndMetering()"));
                    l1Var.l = null;
                }
                l1Var.f1092a.k(null);
                l1Var.l = null;
                if (l1Var.f1097f.length > 0) {
                    i0.c cVar = i0.c.OPTIONAL;
                    if (l1Var.f1095d) {
                        f0.a aVar = new f0.a();
                        aVar.f1472e = true;
                        aVar.f1470c = 1;
                        b.d.b.d1.u0 y = b.d.b.d1.u0.y();
                        CaptureRequest.Key key = CaptureRequest.CONTROL_AF_TRIGGER;
                        i0.a<Integer> aVar2 = b.d.a.d.a.r;
                        StringBuilder x = c.b.a.a.a.x("camera2.captureRequest.option.");
                        x.append(key.getName());
                        y.A(new b.d.b.d1.n(x.toString(), Object.class, key), cVar, 2);
                        aVar.c(new b.d.a.d.a(b.d.b.d1.w0.x(y)));
                        l1Var.f1092a.m(Collections.singletonList(aVar.d()));
                    }
                }
                l1Var.f1097f = new MeteringRectangle[0];
                l1Var.f1098g = new MeteringRectangle[0];
                l1Var.f1099h = new MeteringRectangle[0];
                l1Var.f1092a.n();
            }
        }
        w1 w1Var = this.i;
        if (w1Var.f1229f != z) {
            w1Var.f1229f = z;
            if (!z) {
                synchronized (w1Var.f1226c) {
                    w1Var.f1226c.a(1.0f);
                    a2 = b.d.b.e1.d.a(w1Var.f1226c);
                }
                if (Looper.myLooper() == Looper.getMainLooper()) {
                    w1Var.f1227d.h(a2);
                } else {
                    w1Var.f1227d.i(a2);
                }
                w1Var.f1228e.e();
                w1Var.f1224a.n();
            }
        }
        v1 v1Var = this.j;
        if (v1Var.f1215e != z) {
            v1Var.f1215e = z;
            if (!z) {
                if (v1Var.f1217g) {
                    v1Var.f1217g = false;
                    v1Var.f1211a.g(false);
                    v1Var.a(v1Var.f1212b, 0);
                }
                b.g.a.b<Void> bVar2 = v1Var.f1216f;
                if (bVar2 != null) {
                    bVar2.c(new f0.a("Camera is not active."));
                    v1Var.f1216f = null;
                }
            }
        }
        j1 j1Var = this.k;
        if (z != j1Var.f1080d) {
            j1Var.f1080d = z;
            if (!z) {
                k1 k1Var = j1Var.f1078b;
                synchronized (k1Var.f1085a) {
                    k1Var.f1086b = 0;
                }
            }
        }
        final b.d.a.f.h hVar = this.l;
        hVar.f1371d.execute(new Runnable() { // from class: b.d.a.f.e
            @Override // java.lang.Runnable
            public final void run() {
                h hVar2 = h.this;
                boolean z2 = z;
                if (hVar2.f1368a == z2) {
                    return;
                }
                hVar2.f1368a = z2;
                if (z2) {
                    if (hVar2.f1369b) {
                        o0 o0Var = hVar2.f1370c;
                        o0Var.f1112c.execute(new a0(o0Var));
                        hVar2.f1369b = false;
                        return;
                    }
                    return;
                }
                synchronized (hVar2.f1372e) {
                    hVar2.f1373f = new a.C0012a();
                }
                b.g.a.b<Void> bVar3 = hVar2.f1374g;
                if (bVar3 != null) {
                    bVar3.c(new f0.a("The camera control has became inactive."));
                    hVar2.f1374g = null;
                }
            }
        });
    }

    public void m(List<b.d.b.d1.f0> list) {
        q0 q0Var = q0.this;
        Objects.requireNonNull(list);
        Objects.requireNonNull(q0Var);
        ArrayList arrayList = new ArrayList();
        for (b.d.b.d1.f0 f0Var : list) {
            HashSet hashSet = new HashSet();
            b.d.b.d1.u0.y();
            ArrayList arrayList2 = new ArrayList();
            new ArrayMap();
            hashSet.addAll(f0Var.f1462c);
            b.d.b.d1.u0 z = b.d.b.d1.u0.z(f0Var.f1463d);
            int i = f0Var.f1464e;
            arrayList2.addAll(f0Var.f1465f);
            boolean z2 = f0Var.f1466g;
            b.d.b.d1.g1 g1Var = f0Var.f1467h;
            ArrayMap arrayMap = new ArrayMap();
            for (String str : g1Var.f1480b.keySet()) {
                arrayMap.put(str, g1Var.a(str));
            }
            b.d.b.d1.v0 v0Var = new b.d.b.d1.v0(arrayMap);
            if (f0Var.a().isEmpty() && f0Var.f1466g) {
                boolean z3 = false;
                if (!hashSet.isEmpty()) {
                    b.d.b.u0.d("Camera2CameraImpl", "The capture config builder already has surface inside.", null);
                } else {
                    for (b.d.b.d1.b1 b1Var : Collections.unmodifiableCollection(q0Var.f1141a.c(b.d.b.d1.k.f1511a))) {
                        List<b.d.b.d1.j0> a2 = b1Var.f1419f.a();
                        if (!a2.isEmpty()) {
                            for (b.d.b.d1.j0 j0Var : a2) {
                                hashSet.add(j0Var);
                            }
                        }
                    }
                    if (hashSet.isEmpty()) {
                        b.d.b.u0.d("Camera2CameraImpl", "Unable to find a repeating surface to attach to CaptureConfig", null);
                    } else {
                        z3 = true;
                    }
                }
                if (!z3) {
                }
            }
            ArrayList arrayList3 = new ArrayList(hashSet);
            b.d.b.d1.w0 x = b.d.b.d1.w0.x(z);
            b.d.b.d1.g1 g1Var2 = b.d.b.d1.g1.f1479a;
            ArrayMap arrayMap2 = new ArrayMap();
            for (String str2 : v0Var.f1480b.keySet()) {
                arrayMap2.put(str2, v0Var.a(str2));
            }
            arrayList.add(new b.d.b.d1.f0(arrayList3, x, i, arrayList2, z2, new b.d.b.d1.g1(arrayMap2)));
        }
        q0Var.n("Issue capture request", null);
        q0Var.k.d(arrayList);
    }

    /* JADX WARN: Removed duplicated region for block: B:54:0x00bf A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void n() {
        int i;
        int[] iArr;
        k1 k1Var;
        int i2;
        b1.b bVar = this.f1116g;
        a.C0012a c0012a = new a.C0012a();
        int i3 = 1;
        c0012a.b(CaptureRequest.CONTROL_MODE, 1);
        l1 l1Var = this.f1117h;
        Objects.requireNonNull(l1Var);
        c0012a.b(CaptureRequest.CONTROL_AF_MODE, Integer.valueOf(l1Var.f1092a.i(4)));
        MeteringRectangle[] meteringRectangleArr = l1Var.f1097f;
        if (meteringRectangleArr.length != 0) {
            c0012a.b(CaptureRequest.CONTROL_AF_REGIONS, meteringRectangleArr);
        }
        MeteringRectangle[] meteringRectangleArr2 = l1Var.f1098g;
        if (meteringRectangleArr2.length != 0) {
            c0012a.b(CaptureRequest.CONTROL_AE_REGIONS, meteringRectangleArr2);
        }
        MeteringRectangle[] meteringRectangleArr3 = l1Var.f1099h;
        if (meteringRectangleArr3.length != 0) {
            c0012a.b(CaptureRequest.CONTROL_AWB_REGIONS, meteringRectangleArr3);
        }
        Range<Integer> range = this.m.f1352a;
        if (range != null) {
            c0012a.b(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, range);
        }
        this.i.f1228e.b(c0012a);
        if (this.o) {
            c0012a.b(CaptureRequest.FLASH_MODE, 2);
        } else {
            int i4 = this.p;
            if (i4 == 0) {
                Objects.requireNonNull(this.q);
                i = ((b.d.a.e.y1.p.c) b.d.a.e.y1.p.d.a(b.d.a.e.y1.p.c.class)) != null ? 1 : 2;
            } else if (i4 == 1) {
                i = 3;
            }
            c0012a.b(CaptureRequest.CONTROL_AE_MODE, Integer.valueOf(h(i)));
            CaptureRequest.Key key = CaptureRequest.CONTROL_AWB_MODE;
            iArr = (int[]) this.f1114e.a(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES);
            if (iArr != null || (!j(1, iArr) && !j(1, iArr))) {
                i3 = 0;
            }
            c0012a.b(key, Integer.valueOf(i3));
            j1 j1Var = this.k;
            Objects.requireNonNull(j1Var);
            CaptureRequest.Key key2 = CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION;
            k1Var = j1Var.f1078b;
            synchronized (k1Var.f1085a) {
                i2 = k1Var.f1086b;
            }
            c0012a.b(key2, Integer.valueOf(i2));
            b.d.a.d.a a2 = this.l.a();
            for (i0.a<?> aVar : a2.e()) {
                c0012a.f1011a.A(aVar, i0.c.ALWAYS_OVERRIDE, a2.a(aVar));
            }
            b.d.a.d.a a3 = c0012a.a();
            f0.a aVar2 = bVar.f1421b;
            Objects.requireNonNull(aVar2);
            aVar2.f1469b = b.d.b.d1.u0.z(a3);
            Object f2 = this.l.a().q.f(b.d.a.d.a.w, null);
            if (f2 != null && (f2 instanceof Integer)) {
                this.f1116g.f1421b.f1473f.f1480b.put("Camera2CameraControl", (Integer) f2);
            }
            w.a aVar3 = this.f1115f;
            b.d.b.d1.b1 d2 = this.f1116g.d();
            q0 q0Var = q0.this;
            q0Var.l = d2;
            q0Var.y();
            return;
        }
        i = 1;
        c0012a.b(CaptureRequest.CONTROL_AE_MODE, Integer.valueOf(h(i)));
        CaptureRequest.Key key3 = CaptureRequest.CONTROL_AWB_MODE;
        iArr = (int[]) this.f1114e.a(CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES);
        if (iArr != null) {
        }
        i3 = 0;
        c0012a.b(key3, Integer.valueOf(i3));
        j1 j1Var2 = this.k;
        Objects.requireNonNull(j1Var2);
        CaptureRequest.Key key22 = CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION;
        k1Var = j1Var2.f1078b;
        synchronized (k1Var.f1085a) {
        }
    }
}