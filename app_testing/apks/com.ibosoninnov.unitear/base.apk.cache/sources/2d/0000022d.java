package b.d.a.e;

import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.util.ArrayMap;
import android.view.Surface;
import b.d.a.d.c;
import b.d.a.e.g1;
import b.d.a.e.p1;
import b.d.a.e.u1;
import b.d.b.d1.f0;
import b.d.b.d1.i0;
import b.d.b.d1.j0;
import b.d.b.d1.k1.c.g;
import b.d.b.d1.k1.c.h;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executor;

/* compiled from: CaptureSession.java */
/* loaded from: classes.dex */
public final class g1 {

    /* renamed from: e  reason: collision with root package name */
    public t1 f1043e;

    /* renamed from: f  reason: collision with root package name */
    public p1 f1044f;

    /* renamed from: g  reason: collision with root package name */
    public volatile b.d.b.d1.b1 f1045g;
    public c l;
    public ListenableFuture<Void> m;
    public b.g.a.b<Void> n;

    /* renamed from: a  reason: collision with root package name */
    public final Object f1039a = new Object();

    /* renamed from: b  reason: collision with root package name */
    public final List<b.d.b.d1.f0> f1040b = new ArrayList();

    /* renamed from: c  reason: collision with root package name */
    public final CameraCaptureSession.CaptureCallback f1041c = new a(this);

    /* renamed from: h  reason: collision with root package name */
    public volatile b.d.b.d1.i0 f1046h = b.d.b.d1.w0.q;
    public b.d.a.d.c i = b.d.a.d.c.d();
    public Map<b.d.b.d1.j0, Surface> j = new HashMap();
    public List<b.d.b.d1.j0> k = Collections.emptyList();
    public final b.d.a.e.y1.q.e o = new b.d.a.e.y1.q.e();

    /* renamed from: d  reason: collision with root package name */
    public final d f1042d = new d();

    /* compiled from: CaptureSession.java */
    /* loaded from: classes.dex */
    public class a extends CameraCaptureSession.CaptureCallback {
        public a(g1 g1Var) {
        }

        @Override // android.hardware.camera2.CameraCaptureSession.CaptureCallback
        public void onCaptureCompleted(CameraCaptureSession cameraCaptureSession, CaptureRequest captureRequest, TotalCaptureResult totalCaptureResult) {
        }
    }

    /* compiled from: CaptureSession.java */
    /* loaded from: classes.dex */
    public class b implements b.d.b.d1.k1.c.d<Void> {
        public b() {
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            g1.this.f1043e.a();
            synchronized (g1.this.f1039a) {
                int ordinal = g1.this.l.ordinal();
                if ((ordinal == 3 || ordinal == 5 || ordinal == 6) && !(th instanceof CancellationException)) {
                    b.d.b.u0.d("CaptureSession", "Opening session with fail " + g1.this.l, th);
                    g1.this.b();
                }
            }
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r1) {
        }
    }

    /* compiled from: CaptureSession.java */
    /* loaded from: classes.dex */
    public enum c {
        UNINITIALIZED,
        INITIALIZED,
        GET_SURFACE,
        OPENING,
        OPENED,
        CLOSED,
        RELEASING,
        RELEASED
    }

    /* compiled from: CaptureSession.java */
    /* loaded from: classes.dex */
    public final class d extends p1.a {
        public d() {
        }

        @Override // b.d.a.e.p1.a
        public void m(p1 p1Var) {
            synchronized (g1.this.f1039a) {
                if (g1.this.l != c.UNINITIALIZED) {
                    b.d.b.u0.a("CaptureSession", "CameraCaptureSession.onClosed()", null);
                    g1.this.b();
                } else {
                    throw new IllegalStateException("onClosed() should not be possible in state: " + g1.this.l);
                }
            }
        }

        @Override // b.d.a.e.p1.a
        public void n(p1 p1Var) {
            synchronized (g1.this.f1039a) {
                switch (g1.this.l.ordinal()) {
                    case 0:
                    case 1:
                    case 2:
                    case 4:
                    case 7:
                        throw new IllegalStateException("onConfigureFailed() should not be possible in state: " + g1.this.l);
                    case 3:
                    case 5:
                    case 6:
                        g1.this.b();
                        break;
                }
                b.d.b.u0.b("CaptureSession", "CameraCaptureSession.onConfigureFailed() " + g1.this.l, null);
            }
        }

        @Override // b.d.a.e.p1.a
        public void o(p1 p1Var) {
            synchronized (g1.this.f1039a) {
                switch (g1.this.l.ordinal()) {
                    case 0:
                    case 1:
                    case 2:
                    case 4:
                    case 7:
                        throw new IllegalStateException("onConfigured() should not be possible in state: " + g1.this.l);
                    case 3:
                        g1 g1Var = g1.this;
                        g1Var.l = c.OPENED;
                        g1Var.f1044f = p1Var;
                        if (g1Var.f1045g != null) {
                            c.a c2 = g1.this.i.c();
                            ArrayList arrayList = new ArrayList();
                            for (b.d.a.d.b bVar : c2.f1012a) {
                                Objects.requireNonNull(bVar);
                            }
                            if (!arrayList.isEmpty()) {
                                g1 g1Var2 = g1.this;
                                g1Var2.c(g1Var2.j(arrayList));
                            }
                        }
                        b.d.b.u0.a("CaptureSession", "Attempting to send capture request onConfigured", null);
                        g1.this.f();
                        g1.this.e();
                        break;
                    case 5:
                        g1.this.f1044f = p1Var;
                        break;
                    case 6:
                        p1Var.close();
                        break;
                }
                b.d.b.u0.a("CaptureSession", "CameraCaptureSession.onConfigured() mState=" + g1.this.l, null);
            }
        }

        @Override // b.d.a.e.p1.a
        public void p(p1 p1Var) {
            synchronized (g1.this.f1039a) {
                if (g1.this.l.ordinal() != 0) {
                    b.d.b.u0.a("CaptureSession", "CameraCaptureSession.onReady() " + g1.this.l, null);
                } else {
                    throw new IllegalStateException("onReady() should not be possible in state: " + g1.this.l);
                }
            }
        }
    }

    public g1() {
        this.l = c.UNINITIALIZED;
        this.l = c.INITIALIZED;
    }

    public static b.d.b.d1.i0 g(List<b.d.b.d1.f0> list) {
        b.d.b.d1.u0 y = b.d.b.d1.u0.y();
        for (b.d.b.d1.f0 f0Var : list) {
            b.d.b.d1.i0 i0Var = f0Var.f1463d;
            for (i0.a<?> aVar : i0Var.e()) {
                Object f2 = i0Var.f(aVar, null);
                if (y.b(aVar)) {
                    Object f3 = y.f(aVar, null);
                    if (!Objects.equals(f3, f2)) {
                        StringBuilder x = c.b.a.a.a.x("Detect conflicting option ");
                        x.append(aVar.a());
                        x.append(" : ");
                        x.append(f2);
                        x.append(" != ");
                        x.append(f3);
                        b.d.b.u0.a("CaptureSession", x.toString(), null);
                    }
                } else {
                    y.A(aVar, i0.c.OPTIONAL, f2);
                }
            }
        }
        return y;
    }

    public final CameraCaptureSession.CaptureCallback a(List<b.d.b.d1.q> list, CameraCaptureSession.CaptureCallback... captureCallbackArr) {
        CameraCaptureSession.CaptureCallback s0Var;
        ArrayList arrayList = new ArrayList(list.size() + captureCallbackArr.length);
        for (b.d.b.d1.q qVar : list) {
            if (qVar == null) {
                s0Var = null;
            } else {
                ArrayList arrayList2 = new ArrayList();
                f1.a(qVar, arrayList2);
                if (arrayList2.size() == 1) {
                    s0Var = (CameraCaptureSession.CaptureCallback) arrayList2.get(0);
                } else {
                    s0Var = new s0(arrayList2);
                }
            }
            arrayList.add(s0Var);
        }
        Collections.addAll(arrayList, captureCallbackArr);
        return new s0(arrayList);
    }

    public void b() {
        c cVar = this.l;
        c cVar2 = c.RELEASED;
        if (cVar == cVar2) {
            b.d.b.u0.a("CaptureSession", "Skipping finishClose due to being state RELEASED.", null);
            return;
        }
        this.l = cVar2;
        this.f1044f = null;
        for (b.d.b.d1.j0 j0Var : this.k) {
            j0Var.b();
        }
        this.k.clear();
        b.g.a.b<Void> bVar = this.n;
        if (bVar != null) {
            bVar.a(null);
            this.n = null;
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:55:0x0128, code lost:
        r12.f1044f.h();
        r2.f1241b = new b.d.a.e.x(r12);
     */
    /* JADX WARN: Removed duplicated region for block: B:49:0x010e A[Catch: CameraAccessException -> 0x0140, TryCatch #0 {CameraAccessException -> 0x0140, blocks: (B:6:0x000a, B:7:0x001f, B:9:0x0027, B:11:0x0037, B:12:0x003d, B:13:0x0045, B:15:0x004b, B:17:0x0059, B:21:0x0073, B:24:0x0078, B:26:0x0081, B:27:0x008a, B:29:0x00a6, B:31:0x00ac, B:32:0x00b7, B:34:0x00bd, B:35:0x00c7, B:37:0x00d1, B:39:0x00f0, B:38:0x00eb, B:40:0x00f5, B:42:0x00fb, B:46:0x0104, B:47:0x0108, B:49:0x010e, B:55:0x0128, B:56:0x0134, B:57:0x013a), top: B:62:0x000a }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void c(List<b.d.b.d1.f0> list) {
        boolean z;
        if (list.isEmpty()) {
            return;
        }
        try {
            y0 y0Var = new y0();
            ArrayList arrayList = new ArrayList();
            b.d.b.u0.a("CaptureSession", "Issuing capture request.", null);
            boolean z2 = false;
            boolean z3 = false;
            for (b.d.b.d1.f0 f0Var : list) {
                if (f0Var.a().isEmpty()) {
                    b.d.b.u0.a("CaptureSession", "Skipping issuing empty capture request.", null);
                } else {
                    Iterator<b.d.b.d1.j0> it = f0Var.a().iterator();
                    while (true) {
                        if (!it.hasNext()) {
                            z = true;
                            break;
                        }
                        b.d.b.d1.j0 next = it.next();
                        if (!this.j.containsKey(next)) {
                            b.d.b.u0.a("CaptureSession", "Skipping capture request with invalid surface: " + next, null);
                            z = false;
                            break;
                        }
                    }
                    if (z) {
                        if (f0Var.f1464e == 2) {
                            z3 = true;
                        }
                        f0.a aVar = new f0.a(f0Var);
                        if (this.f1045g != null) {
                            aVar.c(this.f1045g.f1419f.f1463d);
                        }
                        aVar.c(this.f1046h);
                        aVar.c(f0Var.f1463d);
                        CaptureRequest b2 = b.b.a.b(aVar.d(), this.f1044f.f(), this.j);
                        if (b2 == null) {
                            b.d.b.u0.a("CaptureSession", "Skipping issuing request without surface.", null);
                            return;
                        }
                        ArrayList arrayList2 = new ArrayList();
                        for (b.d.b.d1.q qVar : f0Var.f1465f) {
                            f1.a(qVar, arrayList2);
                        }
                        List<CameraCaptureSession.CaptureCallback> list2 = y0Var.f1240a.get(b2);
                        if (list2 != null) {
                            ArrayList arrayList3 = new ArrayList(list2.size() + arrayList2.size());
                            arrayList3.addAll(arrayList2);
                            arrayList3.addAll(list2);
                            y0Var.f1240a.put(b2, arrayList3);
                        } else {
                            y0Var.f1240a.put(b2, arrayList2);
                        }
                        arrayList.add(b2);
                    }
                }
            }
            if (!arrayList.isEmpty()) {
                if (this.o.f1354a && z3) {
                    Iterator it2 = arrayList.iterator();
                    while (it2.hasNext()) {
                        int intValue = ((Integer) ((CaptureRequest) it2.next()).get(CaptureRequest.CONTROL_AE_MODE)).intValue();
                        if (intValue == 2 || intValue == 3) {
                            z2 = true;
                            break;
                        }
                        while (it2.hasNext()) {
                        }
                    }
                }
                this.f1044f.c(arrayList, y0Var);
                return;
            }
            b.d.b.u0.a("CaptureSession", "Skipping issuing burst request due to no valid request elements", null);
        } catch (CameraAccessException e2) {
            StringBuilder x = c.b.a.a.a.x("Unable to access camera: ");
            x.append(e2.getMessage());
            b.d.b.u0.b("CaptureSession", x.toString(), null);
            Thread.dumpStack();
        }
    }

    public void d(List<b.d.b.d1.f0> list) {
        synchronized (this.f1039a) {
            switch (this.l.ordinal()) {
                case 0:
                    throw new IllegalStateException("issueCaptureRequests() should not be possible in state: " + this.l);
                case 1:
                case 2:
                case 3:
                    this.f1040b.addAll(list);
                    break;
                case 4:
                    this.f1040b.addAll(list);
                    e();
                    break;
                case 5:
                case 6:
                case 7:
                    throw new IllegalStateException("Cannot issue capture request on a closed/released session.");
            }
        }
    }

    public void e() {
        if (this.f1040b.isEmpty()) {
            return;
        }
        try {
            c(this.f1040b);
        } finally {
            this.f1040b.clear();
        }
    }

    public void f() {
        if (this.f1045g == null) {
            b.d.b.u0.a("CaptureSession", "Skipping issueRepeatingCaptureRequests for no configuration case.", null);
            return;
        }
        b.d.b.d1.f0 f0Var = this.f1045g.f1419f;
        if (f0Var.a().isEmpty()) {
            b.d.b.u0.a("CaptureSession", "Skipping issueRepeatingCaptureRequests for no surface.", null);
            try {
                this.f1044f.h();
                return;
            } catch (CameraAccessException e2) {
                StringBuilder x = c.b.a.a.a.x("Unable to access camera: ");
                x.append(e2.getMessage());
                b.d.b.u0.b("CaptureSession", x.toString(), null);
                Thread.dumpStack();
                return;
            }
        }
        try {
            b.d.b.u0.a("CaptureSession", "Issuing request for session.", null);
            f0.a aVar = new f0.a(f0Var);
            c.a c2 = this.i.c();
            ArrayList arrayList = new ArrayList();
            for (b.d.a.d.b bVar : c2.f1012a) {
                Objects.requireNonNull(bVar);
            }
            this.f1046h = g(arrayList);
            aVar.c(this.f1046h);
            CaptureRequest b2 = b.b.a.b(aVar.d(), this.f1044f.f(), this.j);
            if (b2 == null) {
                b.d.b.u0.a("CaptureSession", "Skipping issuing empty request for session.", null);
            } else {
                this.f1044f.g(b2, a(f0Var.f1465f, this.f1041c));
            }
        } catch (CameraAccessException e3) {
            StringBuilder x2 = c.b.a.a.a.x("Unable to access camera: ");
            x2.append(e3.getMessage());
            b.d.b.u0.b("CaptureSession", x2.toString(), null);
            Thread.dumpStack();
        }
    }

    public ListenableFuture<Void> h(final b.d.b.d1.b1 b1Var, final CameraDevice cameraDevice, t1 t1Var) {
        synchronized (this.f1039a) {
            if (this.l.ordinal() != 1) {
                b.d.b.u0.b("CaptureSession", "Open not allowed in state: " + this.l, null);
                return new h.a(new IllegalStateException("open() should not allow the state: " + this.l));
            }
            this.l = c.GET_SURFACE;
            ArrayList arrayList = new ArrayList(b1Var.b());
            this.k = arrayList;
            this.f1043e = t1Var;
            b.d.b.d1.k1.c.e c2 = b.d.b.d1.k1.c.e.a(t1Var.f1196a.a(arrayList, 5000L)).c(new b.d.b.d1.k1.c.b() { // from class: b.d.a.e.w
                @Override // b.d.b.d1.k1.c.b
                public final ListenableFuture apply(Object obj) {
                    ListenableFuture<Void> aVar;
                    g1 g1Var = g1.this;
                    b.d.b.d1.b1 b1Var2 = b1Var;
                    CameraDevice cameraDevice2 = cameraDevice;
                    List list = (List) obj;
                    synchronized (g1Var.f1039a) {
                        int ordinal = g1Var.l.ordinal();
                        if (ordinal != 0 && ordinal != 1) {
                            if (ordinal == 2) {
                                try {
                                    b.b.a.j(g1Var.k);
                                    g1Var.j.clear();
                                    for (int i = 0; i < list.size(); i++) {
                                        g1Var.j.put(g1Var.k.get(i), (Surface) list.get(i));
                                    }
                                    ArrayList arrayList2 = new ArrayList(new HashSet(list));
                                    g1Var.l = g1.c.OPENING;
                                    CaptureRequest captureRequest = null;
                                    b.d.b.u0.a("CaptureSession", "Opening capture session.", null);
                                    u1 u1Var = new u1(Arrays.asList(g1Var.f1042d, new u1.a(b1Var2.f1416c)));
                                    b.d.a.d.c cVar = (b.d.a.d.c) b1Var2.f1419f.f1463d.f(b.d.a.d.a.v, b.d.a.d.c.d());
                                    g1Var.i = cVar;
                                    c.a c3 = cVar.c();
                                    ArrayList arrayList3 = new ArrayList();
                                    for (b.d.a.d.b bVar : c3.f1012a) {
                                        Objects.requireNonNull(bVar);
                                    }
                                    f0.a aVar2 = new f0.a(b1Var2.f1419f);
                                    Iterator it = arrayList3.iterator();
                                    while (it.hasNext()) {
                                        aVar2.c(((b.d.b.d1.f0) it.next()).f1463d);
                                    }
                                    ArrayList arrayList4 = new ArrayList();
                                    Iterator it2 = arrayList2.iterator();
                                    while (it2.hasNext()) {
                                        arrayList4.add(new b.d.a.e.y1.o.b((Surface) it2.next()));
                                    }
                                    r1 r1Var = (r1) g1Var.f1043e.f1196a;
                                    r1Var.f1186f = u1Var;
                                    b.d.a.e.y1.o.g gVar = new b.d.a.e.y1.o.g(0, arrayList4, r1Var.f1184d, new q1(r1Var));
                                    try {
                                        b.d.b.d1.f0 d2 = aVar2.d();
                                        if (cameraDevice2 != null) {
                                            CaptureRequest.Builder createCaptureRequest = cameraDevice2.createCaptureRequest(d2.f1464e);
                                            b.b.a.a(createCaptureRequest, d2.f1463d);
                                            captureRequest = createCaptureRequest.build();
                                        }
                                        if (captureRequest != null) {
                                            gVar.f1338a.f(captureRequest);
                                        }
                                        aVar = g1Var.f1043e.f1196a.i(cameraDevice2, gVar);
                                    } catch (CameraAccessException e2) {
                                        aVar = new h.a<>(e2);
                                    }
                                } catch (j0.a e3) {
                                    g1Var.k.clear();
                                    aVar = new h.a<>(e3);
                                }
                            } else if (ordinal != 4) {
                                aVar = new h.a<>(new CancellationException("openCaptureSession() not execute in state: " + g1Var.l));
                            }
                        }
                        aVar = new h.a<>(new IllegalStateException("openCaptureSession() should not be possible in state: " + g1Var.l));
                    }
                    return aVar;
                }
            }, ((r1) this.f1043e.f1196a).f1184d);
            b bVar = new b();
            Executor executor = ((r1) this.f1043e.f1196a).f1184d;
            c2.f1543b.addListener(new g.d(c2, bVar), executor);
            return b.d.b.d1.k1.c.g.d(c2);
        }
    }

    public void i(b.d.b.d1.b1 b1Var) {
        synchronized (this.f1039a) {
            switch (this.l.ordinal()) {
                case 0:
                    throw new IllegalStateException("setSessionConfig() should not be possible in state: " + this.l);
                case 1:
                case 2:
                case 3:
                    this.f1045g = b1Var;
                    break;
                case 4:
                    this.f1045g = b1Var;
                    if (!this.j.keySet().containsAll(b1Var.b())) {
                        b.d.b.u0.b("CaptureSession", "Does not have the proper configured lists", null);
                        return;
                    }
                    b.d.b.u0.a("CaptureSession", "Attempting to submit CaptureRequest after setting", null);
                    f();
                    break;
                case 5:
                case 6:
                case 7:
                    throw new IllegalStateException("Session configuration cannot be set on a closed/released session.");
            }
        }
    }

    public List<b.d.b.d1.f0> j(List<b.d.b.d1.f0> list) {
        ArrayList arrayList = new ArrayList();
        for (b.d.b.d1.f0 f0Var : list) {
            HashSet hashSet = new HashSet();
            b.d.b.d1.u0.y();
            ArrayList arrayList2 = new ArrayList();
            new ArrayMap();
            hashSet.addAll(f0Var.f1462c);
            b.d.b.d1.u0 z = b.d.b.d1.u0.z(f0Var.f1463d);
            arrayList2.addAll(f0Var.f1465f);
            boolean z2 = f0Var.f1466g;
            b.d.b.d1.g1 g1Var = f0Var.f1467h;
            ArrayMap arrayMap = new ArrayMap();
            for (String str : g1Var.f1480b.keySet()) {
                arrayMap.put(str, g1Var.a(str));
            }
            b.d.b.d1.v0 v0Var = new b.d.b.d1.v0(arrayMap);
            for (b.d.b.d1.j0 j0Var : this.f1045g.f1419f.a()) {
                hashSet.add(j0Var);
            }
            ArrayList arrayList3 = new ArrayList(hashSet);
            b.d.b.d1.w0 x = b.d.b.d1.w0.x(z);
            b.d.b.d1.g1 g1Var2 = b.d.b.d1.g1.f1479a;
            ArrayMap arrayMap2 = new ArrayMap();
            for (String str2 : v0Var.f1480b.keySet()) {
                arrayMap2.put(str2, v0Var.a(str2));
            }
            arrayList.add(new b.d.b.d1.f0(arrayList3, x, 1, arrayList2, z2, new b.d.b.d1.g1(arrayMap2)));
        }
        return arrayList;
    }
}