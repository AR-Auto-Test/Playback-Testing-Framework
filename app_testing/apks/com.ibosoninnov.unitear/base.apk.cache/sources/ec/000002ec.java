package b.d.b.d1;

import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.util.ArrayMap;
import b.d.b.d1.f0;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/* compiled from: SessionConfig.java */
/* loaded from: classes.dex */
public final class b1 {

    /* renamed from: a  reason: collision with root package name */
    public final List<j0> f1414a;

    /* renamed from: b  reason: collision with root package name */
    public final List<CameraDevice.StateCallback> f1415b;

    /* renamed from: c  reason: collision with root package name */
    public final List<CameraCaptureSession.StateCallback> f1416c;

    /* renamed from: d  reason: collision with root package name */
    public final List<q> f1417d;

    /* renamed from: e  reason: collision with root package name */
    public final List<c> f1418e;

    /* renamed from: f  reason: collision with root package name */
    public final f0 f1419f;

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Set<j0> f1420a = new HashSet();

        /* renamed from: b  reason: collision with root package name */
        public final f0.a f1421b = new f0.a();

        /* renamed from: c  reason: collision with root package name */
        public final List<CameraDevice.StateCallback> f1422c = new ArrayList();

        /* renamed from: d  reason: collision with root package name */
        public final List<CameraCaptureSession.StateCallback> f1423d = new ArrayList();

        /* renamed from: e  reason: collision with root package name */
        public final List<c> f1424e = new ArrayList();

        /* renamed from: f  reason: collision with root package name */
        public final List<q> f1425f = new ArrayList();
    }

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public static class b extends a {
        public static b e(i1<?> i1Var) {
            d v = i1Var.v(null);
            if (v != null) {
                b bVar = new b();
                v.a(i1Var, bVar);
                return bVar;
            }
            StringBuilder x = c.b.a.a.a.x("Implementation is missing option unpacker for ");
            x.append(i1Var.p(i1Var.toString()));
            throw new IllegalStateException(x.toString());
        }

        public void a(q qVar) {
            this.f1421b.b(qVar);
            this.f1425f.add(qVar);
        }

        public void b(CameraDevice.StateCallback stateCallback) {
            if (!this.f1422c.contains(stateCallback)) {
                this.f1422c.add(stateCallback);
                return;
            }
            throw new IllegalArgumentException("Duplicate device state callback.");
        }

        public void c(CameraCaptureSession.StateCallback stateCallback) {
            if (!this.f1423d.contains(stateCallback)) {
                this.f1423d.add(stateCallback);
                return;
            }
            throw new IllegalArgumentException("Duplicate session state callback.");
        }

        public b1 d() {
            return new b1(new ArrayList(this.f1420a), this.f1422c, this.f1423d, this.f1425f, this.f1424e, this.f1421b.d());
        }
    }

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public interface c {
        void a(b1 b1Var, e eVar);
    }

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public interface d {
        void a(i1<?> i1Var, b bVar);
    }

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public enum e {
        SESSION_ERROR_SURFACE_NEEDS_RESET,
        SESSION_ERROR_UNKNOWN
    }

    /* compiled from: SessionConfig.java */
    /* loaded from: classes.dex */
    public static final class f extends a {

        /* renamed from: g  reason: collision with root package name */
        public boolean f1429g = true;

        /* renamed from: h  reason: collision with root package name */
        public boolean f1430h = false;

        public void a(b1 b1Var) {
            Map<String, Integer> map;
            f0 f0Var = b1Var.f1419f;
            int i = f0Var.f1464e;
            if (i != -1) {
                if (!this.f1430h) {
                    this.f1421b.f1470c = i;
                    this.f1430h = true;
                } else if (this.f1421b.f1470c != i) {
                    StringBuilder x = c.b.a.a.a.x("Invalid configuration due to template type: ");
                    x.append(this.f1421b.f1470c);
                    x.append(" != ");
                    x.append(f0Var.f1464e);
                    b.d.b.u0.a("ValidatingBuilder", x.toString(), null);
                    this.f1429g = false;
                }
            }
            g1 g1Var = b1Var.f1419f.f1467h;
            Map<String, Integer> map2 = this.f1421b.f1473f.f1480b;
            if (map2 != null && (map = g1Var.f1480b) != null) {
                map2.putAll(map);
            }
            this.f1422c.addAll(b1Var.f1415b);
            this.f1423d.addAll(b1Var.f1416c);
            this.f1421b.a(b1Var.f1419f.f1465f);
            this.f1425f.addAll(b1Var.f1417d);
            this.f1424e.addAll(b1Var.f1418e);
            this.f1420a.addAll(b1Var.b());
            this.f1421b.f1468a.addAll(f0Var.a());
            if (!this.f1420a.containsAll(this.f1421b.f1468a)) {
                b.d.b.u0.a("ValidatingBuilder", "Invalid configuration due to capture request surfaces are not a subset of surfaces", null);
                this.f1429g = false;
            }
            this.f1421b.c(f0Var.f1463d);
        }

        public b1 b() {
            if (this.f1429g) {
                return new b1(new ArrayList(this.f1420a), this.f1422c, this.f1423d, this.f1425f, this.f1424e, this.f1421b.d());
            }
            throw new IllegalArgumentException("Unsupported session configuration combination");
        }
    }

    public b1(List<j0> list, List<CameraDevice.StateCallback> list2, List<CameraCaptureSession.StateCallback> list3, List<q> list4, List<c> list5, f0 f0Var) {
        this.f1414a = list;
        this.f1415b = Collections.unmodifiableList(list2);
        this.f1416c = Collections.unmodifiableList(list3);
        this.f1417d = Collections.unmodifiableList(list4);
        this.f1418e = Collections.unmodifiableList(list5);
        this.f1419f = f0Var;
    }

    public static b1 a() {
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList(0);
        ArrayList arrayList3 = new ArrayList(0);
        ArrayList arrayList4 = new ArrayList(0);
        ArrayList arrayList5 = new ArrayList(0);
        HashSet hashSet = new HashSet();
        u0 y = u0.y();
        ArrayList arrayList6 = new ArrayList();
        v0 v0Var = new v0(new ArrayMap());
        ArrayList arrayList7 = new ArrayList(hashSet);
        w0 x = w0.x(y);
        g1 g1Var = g1.f1479a;
        ArrayMap arrayMap = new ArrayMap();
        for (String str : v0Var.f1480b.keySet()) {
            arrayMap.put(str, v0Var.a(str));
        }
        return new b1(arrayList, arrayList2, arrayList3, arrayList4, arrayList5, new f0(arrayList7, x, -1, arrayList6, false, new g1(arrayMap)));
    }

    public List<j0> b() {
        return Collections.unmodifiableList(this.f1414a);
    }
}