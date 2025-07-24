package b.d.b.d1;

import android.util.ArrayMap;
import b.d.b.d1.i0;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/* compiled from: CaptureConfig.java */
/* loaded from: classes.dex */
public final class f0 {

    /* renamed from: a  reason: collision with root package name */
    public static final i0.a<Integer> f1460a = new n("camerax.core.captureConfig.rotation", Integer.TYPE, null);

    /* renamed from: b  reason: collision with root package name */
    public static final i0.a<Integer> f1461b = new n("camerax.core.captureConfig.jpegQuality", Integer.class, null);

    /* renamed from: c  reason: collision with root package name */
    public final List<j0> f1462c;

    /* renamed from: d  reason: collision with root package name */
    public final i0 f1463d;

    /* renamed from: e  reason: collision with root package name */
    public final int f1464e;

    /* renamed from: f  reason: collision with root package name */
    public final List<q> f1465f;

    /* renamed from: g  reason: collision with root package name */
    public final boolean f1466g;

    /* renamed from: h  reason: collision with root package name */
    public final g1 f1467h;

    /* compiled from: CaptureConfig.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public f0(List<j0> list, i0 i0Var, int i, List<q> list2, boolean z, g1 g1Var) {
        this.f1462c = list;
        this.f1463d = i0Var;
        this.f1464e = i;
        this.f1465f = Collections.unmodifiableList(list2);
        this.f1466g = z;
        this.f1467h = g1Var;
    }

    public List<j0> a() {
        return Collections.unmodifiableList(this.f1462c);
    }

    /* compiled from: CaptureConfig.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final Set<j0> f1468a;

        /* renamed from: b  reason: collision with root package name */
        public t0 f1469b;

        /* renamed from: c  reason: collision with root package name */
        public int f1470c;

        /* renamed from: d  reason: collision with root package name */
        public List<q> f1471d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f1472e;

        /* renamed from: f  reason: collision with root package name */
        public v0 f1473f;

        public a() {
            this.f1468a = new HashSet();
            this.f1469b = u0.y();
            this.f1470c = -1;
            this.f1471d = new ArrayList();
            this.f1472e = false;
            this.f1473f = new v0(new ArrayMap());
        }

        public void a(Collection<q> collection) {
            for (q qVar : collection) {
                b(qVar);
            }
        }

        public void b(q qVar) {
            if (!this.f1471d.contains(qVar)) {
                this.f1471d.add(qVar);
                return;
            }
            throw new IllegalArgumentException("duplicate camera capture callback");
        }

        public void c(i0 i0Var) {
            for (i0.a<?> aVar : i0Var.e()) {
                Object f2 = ((w0) this.f1469b).f(aVar, null);
                Object a2 = i0Var.a(aVar);
                if (f2 instanceof s0) {
                    ((s0) f2).f1588a.addAll(((s0) a2).b());
                } else {
                    if (a2 instanceof s0) {
                        a2 = ((s0) a2).clone();
                    }
                    ((u0) this.f1469b).A(aVar, i0Var.g(aVar), a2);
                }
            }
        }

        public f0 d() {
            ArrayList arrayList = new ArrayList(this.f1468a);
            w0 x = w0.x(this.f1469b);
            int i = this.f1470c;
            List<q> list = this.f1471d;
            boolean z = this.f1472e;
            v0 v0Var = this.f1473f;
            g1 g1Var = g1.f1479a;
            ArrayMap arrayMap = new ArrayMap();
            for (String str : v0Var.f1480b.keySet()) {
                arrayMap.put(str, v0Var.a(str));
            }
            return new f0(arrayList, x, i, list, z, new g1(arrayMap));
        }

        public a(f0 f0Var) {
            HashSet hashSet = new HashSet();
            this.f1468a = hashSet;
            this.f1469b = u0.y();
            this.f1470c = -1;
            this.f1471d = new ArrayList();
            this.f1472e = false;
            this.f1473f = new v0(new ArrayMap());
            hashSet.addAll(f0Var.f1462c);
            this.f1469b = u0.z(f0Var.f1463d);
            this.f1470c = f0Var.f1464e;
            this.f1471d.addAll(f0Var.f1465f);
            this.f1472e = f0Var.f1466g;
            g1 g1Var = f0Var.f1467h;
            ArrayMap arrayMap = new ArrayMap();
            for (String str : g1Var.f1480b.keySet()) {
                arrayMap.put(str, g1Var.a(str));
            }
            this.f1473f = new v0(arrayMap);
        }
    }
}