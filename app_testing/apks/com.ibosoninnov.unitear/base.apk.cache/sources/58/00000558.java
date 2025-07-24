package b.t;

import androidx.lifecycle.CompositeGeneratedAdaptersObserver;
import androidx.lifecycle.FullLifecycleObserverAdapter;
import androidx.lifecycle.ReflectiveGenericLifecycleObserver;
import androidx.lifecycle.SingleGeneratedAdapterObserver;
import b.c.a.b.b;
import b.t.e;
import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/* compiled from: LifecycleRegistry.java */
/* loaded from: classes.dex */
public class i extends e {

    /* renamed from: c  reason: collision with root package name */
    public final WeakReference<h> f2580c;

    /* renamed from: a  reason: collision with root package name */
    public b.c.a.b.a<g, a> f2578a = new b.c.a.b.a<>();

    /* renamed from: d  reason: collision with root package name */
    public int f2581d = 0;

    /* renamed from: e  reason: collision with root package name */
    public boolean f2582e = false;

    /* renamed from: f  reason: collision with root package name */
    public boolean f2583f = false;

    /* renamed from: g  reason: collision with root package name */
    public ArrayList<e.b> f2584g = new ArrayList<>();

    /* renamed from: b  reason: collision with root package name */
    public e.b f2579b = e.b.INITIALIZED;

    /* compiled from: LifecycleRegistry.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public e.b f2585a;

        /* renamed from: b  reason: collision with root package name */
        public f f2586b;

        public a(g gVar, e.b bVar) {
            f reflectiveGenericLifecycleObserver;
            Map<Class<?>, Integer> map = k.f2587a;
            boolean z = gVar instanceof f;
            boolean z2 = gVar instanceof c;
            if (z && z2) {
                reflectiveGenericLifecycleObserver = new FullLifecycleObserverAdapter((c) gVar, (f) gVar);
            } else if (z2) {
                reflectiveGenericLifecycleObserver = new FullLifecycleObserverAdapter((c) gVar, null);
            } else if (z) {
                reflectiveGenericLifecycleObserver = (f) gVar;
            } else {
                Class<?> cls = gVar.getClass();
                if (k.c(cls) == 2) {
                    List<Constructor<? extends d>> list = k.f2588b.get(cls);
                    if (list.size() == 1) {
                        reflectiveGenericLifecycleObserver = new SingleGeneratedAdapterObserver(k.a(list.get(0), gVar));
                    } else {
                        d[] dVarArr = new d[list.size()];
                        for (int i = 0; i < list.size(); i++) {
                            dVarArr[i] = k.a(list.get(i), gVar);
                        }
                        reflectiveGenericLifecycleObserver = new CompositeGeneratedAdaptersObserver(dVarArr);
                    }
                } else {
                    reflectiveGenericLifecycleObserver = new ReflectiveGenericLifecycleObserver(gVar);
                }
            }
            this.f2586b = reflectiveGenericLifecycleObserver;
            this.f2585a = bVar;
        }

        public void a(h hVar, e.a aVar) {
            e.b c2 = i.c(aVar);
            this.f2585a = i.e(this.f2585a, c2);
            this.f2586b.e(hVar, aVar);
            this.f2585a = c2;
        }
    }

    public i(h hVar) {
        this.f2580c = new WeakReference<>(hVar);
    }

    public static e.b c(e.a aVar) {
        int ordinal = aVar.ordinal();
        if (ordinal != 0) {
            if (ordinal != 1) {
                if (ordinal == 2) {
                    return e.b.RESUMED;
                }
                if (ordinal != 3) {
                    if (ordinal != 4) {
                        if (ordinal == 5) {
                            return e.b.DESTROYED;
                        }
                        throw new IllegalArgumentException("Unexpected event value " + aVar);
                    }
                }
            }
            return e.b.STARTED;
        }
        return e.b.CREATED;
    }

    public static e.b e(e.b bVar, e.b bVar2) {
        return (bVar2 == null || bVar2.compareTo(bVar) >= 0) ? bVar : bVar2;
    }

    public static e.a i(e.b bVar) {
        int ordinal = bVar.ordinal();
        if (ordinal == 0 || ordinal == 1) {
            return e.a.ON_CREATE;
        }
        if (ordinal != 2) {
            if (ordinal != 3) {
                if (ordinal != 4) {
                    throw new IllegalArgumentException("Unexpected state value " + bVar);
                }
                throw new IllegalArgumentException();
            }
            return e.a.ON_RESUME;
        }
        return e.a.ON_START;
    }

    @Override // b.t.e
    public void a(g gVar) {
        h hVar;
        e.b bVar = this.f2579b;
        e.b bVar2 = e.b.DESTROYED;
        if (bVar != bVar2) {
            bVar2 = e.b.INITIALIZED;
        }
        a aVar = new a(gVar, bVar2);
        if (this.f2578a.d(gVar, aVar) == null && (hVar = this.f2580c.get()) != null) {
            boolean z = this.f2581d != 0 || this.f2582e;
            e.b b2 = b(gVar);
            this.f2581d++;
            while (aVar.f2585a.compareTo(b2) < 0 && this.f2578a.f991f.containsKey(gVar)) {
                this.f2584g.add(aVar.f2585a);
                aVar.a(hVar, i(aVar.f2585a));
                g();
                b2 = b(gVar);
            }
            if (!z) {
                h();
            }
            this.f2581d--;
        }
    }

    public final e.b b(g gVar) {
        b.c.a.b.a<g, a> aVar = this.f2578a;
        e.b bVar = null;
        b.c<g, a> cVar = aVar.f991f.containsKey(gVar) ? aVar.f991f.get(gVar).f999e : null;
        e.b bVar2 = cVar != null ? cVar.f997c.f2585a : null;
        if (!this.f2584g.isEmpty()) {
            ArrayList<e.b> arrayList = this.f2584g;
            bVar = arrayList.get(arrayList.size() - 1);
        }
        return e(e(this.f2579b, bVar2), bVar);
    }

    public void d(e.a aVar) {
        f(c(aVar));
    }

    public final void f(e.b bVar) {
        if (this.f2579b == bVar) {
            return;
        }
        this.f2579b = bVar;
        if (!this.f2582e && this.f2581d == 0) {
            this.f2582e = true;
            h();
            this.f2582e = false;
            return;
        }
        this.f2583f = true;
    }

    public final void g() {
        ArrayList<e.b> arrayList = this.f2584g;
        arrayList.remove(arrayList.size() - 1);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r4v4, resolved type: b.c.a.b.a<b.t.g, b.t.i$a> */
    /* JADX DEBUG: Multi-variable search result rejected for r5v7, resolved type: b.c.a.b.a<b.t.g, b.t.i$a> */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:14:0x002e  */
    /* JADX WARN: Removed duplicated region for block: B:73:0x0132 A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void h() {
        boolean z;
        e.a aVar;
        h hVar = this.f2580c.get();
        if (hVar == null) {
            throw new IllegalStateException("LifecycleOwner of this LifecycleRegistry is alreadygarbage collected. It is too late to change lifecycle state.");
        }
        while (true) {
            b.c.a.b.a<g, a> aVar2 = this.f2578a;
            if (aVar2.f995e != 0) {
                e.b bVar = aVar2.f992b.f997c.f2585a;
                e.b bVar2 = aVar2.f993c.f997c.f2585a;
                if (bVar != bVar2 || this.f2579b != bVar2) {
                    z = false;
                    if (z) {
                        this.f2583f = false;
                        if (this.f2579b.compareTo(aVar2.f992b.f997c.f2585a) < 0) {
                            b.c.a.b.a<g, a> aVar3 = this.f2578a;
                            b.C0011b c0011b = new b.C0011b(aVar3.f993c, aVar3.f992b);
                            aVar3.f994d.put(c0011b, Boolean.FALSE);
                            while (c0011b.hasNext() && !this.f2583f) {
                                Map.Entry entry = (Map.Entry) c0011b.next();
                                a aVar4 = (a) entry.getValue();
                                while (aVar4.f2585a.compareTo(this.f2579b) > 0 && !this.f2583f && this.f2578a.contains(entry.getKey())) {
                                    e.b bVar3 = aVar4.f2585a;
                                    int ordinal = bVar3.ordinal();
                                    if (ordinal == 0) {
                                        throw new IllegalArgumentException();
                                    }
                                    if (ordinal != 1) {
                                        if (ordinal == 2) {
                                            aVar = e.a.ON_DESTROY;
                                        } else if (ordinal == 3) {
                                            aVar = e.a.ON_STOP;
                                        } else if (ordinal == 4) {
                                            aVar = e.a.ON_PAUSE;
                                        } else {
                                            throw new IllegalArgumentException("Unexpected state value " + bVar3);
                                        }
                                        this.f2584g.add(c(aVar));
                                        aVar4.a(hVar, aVar);
                                        g();
                                    } else {
                                        throw new IllegalArgumentException();
                                    }
                                }
                            }
                        }
                        b.c<g, a> cVar = this.f2578a.f993c;
                        if (!this.f2583f && cVar != null && this.f2579b.compareTo(cVar.f997c.f2585a) > 0) {
                            b.c.a.b.b<g, a>.d b2 = this.f2578a.b();
                            while (b2.hasNext() && !this.f2583f) {
                                Map.Entry entry2 = (Map.Entry) b2.next();
                                a aVar5 = (a) entry2.getValue();
                                while (aVar5.f2585a.compareTo(this.f2579b) < 0 && !this.f2583f && this.f2578a.contains(entry2.getKey())) {
                                    this.f2584g.add(aVar5.f2585a);
                                    aVar5.a(hVar, i(aVar5.f2585a));
                                    g();
                                }
                            }
                        }
                    } else {
                        this.f2583f = false;
                        return;
                    }
                }
            }
            z = true;
            if (z) {
            }
        }
    }
}