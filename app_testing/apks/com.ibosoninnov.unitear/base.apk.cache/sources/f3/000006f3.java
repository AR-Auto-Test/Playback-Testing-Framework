package c.c.a;

import android.content.ComponentCallbacks2;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.util.Log;
import c.c.a.c;
import c.c.a.m.v.k;
import c.c.a.n.c;
import c.c.a.n.l;
import c.c.a.n.m;
import c.c.a.n.n;
import c.c.a.n.q;
import c.c.a.n.r;
import c.c.a.n.t;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CopyOnWriteArrayList;

/* compiled from: RequestManager.java */
/* loaded from: classes.dex */
public class i implements ComponentCallbacks2, m {

    /* renamed from: b  reason: collision with root package name */
    public static final c.c.a.q.f f3450b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.b f3451c;

    /* renamed from: d  reason: collision with root package name */
    public final Context f3452d;

    /* renamed from: e  reason: collision with root package name */
    public final l f3453e;

    /* renamed from: f  reason: collision with root package name */
    public final r f3454f;

    /* renamed from: g  reason: collision with root package name */
    public final q f3455g;

    /* renamed from: h  reason: collision with root package name */
    public final t f3456h;
    public final Runnable i;
    public final c.c.a.n.c j;
    public final CopyOnWriteArrayList<c.c.a.q.e<Object>> k;
    public c.c.a.q.f l;

    /* compiled from: RequestManager.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            i iVar = i.this;
            iVar.f3453e.a(iVar);
        }
    }

    /* compiled from: RequestManager.java */
    /* loaded from: classes.dex */
    public class b implements c.a {

        /* renamed from: a  reason: collision with root package name */
        public final r f3458a;

        public b(r rVar) {
            this.f3458a = rVar;
        }
    }

    static {
        c.c.a.q.f d2 = new c.c.a.q.f().d(Bitmap.class);
        d2.u = true;
        f3450b = d2;
        new c.c.a.q.f().d(c.c.a.m.x.g.c.class).u = true;
        new c.c.a.q.f().e(k.f3732b).k(f.LOW).o(true);
    }

    public i(c.c.a.b bVar, l lVar, q qVar, Context context) {
        c.c.a.n.c nVar;
        c.c.a.q.f fVar;
        r rVar = new r();
        c.c.a.n.d dVar = bVar.j;
        this.f3456h = new t();
        a aVar = new a();
        this.i = aVar;
        this.f3451c = bVar;
        this.f3453e = lVar;
        this.f3455g = qVar;
        this.f3454f = rVar;
        this.f3452d = context;
        Context applicationContext = context.getApplicationContext();
        b bVar2 = new b(rVar);
        Objects.requireNonNull((c.c.a.n.f) dVar);
        boolean z = b.j.c.a.a(applicationContext, "android.permission.ACCESS_NETWORK_STATE") == 0;
        if (Log.isLoggable("ConnectivityMonitor", 3)) {
            Log.d("ConnectivityMonitor", z ? "ACCESS_NETWORK_STATE permission granted, registering connectivity monitor" : "ACCESS_NETWORK_STATE permission missing, cannot register connectivity monitor");
        }
        if (z) {
            nVar = new c.c.a.n.e(applicationContext, bVar2);
        } else {
            nVar = new n();
        }
        this.j = nVar;
        if (c.c.a.s.j.h()) {
            c.c.a.s.j.f().post(aVar);
        } else {
            lVar.a(this);
        }
        lVar.a(nVar);
        this.k = new CopyOnWriteArrayList<>(bVar.f3414f.f3430f);
        d dVar2 = bVar.f3414f;
        synchronized (dVar2) {
            if (dVar2.k == null) {
                Objects.requireNonNull((c.a) dVar2.f3429e);
                c.c.a.q.f fVar2 = new c.c.a.q.f();
                fVar2.u = true;
                dVar2.k = fVar2;
            }
            fVar = dVar2.k;
        }
        synchronized (this) {
            c.c.a.q.f clone = fVar.clone();
            if (clone.u && !clone.w) {
                throw new IllegalStateException("You cannot auto lock an already locked options object, try clone() first");
            }
            clone.w = true;
            clone.u = true;
            this.l = clone;
        }
        synchronized (bVar.k) {
            if (!bVar.k.contains(this)) {
                bVar.k.add(this);
            } else {
                throw new IllegalStateException("Cannot register already registered manager");
            }
        }
    }

    public void i(c.c.a.q.j.h<?> hVar) {
        boolean z;
        if (hVar == null) {
            return;
        }
        boolean n = n(hVar);
        c.c.a.q.c f2 = hVar.f();
        if (n) {
            return;
        }
        c.c.a.b bVar = this.f3451c;
        synchronized (bVar.k) {
            Iterator<i> it = bVar.k.iterator();
            while (true) {
                if (!it.hasNext()) {
                    z = false;
                    break;
                } else if (it.next().n(hVar)) {
                    z = true;
                    break;
                }
            }
        }
        if (z || f2 == null) {
            return;
        }
        hVar.c(null);
        f2.clear();
    }

    public h<Drawable> j(Integer num) {
        PackageInfo packageInfo;
        String uuid;
        h hVar = new h(this.f3451c, this, Drawable.class, this.f3452d);
        h D = hVar.D(num);
        Context context = hVar.B;
        int i = c.c.a.r.a.f4167b;
        ConcurrentMap<String, c.c.a.m.m> concurrentMap = c.c.a.r.b.f4170a;
        String packageName = context.getPackageName();
        c.c.a.m.m mVar = c.c.a.r.b.f4170a.get(packageName);
        if (mVar == null) {
            try {
                packageInfo = context.getPackageManager().getPackageInfo(context.getPackageName(), 0);
            } catch (PackageManager.NameNotFoundException e2) {
                StringBuilder x = c.b.a.a.a.x("Cannot resolve info for");
                x.append(context.getPackageName());
                Log.e("AppVersionSignature", x.toString(), e2);
                packageInfo = null;
            }
            if (packageInfo != null) {
                uuid = String.valueOf(packageInfo.versionCode);
            } else {
                uuid = UUID.randomUUID().toString();
            }
            c.c.a.r.d dVar = new c.c.a.r.d(uuid);
            mVar = c.c.a.r.b.f4170a.putIfAbsent(packageName, dVar);
            if (mVar == null) {
                mVar = dVar;
            }
        }
        return D.a(new c.c.a.q.f().n(new c.c.a.r.a(context.getResources().getConfiguration().uiMode & 48, mVar)));
    }

    public h<Drawable> k(String str) {
        return new h(this.f3451c, this, Drawable.class, this.f3452d).D(str);
    }

    public synchronized void l() {
        r rVar = this.f3454f;
        rVar.f4100c = true;
        Iterator it = ((ArrayList) c.c.a.s.j.e(rVar.f4098a)).iterator();
        while (it.hasNext()) {
            c.c.a.q.c cVar = (c.c.a.q.c) it.next();
            if (cVar.isRunning()) {
                cVar.pause();
                rVar.f4099b.add(cVar);
            }
        }
    }

    public synchronized void m() {
        r rVar = this.f3454f;
        rVar.f4100c = false;
        Iterator it = ((ArrayList) c.c.a.s.j.e(rVar.f4098a)).iterator();
        while (it.hasNext()) {
            c.c.a.q.c cVar = (c.c.a.q.c) it.next();
            if (!cVar.i() && !cVar.isRunning()) {
                cVar.g();
            }
        }
        rVar.f4099b.clear();
    }

    public synchronized boolean n(c.c.a.q.j.h<?> hVar) {
        c.c.a.q.c f2 = hVar.f();
        if (f2 == null) {
            return true;
        }
        if (this.f3454f.a(f2)) {
            this.f3456h.f4108b.remove(hVar);
            hVar.c(null);
            return true;
        }
        return false;
    }

    @Override // android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
    }

    @Override // c.c.a.n.m
    public synchronized void onDestroy() {
        this.f3456h.onDestroy();
        for (c.c.a.q.j.h<?> hVar : c.c.a.s.j.e(this.f3456h.f4108b)) {
            i(hVar);
        }
        this.f3456h.f4108b.clear();
        r rVar = this.f3454f;
        Iterator it = ((ArrayList) c.c.a.s.j.e(rVar.f4098a)).iterator();
        while (it.hasNext()) {
            rVar.a((c.c.a.q.c) it.next());
        }
        rVar.f4099b.clear();
        this.f3453e.b(this);
        this.f3453e.b(this.j);
        c.c.a.s.j.f().removeCallbacks(this.i);
        c.c.a.b bVar = this.f3451c;
        synchronized (bVar.k) {
            if (bVar.k.contains(this)) {
                bVar.k.remove(this);
            } else {
                throw new IllegalStateException("Cannot unregister not yet registered manager");
            }
        }
    }

    @Override // android.content.ComponentCallbacks
    public void onLowMemory() {
    }

    @Override // c.c.a.n.m
    public synchronized void onStart() {
        m();
        this.f3456h.onStart();
    }

    @Override // c.c.a.n.m
    public synchronized void onStop() {
        l();
        this.f3456h.onStop();
    }

    @Override // android.content.ComponentCallbacks2
    public void onTrimMemory(int i) {
    }

    public synchronized String toString() {
        return super.toString() + "{tracker=" + this.f3454f + ", treeNode=" + this.f3455g + "}";
    }
}