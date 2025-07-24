package b.d.c;

import android.content.Context;
import androidx.camera.lifecycle.LifecycleCamera;
import androidx.camera.lifecycle.LifecycleCameraRepository;
import b.d.b.a1;
import b.d.b.d1.a0;
import b.d.b.d1.j1;
import b.d.b.d1.k1.c.f;
import b.d.b.d1.x;
import b.d.b.e0;
import b.d.b.e1.c;
import b.d.b.h0;
import b.d.b.j0;
import b.d.b.n0;
import b.d.b.o0;
import b.d.b.u0;
import b.j.b.d;
import b.t.e;
import b.t.h;
import b.t.i;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

/* compiled from: ProcessCameraProvider.java */
/* loaded from: classes.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public static final c f1717a = new c();

    /* renamed from: b  reason: collision with root package name */
    public final LifecycleCameraRepository f1718b = new LifecycleCameraRepository();

    /* renamed from: c  reason: collision with root package name */
    public n0 f1719c;

    public static ListenableFuture<c> b(Context context) {
        ListenableFuture<n0> c2;
        Objects.requireNonNull(context);
        Object obj = n0.f1647a;
        d.h(context, "Context must not be null.");
        synchronized (n0.f1647a) {
            boolean z = true;
            boolean z2 = n0.f1649c != null;
            c2 = n0.c();
            if (c2.isDone()) {
                try {
                    c2.get();
                } catch (InterruptedException e2) {
                    throw new RuntimeException("Unexpected thread interrupt. Should not be possible since future is already complete.", e2);
                } catch (ExecutionException unused) {
                    n0.f();
                    c2 = null;
                }
            }
            if (c2 == null) {
                if (!z2) {
                    o0.b b2 = n0.b(context);
                    if (b2 != null) {
                        if (n0.f1649c != null) {
                            z = false;
                        }
                        d.k(z, "CameraX has already been configured. To use a different configuration, shutdown() must be called.");
                        n0.f1649c = b2;
                        Integer num = (Integer) b2.getCameraXConfig().f(o0.v, null);
                        if (num != null) {
                            u0.f1672a = num.intValue();
                        }
                    } else {
                        throw new IllegalStateException("CameraX is not configured properly. The most likely cause is you did not include a default implementation in your build such as 'camera-camera2'.");
                    }
                }
                n0.d(context);
                c2 = n0.c();
            }
        }
        a aVar = a.f1714a;
        Executor f2 = b.b.a.f();
        b.d.b.d1.k1.c.c cVar = new b.d.b.d1.k1.c.c(new f(aVar), c2);
        c2.addListener(cVar, f2);
        return cVar;
    }

    public e0 a(h hVar, j0 j0Var, a1... a1VarArr) {
        LifecycleCamera lifecycleCamera;
        Collection<LifecycleCamera> unmodifiableCollection;
        LifecycleCamera lifecycleCamera2;
        boolean contains;
        b.b.a.c();
        LinkedHashSet linkedHashSet = new LinkedHashSet(j0Var.f1631c);
        for (a1 a1Var : a1VarArr) {
            j0 t = a1Var.f1384f.t(null);
            if (t != null) {
                Iterator<h0> it = t.f1631c.iterator();
                while (it.hasNext()) {
                    linkedHashSet.add(it.next());
                }
            }
        }
        LinkedHashSet<a0> a2 = new j0(linkedHashSet).a(this.f1719c.f1652f.a());
        c.b bVar = new c.b(a2);
        LifecycleCameraRepository lifecycleCameraRepository = this.f1718b;
        synchronized (lifecycleCameraRepository.f189a) {
            lifecycleCamera = lifecycleCameraRepository.f190b.get(new b(hVar, bVar));
        }
        LifecycleCameraRepository lifecycleCameraRepository2 = this.f1718b;
        synchronized (lifecycleCameraRepository2.f189a) {
            unmodifiableCollection = Collections.unmodifiableCollection(lifecycleCameraRepository2.f190b.values());
        }
        for (a1 a1Var2 : a1VarArr) {
            for (LifecycleCamera lifecycleCamera3 : unmodifiableCollection) {
                synchronized (lifecycleCamera3.f185a) {
                    contains = ((ArrayList) lifecycleCamera3.f187c.k()).contains(a1Var2);
                }
                if (contains && lifecycleCamera3 != lifecycleCamera) {
                    throw new IllegalStateException(String.format("Use case %s already bound to a different lifecycle.", a1Var2));
                }
            }
        }
        if (lifecycleCamera == null) {
            LifecycleCameraRepository lifecycleCameraRepository3 = this.f1718b;
            n0 n0Var = this.f1719c;
            x xVar = n0Var.m;
            if (xVar != null) {
                j1 j1Var = n0Var.n;
                if (j1Var != null) {
                    b.d.b.e1.c cVar = new b.d.b.e1.c(a2, xVar, j1Var);
                    synchronized (lifecycleCameraRepository3.f189a) {
                        d.e(lifecycleCameraRepository3.f190b.get(new b(hVar, cVar.f1601e)) == null, "LifecycleCamera already exists for the given LifecycleOwner and set of cameras");
                        if (((i) hVar.getLifecycle()).f2579b != e.b.DESTROYED) {
                            lifecycleCamera2 = new LifecycleCamera(hVar, cVar);
                            if (((ArrayList) cVar.k()).isEmpty()) {
                                lifecycleCamera2.m();
                            }
                            lifecycleCameraRepository3.d(lifecycleCamera2);
                        } else {
                            throw new IllegalArgumentException("Trying to create LifecycleCamera with destroyed lifecycle.");
                        }
                    }
                    lifecycleCamera = lifecycleCamera2;
                } else {
                    throw new IllegalStateException("CameraX not initialized yet.");
                }
            } else {
                throw new IllegalStateException("CameraX not initialized yet.");
            }
        }
        if (a1VarArr.length != 0) {
            this.f1718b.a(lifecycleCamera, null, Arrays.asList(a1VarArr));
        }
        return lifecycleCamera;
    }

    public void c() {
        b.b.a.c();
        LifecycleCameraRepository lifecycleCameraRepository = this.f1718b;
        synchronized (lifecycleCameraRepository.f189a) {
            for (LifecycleCameraRepository.a aVar : lifecycleCameraRepository.f190b.keySet()) {
                LifecycleCamera lifecycleCamera = lifecycleCameraRepository.f190b.get(aVar);
                synchronized (lifecycleCamera.f185a) {
                    b.d.b.e1.c cVar = lifecycleCamera.f187c;
                    cVar.l(cVar.k());
                }
                lifecycleCameraRepository.f(lifecycleCamera.k());
            }
        }
    }
}