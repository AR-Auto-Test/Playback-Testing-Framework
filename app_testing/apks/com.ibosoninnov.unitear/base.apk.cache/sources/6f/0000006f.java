package androidx.camera.lifecycle;

import b.d.b.a1;
import b.d.b.b1;
import b.d.b.e1.c;
import b.d.c.b;
import b.j.b.d;
import b.t.e;
import b.t.g;
import b.t.h;
import b.t.i;
import b.t.o;
import com.google.auto.value.AutoValue;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/* loaded from: classes.dex */
public final class LifecycleCameraRepository {

    /* renamed from: a  reason: collision with root package name */
    public final Object f189a = new Object();

    /* renamed from: b  reason: collision with root package name */
    public final Map<a, LifecycleCamera> f190b = new HashMap();

    /* renamed from: c  reason: collision with root package name */
    public final Map<LifecycleCameraRepositoryObserver, Set<a>> f191c = new HashMap();

    /* renamed from: d  reason: collision with root package name */
    public final ArrayDeque<h> f192d = new ArrayDeque<>();

    /* loaded from: classes.dex */
    public static class LifecycleCameraRepositoryObserver implements g {

        /* renamed from: a  reason: collision with root package name */
        public final LifecycleCameraRepository f193a;

        /* renamed from: b  reason: collision with root package name */
        public final h f194b;

        public LifecycleCameraRepositoryObserver(h hVar, LifecycleCameraRepository lifecycleCameraRepository) {
            this.f194b = hVar;
            this.f193a = lifecycleCameraRepository;
        }

        @o(e.a.ON_DESTROY)
        public void onDestroy(h hVar) {
            LifecycleCameraRepository lifecycleCameraRepository = this.f193a;
            synchronized (lifecycleCameraRepository.f189a) {
                LifecycleCameraRepositoryObserver b2 = lifecycleCameraRepository.b(hVar);
                if (b2 == null) {
                    return;
                }
                lifecycleCameraRepository.f(hVar);
                for (a aVar : lifecycleCameraRepository.f191c.get(b2)) {
                    lifecycleCameraRepository.f190b.remove(aVar);
                }
                lifecycleCameraRepository.f191c.remove(b2);
                ((i) b2.f194b.getLifecycle()).f2578a.e(b2);
            }
        }

        @o(e.a.ON_START)
        public void onStart(h hVar) {
            this.f193a.e(hVar);
        }

        @o(e.a.ON_STOP)
        public void onStop(h hVar) {
            this.f193a.f(hVar);
        }
    }

    @AutoValue
    /* loaded from: classes.dex */
    public static abstract class a {
        public abstract c.b a();

        public abstract h b();
    }

    public void a(LifecycleCamera lifecycleCamera, b1 b1Var, Collection<a1> collection) {
        synchronized (this.f189a) {
            boolean z = true;
            d.d(!collection.isEmpty());
            h k = lifecycleCamera.k();
            for (a aVar : this.f191c.get(b(k))) {
                LifecycleCamera lifecycleCamera2 = this.f190b.get(aVar);
                Objects.requireNonNull(lifecycleCamera2);
                if (!lifecycleCamera2.equals(lifecycleCamera) && !lifecycleCamera2.l().isEmpty()) {
                    throw new IllegalArgumentException("Multiple LifecycleCameras with use cases are registered to the same LifecycleOwner.");
                }
            }
            try {
                synchronized (lifecycleCamera.f187c.f1604h) {
                }
                synchronized (lifecycleCamera.f185a) {
                    lifecycleCamera.f187c.c(collection);
                }
                if (((i) k.getLifecycle()).f2579b.compareTo(e.b.STARTED) < 0) {
                    z = false;
                }
                if (z) {
                    e(k);
                }
            } catch (c.a e2) {
                throw new IllegalArgumentException(e2.getMessage());
            }
        }
    }

    public final LifecycleCameraRepositoryObserver b(h hVar) {
        synchronized (this.f189a) {
            for (LifecycleCameraRepositoryObserver lifecycleCameraRepositoryObserver : this.f191c.keySet()) {
                if (hVar.equals(lifecycleCameraRepositoryObserver.f194b)) {
                    return lifecycleCameraRepositoryObserver;
                }
            }
            return null;
        }
    }

    public final boolean c(h hVar) {
        synchronized (this.f189a) {
            LifecycleCameraRepositoryObserver b2 = b(hVar);
            if (b2 == null) {
                return false;
            }
            for (a aVar : this.f191c.get(b2)) {
                LifecycleCamera lifecycleCamera = this.f190b.get(aVar);
                Objects.requireNonNull(lifecycleCamera);
                if (!lifecycleCamera.l().isEmpty()) {
                    return true;
                }
            }
            return false;
        }
    }

    public final void d(LifecycleCamera lifecycleCamera) {
        Set<a> hashSet;
        synchronized (this.f189a) {
            h k = lifecycleCamera.k();
            b bVar = new b(k, lifecycleCamera.f187c.f1601e);
            LifecycleCameraRepositoryObserver b2 = b(k);
            if (b2 != null) {
                hashSet = this.f191c.get(b2);
            } else {
                hashSet = new HashSet<>();
            }
            hashSet.add(bVar);
            this.f190b.put(bVar, lifecycleCamera);
            if (b2 == null) {
                LifecycleCameraRepositoryObserver lifecycleCameraRepositoryObserver = new LifecycleCameraRepositoryObserver(k, this);
                this.f191c.put(lifecycleCameraRepositoryObserver, hashSet);
                k.getLifecycle().a(lifecycleCameraRepositoryObserver);
            }
        }
    }

    public void e(h hVar) {
        synchronized (this.f189a) {
            if (c(hVar)) {
                if (this.f192d.isEmpty()) {
                    this.f192d.push(hVar);
                } else {
                    h peek = this.f192d.peek();
                    if (!hVar.equals(peek)) {
                        g(peek);
                        this.f192d.remove(hVar);
                        this.f192d.push(hVar);
                    }
                }
                h(hVar);
            }
        }
    }

    public void f(h hVar) {
        synchronized (this.f189a) {
            this.f192d.remove(hVar);
            g(hVar);
            if (!this.f192d.isEmpty()) {
                h(this.f192d.peek());
            }
        }
    }

    public final void g(h hVar) {
        synchronized (this.f189a) {
            for (a aVar : this.f191c.get(b(hVar))) {
                LifecycleCamera lifecycleCamera = this.f190b.get(aVar);
                Objects.requireNonNull(lifecycleCamera);
                lifecycleCamera.m();
            }
        }
    }

    public final void h(h hVar) {
        synchronized (this.f189a) {
            for (a aVar : this.f191c.get(b(hVar))) {
                LifecycleCamera lifecycleCamera = this.f190b.get(aVar);
                Objects.requireNonNull(lifecycleCamera);
                if (!lifecycleCamera.l().isEmpty()) {
                    lifecycleCamera.n();
                }
            }
        }
    }
}