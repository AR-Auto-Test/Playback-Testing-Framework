package b.d.b;

import android.util.Size;
import android.view.Surface;
import b.d.b.d1.k1.c.g;
import com.google.auto.value.AutoValue;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: SurfaceRequest.java */
/* loaded from: classes.dex */
public final class z0 {

    /* renamed from: a  reason: collision with root package name */
    public final Size f1700a;

    /* renamed from: b  reason: collision with root package name */
    public final b.d.b.d1.a0 f1701b;

    /* renamed from: c  reason: collision with root package name */
    public final ListenableFuture<Surface> f1702c;

    /* renamed from: d  reason: collision with root package name */
    public final b.g.a.b<Surface> f1703d;

    /* renamed from: e  reason: collision with root package name */
    public final ListenableFuture<Void> f1704e;

    /* renamed from: f  reason: collision with root package name */
    public final b.g.a.b<Void> f1705f;

    /* renamed from: g  reason: collision with root package name */
    public final b.d.b.d1.j0 f1706g;

    /* compiled from: SurfaceRequest.java */
    /* loaded from: classes.dex */
    public class a implements b.d.b.d1.k1.c.d<Void> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ b.g.a.b f1707a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ListenableFuture f1708b;

        public a(z0 z0Var, b.g.a.b bVar, ListenableFuture listenableFuture) {
            this.f1707a = bVar;
            this.f1708b = listenableFuture;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            if (th instanceof e) {
                b.j.b.d.k(this.f1708b.cancel(false), null);
            } else {
                b.j.b.d.k(this.f1707a.a(null), null);
            }
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r2) {
            b.j.b.d.k(this.f1707a.a(null), null);
        }
    }

    /* compiled from: SurfaceRequest.java */
    /* loaded from: classes.dex */
    public class b extends b.d.b.d1.j0 {
        public b() {
        }

        @Override // b.d.b.d1.j0
        public ListenableFuture<Surface> g() {
            return z0.this.f1702c;
        }
    }

    /* compiled from: SurfaceRequest.java */
    /* loaded from: classes.dex */
    public class c implements b.d.b.d1.k1.c.d<Surface> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ ListenableFuture f1709a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ b.g.a.b f1710b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ String f1711c;

        public c(z0 z0Var, ListenableFuture listenableFuture, b.g.a.b bVar, String str) {
            this.f1709a = listenableFuture;
            this.f1710b = bVar;
            this.f1711c = str;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            if (th instanceof CancellationException) {
                b.j.b.d.k(this.f1710b.c(new e(c.b.a.a.a.v(new StringBuilder(), this.f1711c, " cancelled."), th)), null);
            } else {
                this.f1710b.a(null);
            }
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Surface surface) {
            b.d.b.d1.k1.c.g.e(this.f1709a, this.f1710b);
        }
    }

    /* compiled from: SurfaceRequest.java */
    /* loaded from: classes.dex */
    public class d implements b.d.b.d1.k1.c.d<Void> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ b.j.i.a f1712a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ Surface f1713b;

        public d(z0 z0Var, b.j.i.a aVar, Surface surface) {
            this.f1712a = aVar;
            this.f1713b = surface;
        }

        @Override // b.d.b.d1.k1.c.d
        public void onFailure(Throwable th) {
            b.j.b.d.k(th instanceof e, "Camera surface session should only fail with request cancellation. Instead failed due to:\n" + th);
            this.f1712a.accept(new d0(1, this.f1713b));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // b.d.b.d1.k1.c.d
        public void onSuccess(Void r4) {
            this.f1712a.accept(new d0(0, this.f1713b));
        }
    }

    /* compiled from: SurfaceRequest.java */
    /* loaded from: classes.dex */
    public static final class e extends RuntimeException {
        public e(String str, Throwable th) {
            super(str, th);
        }
    }

    /* compiled from: SurfaceRequest.java */
    @AutoValue
    /* loaded from: classes.dex */
    public static abstract class f {
        public abstract int a();

        public abstract Surface b();
    }

    public z0(Size size, b.d.b.d1.a0 a0Var, boolean z) {
        this.f1700a = size;
        this.f1701b = a0Var;
        final String str = "SurfaceRequest[size: " + size + ", id: " + hashCode() + "]";
        final AtomicReference atomicReference = new AtomicReference(null);
        ListenableFuture d2 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.x
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar) {
                AtomicReference atomicReference2 = atomicReference;
                String str2 = str;
                atomicReference2.set(bVar);
                return str2 + "-cancellation";
            }
        });
        b.g.a.b<Void> bVar = (b.g.a.b) atomicReference.get();
        Objects.requireNonNull(bVar);
        this.f1705f = bVar;
        final AtomicReference atomicReference2 = new AtomicReference(null);
        ListenableFuture<Void> d3 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.y
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar2) {
                AtomicReference atomicReference3 = atomicReference2;
                String str2 = str;
                atomicReference3.set(bVar2);
                return str2 + "-status";
            }
        });
        this.f1704e = d3;
        d3.addListener(new g.d(d3, new a(this, bVar, d2)), b.b.a.f());
        b.g.a.b bVar2 = (b.g.a.b) atomicReference2.get();
        Objects.requireNonNull(bVar2);
        final AtomicReference atomicReference3 = new AtomicReference(null);
        ListenableFuture<Surface> d4 = b.e.a.d(new b.g.a.d() { // from class: b.d.b.w
            @Override // b.g.a.d
            public final Object a(b.g.a.b bVar3) {
                AtomicReference atomicReference4 = atomicReference3;
                String str2 = str;
                atomicReference4.set(bVar3);
                return str2 + "-Surface";
            }
        });
        this.f1702c = d4;
        b.g.a.b<Surface> bVar3 = (b.g.a.b) atomicReference3.get();
        Objects.requireNonNull(bVar3);
        this.f1703d = bVar3;
        b bVar4 = new b();
        this.f1706g = bVar4;
        ListenableFuture<Void> d5 = bVar4.d();
        d4.addListener(new g.d(d4, new c(this, d5, bVar2, str)), b.b.a.f());
        d5.addListener(new Runnable() { // from class: b.d.b.v
            @Override // java.lang.Runnable
            public final void run() {
                z0.this.f1702c.cancel(true);
            }
        }, b.b.a.f());
    }

    public void a(final Surface surface, Executor executor, final b.j.i.a<f> aVar) {
        if (!this.f1703d.a(surface) && !this.f1702c.isCancelled()) {
            b.j.b.d.k(this.f1702c.isDone(), null);
            try {
                this.f1702c.get();
                executor.execute(new Runnable() { // from class: b.d.b.t
                    @Override // java.lang.Runnable
                    public final void run() {
                        b.j.i.a.this.accept(new d0(3, surface));
                    }
                });
                return;
            } catch (InterruptedException | ExecutionException unused) {
                executor.execute(new Runnable() { // from class: b.d.b.u
                    @Override // java.lang.Runnable
                    public final void run() {
                        b.j.i.a.this.accept(new d0(4, surface));
                    }
                });
                return;
            }
        }
        ListenableFuture<Void> listenableFuture = this.f1704e;
        listenableFuture.addListener(new g.d(listenableFuture, new d(this, aVar, surface)), executor);
    }
}