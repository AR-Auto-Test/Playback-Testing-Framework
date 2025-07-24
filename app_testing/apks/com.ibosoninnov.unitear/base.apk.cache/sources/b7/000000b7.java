package androidx.lifecycle;

import androidx.databinding.ViewDataBinding;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.t.n;
import java.util.Map;

/* loaded from: classes.dex */
public abstract class LiveData<T> {

    /* renamed from: a  reason: collision with root package name */
    public static final Object f311a = new Object();

    /* renamed from: b  reason: collision with root package name */
    public final Object f312b;

    /* renamed from: c  reason: collision with root package name */
    public b.c.a.b.b<n<? super T>, LiveData<T>.b> f313c;

    /* renamed from: d  reason: collision with root package name */
    public int f314d;

    /* renamed from: e  reason: collision with root package name */
    public volatile Object f315e;

    /* renamed from: f  reason: collision with root package name */
    public volatile Object f316f;

    /* renamed from: g  reason: collision with root package name */
    public int f317g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f318h;
    public boolean i;
    public final Runnable j;

    /* loaded from: classes.dex */
    public class LifecycleBoundObserver extends LiveData<T>.b implements f {

        /* renamed from: e  reason: collision with root package name */
        public final h f319e;

        public LifecycleBoundObserver(h hVar, n<? super T> nVar) {
            super(nVar);
            this.f319e = hVar;
        }

        @Override // androidx.lifecycle.LiveData.b
        public void b() {
            ((i) this.f319e.getLifecycle()).f2578a.e(this);
        }

        @Override // b.t.f
        public void e(h hVar, e.a aVar) {
            if (((i) this.f319e.getLifecycle()).f2579b == e.b.DESTROYED) {
                LiveData.this.g(this.f322a);
            } else {
                a(h());
            }
        }

        @Override // androidx.lifecycle.LiveData.b
        public boolean g(h hVar) {
            return this.f319e == hVar;
        }

        @Override // androidx.lifecycle.LiveData.b
        public boolean h() {
            return ((i) this.f319e.getLifecycle()).f2579b.compareTo(e.b.STARTED) >= 0;
        }
    }

    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        /* JADX DEBUG: Multi-variable search result rejected for r0v2, resolved type: androidx.lifecycle.LiveData */
        /* JADX WARN: Multi-variable type inference failed */
        @Override // java.lang.Runnable
        public void run() {
            Object obj;
            synchronized (LiveData.this.f312b) {
                obj = LiveData.this.f316f;
                LiveData.this.f316f = LiveData.f311a;
            }
            LiveData.this.h(obj);
        }
    }

    /* loaded from: classes.dex */
    public abstract class b {

        /* renamed from: a  reason: collision with root package name */
        public final n<? super T> f322a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f323b;

        /* renamed from: c  reason: collision with root package name */
        public int f324c = -1;

        public b(n<? super T> nVar) {
            this.f322a = nVar;
        }

        public void a(boolean z) {
            if (z == this.f323b) {
                return;
            }
            this.f323b = z;
            LiveData liveData = LiveData.this;
            int i = liveData.f314d;
            boolean z2 = i == 0;
            liveData.f314d = i + (z ? 1 : -1);
            if (z2 && z) {
                liveData.e();
            }
            LiveData liveData2 = LiveData.this;
            if (liveData2.f314d == 0 && !this.f323b) {
                liveData2.f();
            }
            if (this.f323b) {
                LiveData.this.c(this);
            }
        }

        public void b() {
        }

        public boolean g(h hVar) {
            return false;
        }

        public abstract boolean h();
    }

    public LiveData(T t) {
        this.f312b = new Object();
        this.f313c = new b.c.a.b.b<>();
        this.f314d = 0;
        this.f316f = f311a;
        this.j = new a();
        this.f315e = t;
        this.f317g = 0;
    }

    public static void a(String str) {
        if (!b.c.a.a.a.c().f985b.a()) {
            throw new IllegalStateException(c.b.a.a.a.r("Cannot invoke ", str, " on a background thread"));
        }
    }

    public final void b(LiveData<T>.b bVar) {
        if (bVar.f323b) {
            if (!bVar.h()) {
                bVar.a(false);
                return;
            }
            int i = bVar.f324c;
            int i2 = this.f317g;
            if (i >= i2) {
                return;
            }
            bVar.f324c = i2;
            ViewDataBinding.e eVar = (ViewDataBinding.e) bVar.f322a;
            ViewDataBinding.g<LiveData<?>> gVar = eVar.f264a;
            ViewDataBinding viewDataBinding = (ViewDataBinding) gVar.get();
            if (viewDataBinding == null) {
                gVar.a();
            }
            if (viewDataBinding != null) {
                ViewDataBinding.g<LiveData<?>> gVar2 = eVar.f264a;
                int i3 = gVar2.f267b;
                LiveData<?> liveData = gVar2.f268c;
                int i4 = ViewDataBinding.f257c;
                if (viewDataBinding.i(i3, liveData, 0)) {
                    viewDataBinding.l();
                }
            }
        }
    }

    public void c(LiveData<T>.b bVar) {
        if (this.f318h) {
            this.i = true;
            return;
        }
        this.f318h = true;
        do {
            this.i = false;
            if (bVar != null) {
                b(bVar);
                bVar = null;
            } else {
                b.c.a.b.b<n<? super T>, LiveData<T>.b>.d b2 = this.f313c.b();
                while (b2.hasNext()) {
                    b((b) ((Map.Entry) b2.next()).getValue());
                    if (this.i) {
                        break;
                    }
                }
            }
        } while (this.i);
        this.f318h = false;
    }

    public void d(h hVar, n<? super T> nVar) {
        a("observe");
        if (((i) hVar.getLifecycle()).f2579b == e.b.DESTROYED) {
            return;
        }
        LifecycleBoundObserver lifecycleBoundObserver = new LifecycleBoundObserver(hVar, nVar);
        LiveData<T>.b d2 = this.f313c.d(nVar, lifecycleBoundObserver);
        if (d2 != null && !d2.g(hVar)) {
            throw new IllegalArgumentException("Cannot add the same observer with different lifecycles");
        }
        if (d2 != null) {
            return;
        }
        hVar.getLifecycle().a(lifecycleBoundObserver);
    }

    public void e() {
    }

    public void f() {
    }

    public void g(n<? super T> nVar) {
        a("removeObserver");
        LiveData<T>.b e2 = this.f313c.e(nVar);
        if (e2 == null) {
            return;
        }
        e2.b();
        e2.a(false);
    }

    public void h(T t) {
        a("setValue");
        this.f317g++;
        this.f315e = t;
        c(null);
    }

    public LiveData() {
        this.f312b = new Object();
        this.f313c = new b.c.a.b.b<>();
        this.f314d = 0;
        Object obj = f311a;
        this.f316f = obj;
        this.j = new a();
        this.f315e = obj;
        this.f317g = -1;
    }
}