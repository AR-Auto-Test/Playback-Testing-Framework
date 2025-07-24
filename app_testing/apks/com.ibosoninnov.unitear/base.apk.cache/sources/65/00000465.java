package b.j.g;

import android.os.Handler;
import java.util.concurrent.Callable;

/* compiled from: RequestExecutor.java */
/* loaded from: classes.dex */
public class o<T> implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public Callable<T> f2160b;

    /* renamed from: c  reason: collision with root package name */
    public b.j.i.a<T> f2161c;

    /* renamed from: d  reason: collision with root package name */
    public Handler f2162d;

    /* compiled from: RequestExecutor.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ b.j.i.a f2163b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ Object f2164c;

        public a(o oVar, b.j.i.a aVar, Object obj) {
            this.f2163b = aVar;
            this.f2164c = obj;
        }

        /* JADX DEBUG: Multi-variable search result rejected for r0v0, resolved type: b.j.i.a */
        /* JADX WARN: Multi-variable type inference failed */
        @Override // java.lang.Runnable
        public void run() {
            this.f2163b.accept(this.f2164c);
        }
    }

    public o(Handler handler, Callable<T> callable, b.j.i.a<T> aVar) {
        this.f2160b = callable;
        this.f2161c = aVar;
        this.f2162d = handler;
    }

    @Override // java.lang.Runnable
    public void run() {
        T t;
        try {
            t = this.f2160b.call();
        } catch (Exception unused) {
            t = null;
        }
        this.f2162d.post(new a(this, this.f2161c, t));
    }
}