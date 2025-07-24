package b.o.a;

import android.os.SystemClock;
import android.view.Choreographer;
import b.f.h;
import java.util.ArrayList;

/* compiled from: AnimationHandler.java */
/* loaded from: classes.dex */
public class a {

    /* renamed from: a  reason: collision with root package name */
    public static final ThreadLocal<a> f2339a = new ThreadLocal<>();

    /* renamed from: e  reason: collision with root package name */
    public c f2343e;

    /* renamed from: b  reason: collision with root package name */
    public final h<b, Long> f2340b = new h<>();

    /* renamed from: c  reason: collision with root package name */
    public final ArrayList<b> f2341c = new ArrayList<>();

    /* renamed from: d  reason: collision with root package name */
    public final C0043a f2342d = new C0043a();

    /* renamed from: f  reason: collision with root package name */
    public long f2344f = 0;

    /* renamed from: g  reason: collision with root package name */
    public boolean f2345g = false;

    /* compiled from: AnimationHandler.java */
    /* renamed from: b.o.a.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0043a {
        public C0043a() {
        }

        /* JADX WARN: Removed duplicated region for block: B:17:0x0043  */
        /* JADX WARN: Removed duplicated region for block: B:37:0x0046 A[SYNTHETIC] */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void a() {
            boolean z;
            a.this.f2344f = SystemClock.uptimeMillis();
            a aVar = a.this;
            long j = aVar.f2344f;
            long uptimeMillis = SystemClock.uptimeMillis();
            for (int i = 0; i < aVar.f2341c.size(); i++) {
                b bVar = aVar.f2341c.get(i);
                if (bVar != null) {
                    Long orDefault = aVar.f2340b.getOrDefault(bVar, null);
                    if (orDefault != null) {
                        if (orDefault.longValue() < uptimeMillis) {
                            aVar.f2340b.remove(bVar);
                        } else {
                            z = false;
                            if (!z) {
                                bVar.a(j);
                            }
                        }
                    }
                    z = true;
                    if (!z) {
                    }
                }
            }
            if (aVar.f2345g) {
                int size = aVar.f2341c.size();
                while (true) {
                    size--;
                    if (size < 0) {
                        break;
                    } else if (aVar.f2341c.get(size) == null) {
                        aVar.f2341c.remove(size);
                    }
                }
                aVar.f2345g = false;
            }
            if (a.this.f2341c.size() > 0) {
                a aVar2 = a.this;
                if (aVar2.f2343e == null) {
                    aVar2.f2343e = new d(aVar2.f2342d);
                }
                d dVar = (d) aVar2.f2343e;
                dVar.f2348b.postFrameCallback(dVar.f2349c);
            }
        }
    }

    /* compiled from: AnimationHandler.java */
    /* loaded from: classes.dex */
    public interface b {
        boolean a(long j);
    }

    /* compiled from: AnimationHandler.java */
    /* loaded from: classes.dex */
    public static abstract class c {

        /* renamed from: a  reason: collision with root package name */
        public final C0043a f2347a;

        public c(C0043a c0043a) {
            this.f2347a = c0043a;
        }
    }

    /* compiled from: AnimationHandler.java */
    /* loaded from: classes.dex */
    public static class d extends c {

        /* renamed from: b  reason: collision with root package name */
        public final Choreographer f2348b;

        /* renamed from: c  reason: collision with root package name */
        public final Choreographer.FrameCallback f2349c;

        /* compiled from: AnimationHandler.java */
        /* renamed from: b.o.a.a$d$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class Choreographer$FrameCallbackC0044a implements Choreographer.FrameCallback {
            public Choreographer$FrameCallbackC0044a() {
            }

            @Override // android.view.Choreographer.FrameCallback
            public void doFrame(long j) {
                d.this.f2347a.a();
            }
        }

        public d(C0043a c0043a) {
            super(c0043a);
            this.f2348b = Choreographer.getInstance();
            this.f2349c = new Choreographer$FrameCallbackC0044a();
        }
    }

    public static a a() {
        ThreadLocal<a> threadLocal = f2339a;
        if (threadLocal.get() == null) {
            threadLocal.set(new a());
        }
        return threadLocal.get();
    }
}