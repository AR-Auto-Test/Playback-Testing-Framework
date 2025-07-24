package c.c.a.m.x.g;

import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.SystemClock;
import c.c.a.m.t;
import c.c.a.m.v.k;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/* compiled from: GifFrameLoader.java */
/* loaded from: classes.dex */
public class g {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.l.a f4045a;

    /* renamed from: b  reason: collision with root package name */
    public final Handler f4046b;

    /* renamed from: c  reason: collision with root package name */
    public final List<b> f4047c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.i f4048d;

    /* renamed from: e  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f4049e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f4050f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f4051g;

    /* renamed from: h  reason: collision with root package name */
    public c.c.a.h<Bitmap> f4052h;
    public a i;
    public boolean j;
    public a k;
    public Bitmap l;
    public t<Bitmap> m;
    public a n;
    public int o;
    public int p;
    public int q;

    /* compiled from: GifFrameLoader.java */
    /* loaded from: classes.dex */
    public static class a extends c.c.a.q.j.c<Bitmap> {

        /* renamed from: e  reason: collision with root package name */
        public final Handler f4053e;

        /* renamed from: f  reason: collision with root package name */
        public final int f4054f;

        /* renamed from: g  reason: collision with root package name */
        public final long f4055g;

        /* renamed from: h  reason: collision with root package name */
        public Bitmap f4056h;

        public a(Handler handler, int i, long j) {
            this.f4053e = handler;
            this.f4054f = i;
            this.f4055g = j;
        }

        @Override // c.c.a.q.j.h
        public void b(Object obj, c.c.a.q.k.b bVar) {
            this.f4056h = (Bitmap) obj;
            this.f4053e.sendMessageAtTime(this.f4053e.obtainMessage(1, this), this.f4055g);
        }

        @Override // c.c.a.q.j.h
        public void g(Drawable drawable) {
            this.f4056h = null;
        }
    }

    /* compiled from: GifFrameLoader.java */
    /* loaded from: classes.dex */
    public interface b {
        void a();
    }

    /* compiled from: GifFrameLoader.java */
    /* loaded from: classes.dex */
    public class c implements Handler.Callback {
        public c() {
        }

        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            if (i == 1) {
                g.this.b((a) message.obj);
                return true;
            } else if (i == 2) {
                g.this.f4048d.i((a) message.obj);
                return false;
            } else {
                return false;
            }
        }
    }

    public g(c.c.a.b bVar, c.c.a.l.a aVar, int i, int i2, t<Bitmap> tVar, Bitmap bitmap) {
        c.c.a.m.v.c0.d dVar = bVar.f3412d;
        c.c.a.i e2 = c.c.a.b.e(bVar.f3414f.getBaseContext());
        c.c.a.i e3 = c.c.a.b.e(bVar.f3414f.getBaseContext());
        Objects.requireNonNull(e3);
        c.c.a.h<Bitmap> a2 = new c.c.a.h(e3.f3451c, e3, Bitmap.class, e3.f3452d).a(c.c.a.i.f3450b).a(new c.c.a.q.f().e(k.f3731a).u(true).o(true).i(i, i2));
        this.f4047c = new ArrayList();
        this.f4048d = e2;
        Handler handler = new Handler(Looper.getMainLooper(), new c());
        this.f4049e = dVar;
        this.f4046b = handler;
        this.f4052h = a2;
        this.f4045a = aVar;
        c(tVar, bitmap);
    }

    public final void a() {
        if (!this.f4050f || this.f4051g) {
            return;
        }
        a aVar = this.n;
        if (aVar != null) {
            this.n = null;
            b(aVar);
            return;
        }
        this.f4051g = true;
        long uptimeMillis = SystemClock.uptimeMillis() + this.f4045a.d();
        this.f4045a.b();
        this.k = new a(this.f4046b, this.f4045a.f(), uptimeMillis);
        c.c.a.h<Bitmap> D = this.f4052h.a(new c.c.a.q.f().n(new c.c.a.r.d(Double.valueOf(Math.random())))).D(this.f4045a);
        D.A(this.k, null, D, c.c.a.s.e.f4184a);
    }

    public void b(a aVar) {
        this.f4051g = false;
        if (this.j) {
            this.f4046b.obtainMessage(2, aVar).sendToTarget();
        } else if (!this.f4050f) {
            this.n = aVar;
        } else {
            if (aVar.f4056h != null) {
                Bitmap bitmap = this.l;
                if (bitmap != null) {
                    this.f4049e.d(bitmap);
                    this.l = null;
                }
                a aVar2 = this.i;
                this.i = aVar;
                int size = this.f4047c.size();
                while (true) {
                    size--;
                    if (size < 0) {
                        break;
                    }
                    this.f4047c.get(size).a();
                }
                if (aVar2 != null) {
                    this.f4046b.obtainMessage(2, aVar2).sendToTarget();
                }
            }
            a();
        }
    }

    public void c(t<Bitmap> tVar, Bitmap bitmap) {
        Objects.requireNonNull(tVar, "Argument must not be null");
        this.m = tVar;
        Objects.requireNonNull(bitmap, "Argument must not be null");
        this.l = bitmap;
        this.f4052h = this.f4052h.a(new c.c.a.q.f().q(tVar, true));
        this.o = c.c.a.s.j.d(bitmap);
        this.p = bitmap.getWidth();
        this.q = bitmap.getHeight();
    }
}