package c.c.a.m.x.c;

import android.graphics.Bitmap;
import c.c.a.m.x.c.m;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;
import java.util.Queue;

/* compiled from: StreamBitmapDecoder.java */
/* loaded from: classes.dex */
public class z implements c.c.a.m.r<InputStream, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final m f4016a;

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f4017b;

    /* compiled from: StreamBitmapDecoder.java */
    /* loaded from: classes.dex */
    public static class a implements m.b {

        /* renamed from: a  reason: collision with root package name */
        public final w f4018a;

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.s.d f4019b;

        public a(w wVar, c.c.a.s.d dVar) {
            this.f4018a = wVar;
            this.f4019b = dVar;
        }

        @Override // c.c.a.m.x.c.m.b
        public void a(c.c.a.m.v.c0.d dVar, Bitmap bitmap) {
            IOException iOException = this.f4019b.f4183d;
            if (iOException != null) {
                if (bitmap != null) {
                    dVar.d(bitmap);
                }
                throw iOException;
            }
        }

        @Override // c.c.a.m.x.c.m.b
        public void b() {
            w wVar = this.f4018a;
            synchronized (wVar) {
                wVar.f4008d = wVar.f4006b.length;
            }
        }
    }

    public z(m mVar, c.c.a.m.v.c0.b bVar) {
        this.f4016a = mVar;
        this.f4017b = bVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(InputStream inputStream, c.c.a.m.p pVar) {
        Objects.requireNonNull(this.f4016a);
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(InputStream inputStream, int i, int i2, c.c.a.m.p pVar) {
        boolean z;
        w wVar;
        c.c.a.s.d poll;
        InputStream inputStream2 = inputStream;
        if (inputStream2 instanceof w) {
            wVar = (w) inputStream2;
            z = false;
        } else {
            z = true;
            wVar = new w(inputStream2, this.f4017b);
        }
        Queue<c.c.a.s.d> queue = c.c.a.s.d.f4181b;
        synchronized (queue) {
            poll = queue.poll();
        }
        if (poll == null) {
            poll = new c.c.a.s.d();
        }
        poll.f4182c = wVar;
        try {
            return this.f4016a.b(new c.c.a.s.h(poll), i, i2, pVar, new a(wVar, poll));
        } finally {
            poll.release();
            if (z) {
                wVar.release();
            }
        }
    }
}