package c.c.a.m.v.d0;

import android.util.Log;
import c.c.a.k.a;
import c.c.a.m.m;
import c.c.a.m.v.d0.a;
import c.c.a.m.v.d0.c;
import java.io.File;
import java.io.IOException;

/* compiled from: DiskLruCacheWrapper.java */
/* loaded from: classes.dex */
public class e implements a {

    /* renamed from: b  reason: collision with root package name */
    public final File f3657b;

    /* renamed from: c  reason: collision with root package name */
    public final long f3658c;

    /* renamed from: e  reason: collision with root package name */
    public c.c.a.k.a f3660e;

    /* renamed from: d  reason: collision with root package name */
    public final c f3659d = new c();

    /* renamed from: a  reason: collision with root package name */
    public final k f3656a = new k();

    @Deprecated
    public e(File file, long j) {
        this.f3657b = file;
        this.f3658c = j;
    }

    @Override // c.c.a.m.v.d0.a
    public void a(m mVar, a.b bVar) {
        c.a aVar;
        boolean z;
        String a2 = this.f3656a.a(mVar);
        c cVar = this.f3659d;
        synchronized (cVar) {
            aVar = cVar.f3649a.get(a2);
            if (aVar == null) {
                c.b bVar2 = cVar.f3650b;
                synchronized (bVar2.f3653a) {
                    aVar = bVar2.f3653a.poll();
                }
                if (aVar == null) {
                    aVar = new c.a();
                }
                cVar.f3649a.put(a2, aVar);
            }
            aVar.f3652b++;
        }
        aVar.f3651a.lock();
        try {
            if (Log.isLoggable("DiskLruCacheWrapper", 2)) {
                Log.v("DiskLruCacheWrapper", "Put: Obtained: " + a2 + " for for Key: " + mVar);
            }
            try {
                c.c.a.k.a c2 = c();
                if (c2.H(a2) == null) {
                    a.c F = c2.F(a2);
                    if (F != null) {
                        try {
                            c.c.a.m.v.f fVar = (c.c.a.m.v.f) bVar;
                            if (fVar.f3689a.a(fVar.f3690b, F.b(0), fVar.f3691c)) {
                                c.c.a.k.a.B(c.c.a.k.a.this, F, true);
                                F.f3470c = true;
                            }
                            if (!z) {
                                try {
                                    F.a();
                                } catch (IOException unused) {
                                }
                            }
                        } finally {
                            if (!F.f3470c) {
                                try {
                                    F.a();
                                } catch (IOException unused2) {
                                }
                            }
                        }
                    } else {
                        throw new IllegalStateException("Had two simultaneous puts for: " + a2);
                    }
                }
            } catch (IOException e2) {
                if (Log.isLoggable("DiskLruCacheWrapper", 5)) {
                    Log.w("DiskLruCacheWrapper", "Unable to put to disk cache", e2);
                }
            }
        } finally {
            this.f3659d.a(a2);
        }
    }

    @Override // c.c.a.m.v.d0.a
    public File b(m mVar) {
        String a2 = this.f3656a.a(mVar);
        if (Log.isLoggable("DiskLruCacheWrapper", 2)) {
            Log.v("DiskLruCacheWrapper", "Get: Obtained: " + a2 + " for for Key: " + mVar);
        }
        try {
            a.e H = c().H(a2);
            if (H != null) {
                return H.f3480a[0];
            }
            return null;
        } catch (IOException e2) {
            if (Log.isLoggable("DiskLruCacheWrapper", 5)) {
                Log.w("DiskLruCacheWrapper", "Unable to get from disk cache", e2);
                return null;
            }
            return null;
        }
    }

    public final synchronized c.c.a.k.a c() {
        if (this.f3660e == null) {
            this.f3660e = c.c.a.k.a.J(this.f3657b, 1, 1, this.f3658c);
        }
        return this.f3660e;
    }
}