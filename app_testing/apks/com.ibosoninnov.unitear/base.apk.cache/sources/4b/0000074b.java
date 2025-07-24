package c.c.a.m.v.c0;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/* compiled from: LruBitmapPool.java */
/* loaded from: classes.dex */
public class j implements d {

    /* renamed from: a  reason: collision with root package name */
    public static final Bitmap.Config f3622a = Bitmap.Config.ARGB_8888;

    /* renamed from: b  reason: collision with root package name */
    public final k f3623b;

    /* renamed from: c  reason: collision with root package name */
    public final Set<Bitmap.Config> f3624c;

    /* renamed from: d  reason: collision with root package name */
    public final a f3625d;

    /* renamed from: e  reason: collision with root package name */
    public long f3626e;

    /* renamed from: f  reason: collision with root package name */
    public long f3627f;

    /* renamed from: g  reason: collision with root package name */
    public int f3628g;

    /* renamed from: h  reason: collision with root package name */
    public int f3629h;
    public int i;
    public int j;

    /* compiled from: LruBitmapPool.java */
    /* loaded from: classes.dex */
    public interface a {
    }

    /* compiled from: LruBitmapPool.java */
    /* loaded from: classes.dex */
    public static final class b implements a {
    }

    public j(long j) {
        m mVar = new m();
        HashSet hashSet = new HashSet(Arrays.asList(Bitmap.Config.values()));
        int i = Build.VERSION.SDK_INT;
        hashSet.add(null);
        if (i >= 26) {
            hashSet.remove(Bitmap.Config.HARDWARE);
        }
        Set<Bitmap.Config> unmodifiableSet = Collections.unmodifiableSet(hashSet);
        this.f3626e = j;
        this.f3623b = mVar;
        this.f3624c = unmodifiableSet;
        this.f3625d = new b();
    }

    @Override // c.c.a.m.v.c0.d
    @SuppressLint({"InlinedApi"})
    public void a(int i) {
        if (Log.isLoggable("LruBitmapPool", 3)) {
            c.b.a.a.a.L("trimMemory, level=", i, "LruBitmapPool");
        }
        if (i >= 40 || i >= 20) {
            if (Log.isLoggable("LruBitmapPool", 3)) {
                Log.d("LruBitmapPool", "clearMemory");
            }
            i(0L);
        } else if (i >= 20 || i == 15) {
            i(this.f3626e / 2);
        }
    }

    @Override // c.c.a.m.v.c0.d
    public void b() {
        if (Log.isLoggable("LruBitmapPool", 3)) {
            Log.d("LruBitmapPool", "clearMemory");
        }
        i(0L);
    }

    @Override // c.c.a.m.v.c0.d
    public Bitmap c(int i, int i2, Bitmap.Config config) {
        Bitmap h2 = h(i, i2, config);
        if (h2 == null) {
            if (config == null) {
                config = f3622a;
            }
            return Bitmap.createBitmap(i, i2, config);
        }
        return h2;
    }

    @Override // c.c.a.m.v.c0.d
    public synchronized void d(Bitmap bitmap) {
        try {
            if (bitmap != null) {
                if (!bitmap.isRecycled()) {
                    if (bitmap.isMutable()) {
                        Objects.requireNonNull((m) this.f3623b);
                        if (c.c.a.s.j.d(bitmap) <= this.f3626e && this.f3624c.contains(bitmap.getConfig())) {
                            Objects.requireNonNull((m) this.f3623b);
                            int d2 = c.c.a.s.j.d(bitmap);
                            ((m) this.f3623b).f(bitmap);
                            Objects.requireNonNull((b) this.f3625d);
                            this.i++;
                            this.f3627f += d2;
                            if (Log.isLoggable("LruBitmapPool", 2)) {
                                Log.v("LruBitmapPool", "Put bitmap in pool=" + ((m) this.f3623b).e(bitmap));
                            }
                            f();
                            i(this.f3626e);
                            return;
                        }
                    }
                    if (Log.isLoggable("LruBitmapPool", 2)) {
                        Log.v("LruBitmapPool", "Reject bitmap from pool, bitmap: " + ((m) this.f3623b).e(bitmap) + ", is mutable: " + bitmap.isMutable() + ", is allowed config: " + this.f3624c.contains(bitmap.getConfig()));
                    }
                    bitmap.recycle();
                    return;
                }
                throw new IllegalStateException("Cannot pool recycled bitmap");
            }
            throw new NullPointerException("Bitmap must not be null");
        } catch (Throwable th) {
            throw th;
        }
    }

    @Override // c.c.a.m.v.c0.d
    public Bitmap e(int i, int i2, Bitmap.Config config) {
        Bitmap h2 = h(i, i2, config);
        if (h2 != null) {
            h2.eraseColor(0);
            return h2;
        }
        if (config == null) {
            config = f3622a;
        }
        return Bitmap.createBitmap(i, i2, config);
    }

    public final void f() {
        if (Log.isLoggable("LruBitmapPool", 2)) {
            g();
        }
    }

    public final void g() {
        StringBuilder x = c.b.a.a.a.x("Hits=");
        x.append(this.f3628g);
        x.append(", misses=");
        x.append(this.f3629h);
        x.append(", puts=");
        x.append(this.i);
        x.append(", evictions=");
        x.append(this.j);
        x.append(", currentSize=");
        x.append(this.f3627f);
        x.append(", maxSize=");
        x.append(this.f3626e);
        x.append("\nStrategy=");
        x.append(this.f3623b);
        Log.v("LruBitmapPool", x.toString());
    }

    public final synchronized Bitmap h(int i, int i2, Bitmap.Config config) {
        Bitmap b2;
        if (Build.VERSION.SDK_INT >= 26 && config == Bitmap.Config.HARDWARE) {
            throw new IllegalArgumentException("Cannot create a mutable Bitmap with config: " + config + ". Consider setting Downsampler#ALLOW_HARDWARE_CONFIG to false in your RequestOptions and/or in GlideBuilder.setDefaultRequestOptions");
        }
        b2 = ((m) this.f3623b).b(i, i2, config != null ? config : f3622a);
        if (b2 == null) {
            if (Log.isLoggable("LruBitmapPool", 3)) {
                StringBuilder sb = new StringBuilder();
                sb.append("Missing bitmap=");
                Objects.requireNonNull((m) this.f3623b);
                sb.append(m.c(c.c.a.s.j.c(i, i2, config), config));
                Log.d("LruBitmapPool", sb.toString());
            }
            this.f3629h++;
        } else {
            this.f3628g++;
            long j = this.f3627f;
            Objects.requireNonNull((m) this.f3623b);
            this.f3627f = j - c.c.a.s.j.d(b2);
            Objects.requireNonNull((b) this.f3625d);
            b2.setHasAlpha(true);
            b2.setPremultiplied(true);
        }
        if (Log.isLoggable("LruBitmapPool", 2)) {
            StringBuilder sb2 = new StringBuilder();
            sb2.append("Get bitmap=");
            Objects.requireNonNull((m) this.f3623b);
            sb2.append(m.c(c.c.a.s.j.c(i, i2, config), config));
            Log.v("LruBitmapPool", sb2.toString());
        }
        f();
        return b2;
    }

    public final synchronized void i(long j) {
        while (this.f3627f > j) {
            m mVar = (m) this.f3623b;
            Bitmap c2 = mVar.f3636g.c();
            if (c2 != null) {
                mVar.a(Integer.valueOf(c.c.a.s.j.d(c2)), c2);
            }
            if (c2 == null) {
                if (Log.isLoggable("LruBitmapPool", 5)) {
                    Log.w("LruBitmapPool", "Size mismatch, resetting");
                    g();
                }
                this.f3627f = 0L;
                return;
            }
            Objects.requireNonNull((b) this.f3625d);
            long j2 = this.f3627f;
            Objects.requireNonNull((m) this.f3623b);
            this.f3627f = j2 - c.c.a.s.j.d(c2);
            this.j++;
            if (Log.isLoggable("LruBitmapPool", 3)) {
                Log.d("LruBitmapPool", "Evicting bitmap=" + ((m) this.f3623b).e(c2));
            }
            f();
            c2.recycle();
        }
    }
}