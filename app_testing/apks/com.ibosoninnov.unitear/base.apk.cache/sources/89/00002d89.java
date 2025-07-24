package g;

import java.io.InterruptedIOException;
import java.util.concurrent.TimeUnit;

/* compiled from: Timeout.java */
/* loaded from: classes2.dex */
public class y {

    /* renamed from: a  reason: collision with root package name */
    public static final y f6220a = new a();

    /* renamed from: b  reason: collision with root package name */
    public boolean f6221b;

    /* renamed from: c  reason: collision with root package name */
    public long f6222c;

    /* renamed from: d  reason: collision with root package name */
    public long f6223d;

    /* compiled from: Timeout.java */
    /* loaded from: classes2.dex */
    public class a extends y {
        @Override // g.y
        public y d(long j) {
            return this;
        }

        @Override // g.y
        public void f() {
        }

        @Override // g.y
        public y g(long j, TimeUnit timeUnit) {
            return this;
        }
    }

    public y a() {
        this.f6221b = false;
        return this;
    }

    public y b() {
        this.f6223d = 0L;
        return this;
    }

    public long c() {
        if (this.f6221b) {
            return this.f6222c;
        }
        throw new IllegalStateException("No deadline");
    }

    public y d(long j) {
        this.f6221b = true;
        this.f6222c = j;
        return this;
    }

    public boolean e() {
        return this.f6221b;
    }

    public void f() {
        if (!Thread.interrupted()) {
            if (this.f6221b && this.f6222c - System.nanoTime() <= 0) {
                throw new InterruptedIOException("deadline reached");
            }
            return;
        }
        Thread.currentThread().interrupt();
        throw new InterruptedIOException("interrupted");
    }

    public y g(long j, TimeUnit timeUnit) {
        if (j >= 0) {
            if (timeUnit != null) {
                this.f6223d = timeUnit.toNanos(j);
                return this;
            }
            throw new IllegalArgumentException("unit == null");
        }
        throw new IllegalArgumentException(c.b.a.a.a.l("timeout < 0: ", j));
    }
}