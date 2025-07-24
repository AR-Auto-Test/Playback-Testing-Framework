package g;

import java.util.concurrent.TimeUnit;

/* compiled from: ForwardingTimeout.java */
/* loaded from: classes2.dex */
public class k extends y {

    /* renamed from: e  reason: collision with root package name */
    public y f6185e;

    public k(y yVar) {
        if (yVar != null) {
            this.f6185e = yVar;
            return;
        }
        throw new IllegalArgumentException("delegate == null");
    }

    @Override // g.y
    public y a() {
        return this.f6185e.a();
    }

    @Override // g.y
    public y b() {
        return this.f6185e.b();
    }

    @Override // g.y
    public long c() {
        return this.f6185e.c();
    }

    @Override // g.y
    public y d(long j) {
        return this.f6185e.d(j);
    }

    @Override // g.y
    public boolean e() {
        return this.f6185e.e();
    }

    @Override // g.y
    public void f() {
        this.f6185e.f();
    }

    @Override // g.y
    public y g(long j, TimeUnit timeUnit) {
        return this.f6185e.g(j, timeUnit);
    }
}