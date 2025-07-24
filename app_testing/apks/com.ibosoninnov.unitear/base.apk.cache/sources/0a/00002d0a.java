package f.g0.g;

import f.d0;
import f.t;

/* compiled from: RealResponseBody.java */
/* loaded from: classes2.dex */
public final class g extends d0 {

    /* renamed from: b  reason: collision with root package name */
    public final String f5833b;

    /* renamed from: c  reason: collision with root package name */
    public final long f5834c;

    /* renamed from: d  reason: collision with root package name */
    public final g.g f5835d;

    public g(String str, long j, g.g gVar) {
        this.f5833b = str;
        this.f5834c = j;
        this.f5835d = gVar;
    }

    @Override // f.d0
    public long C() {
        return this.f5834c;
    }

    @Override // f.d0
    public t D() {
        String str = this.f5833b;
        if (str != null) {
            return t.a(str);
        }
        return null;
    }

    @Override // f.d0
    public g.g E() {
        return this.f5835d;
    }
}