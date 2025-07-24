package b.d.b;

import b.d.b.d1.g1;
import java.util.Objects;

/* compiled from: AutoValue_ImmutableImageInfo.java */
/* loaded from: classes.dex */
public final class c0 extends s0 {

    /* renamed from: a  reason: collision with root package name */
    public final g1 f1392a;

    /* renamed from: b  reason: collision with root package name */
    public final long f1393b;

    /* renamed from: c  reason: collision with root package name */
    public final int f1394c;

    public c0(g1 g1Var, long j, int i) {
        Objects.requireNonNull(g1Var, "Null tagBundle");
        this.f1392a = g1Var;
        this.f1393b = j;
        this.f1394c = i;
    }

    @Override // b.d.b.s0, b.d.b.q0
    public g1 a() {
        return this.f1392a;
    }

    @Override // b.d.b.s0
    public int b() {
        return this.f1394c;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof s0) {
            s0 s0Var = (s0) obj;
            return this.f1392a.equals(s0Var.a()) && this.f1393b == s0Var.getTimestamp() && this.f1394c == s0Var.b();
        }
        return false;
    }

    @Override // b.d.b.s0, b.d.b.q0
    public long getTimestamp() {
        return this.f1393b;
    }

    public int hashCode() {
        long j = this.f1393b;
        return ((((this.f1392a.hashCode() ^ 1000003) * 1000003) ^ ((int) (j ^ (j >>> 32)))) * 1000003) ^ this.f1394c;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ImmutableImageInfo{tagBundle=");
        x.append(this.f1392a);
        x.append(", timestamp=");
        x.append(this.f1393b);
        x.append(", rotationDegrees=");
        return c.b.a.a.a.s(x, this.f1394c, "}");
    }
}