package b.d.b.d1;

import b.d.b.d1.e1;
import java.util.Objects;

/* compiled from: AutoValue_SurfaceConfig.java */
/* loaded from: classes.dex */
public final class o extends e1 {

    /* renamed from: a  reason: collision with root package name */
    public final e1.b f1580a;

    /* renamed from: b  reason: collision with root package name */
    public final e1.a f1581b;

    public o(e1.b bVar, e1.a aVar) {
        Objects.requireNonNull(bVar, "Null configType");
        this.f1580a = bVar;
        Objects.requireNonNull(aVar, "Null configSize");
        this.f1581b = aVar;
    }

    @Override // b.d.b.d1.e1
    public e1.a a() {
        return this.f1581b;
    }

    @Override // b.d.b.d1.e1
    public e1.b b() {
        return this.f1580a;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof e1) {
            e1 e1Var = (e1) obj;
            return this.f1580a.equals(e1Var.b()) && this.f1581b.equals(e1Var.a());
        }
        return false;
    }

    public int hashCode() {
        return ((this.f1580a.hashCode() ^ 1000003) * 1000003) ^ this.f1581b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("SurfaceConfig{configType=");
        x.append(this.f1580a);
        x.append(", configSize=");
        x.append(this.f1581b);
        x.append("}");
        return x.toString();
    }
}