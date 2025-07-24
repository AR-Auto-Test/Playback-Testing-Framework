package b.d.b;

import android.view.Surface;
import b.d.b.z0;
import java.util.Objects;

/* compiled from: AutoValue_SurfaceRequest_Result.java */
/* loaded from: classes.dex */
public final class d0 extends z0.f {

    /* renamed from: a  reason: collision with root package name */
    public final int f1397a;

    /* renamed from: b  reason: collision with root package name */
    public final Surface f1398b;

    public d0(int i, Surface surface) {
        this.f1397a = i;
        Objects.requireNonNull(surface, "Null surface");
        this.f1398b = surface;
    }

    @Override // b.d.b.z0.f
    public int a() {
        return this.f1397a;
    }

    @Override // b.d.b.z0.f
    public Surface b() {
        return this.f1398b;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof z0.f) {
            z0.f fVar = (z0.f) obj;
            return this.f1397a == fVar.a() && this.f1398b.equals(fVar.b());
        }
        return false;
    }

    public int hashCode() {
        return ((this.f1397a ^ 1000003) * 1000003) ^ this.f1398b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Result{resultCode=");
        x.append(this.f1397a);
        x.append(", surface=");
        x.append(this.f1398b);
        x.append("}");
        return x.toString();
    }
}