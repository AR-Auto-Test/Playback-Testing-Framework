package b.d.b.d1;

import android.os.Handler;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: AutoValue_CameraThreadConfig.java */
/* loaded from: classes.dex */
public final class m extends d0 {

    /* renamed from: a  reason: collision with root package name */
    public final Executor f1568a;

    /* renamed from: b  reason: collision with root package name */
    public final Handler f1569b;

    public m(Executor executor, Handler handler) {
        Objects.requireNonNull(executor, "Null cameraExecutor");
        this.f1568a = executor;
        Objects.requireNonNull(handler, "Null schedulerHandler");
        this.f1569b = handler;
    }

    @Override // b.d.b.d1.d0
    public Executor a() {
        return this.f1568a;
    }

    @Override // b.d.b.d1.d0
    public Handler b() {
        return this.f1569b;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof d0) {
            d0 d0Var = (d0) obj;
            return this.f1568a.equals(d0Var.a()) && this.f1569b.equals(d0Var.b());
        }
        return false;
    }

    public int hashCode() {
        return ((this.f1568a.hashCode() ^ 1000003) * 1000003) ^ this.f1569b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("CameraThreadConfig{cameraExecutor=");
        x.append(this.f1568a);
        x.append(", schedulerHandler=");
        x.append(this.f1569b);
        x.append("}");
        return x.toString();
    }
}