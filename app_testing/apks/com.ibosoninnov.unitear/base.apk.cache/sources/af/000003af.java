package b.d.c;

import androidx.camera.lifecycle.LifecycleCameraRepository;
import b.d.b.e1.c;
import b.t.h;
import java.util.Objects;

/* compiled from: AutoValue_LifecycleCameraRepository_Key.java */
/* loaded from: classes.dex */
public final class b extends LifecycleCameraRepository.a {

    /* renamed from: a  reason: collision with root package name */
    public final h f1715a;

    /* renamed from: b  reason: collision with root package name */
    public final c.b f1716b;

    public b(h hVar, c.b bVar) {
        Objects.requireNonNull(hVar, "Null lifecycleOwner");
        this.f1715a = hVar;
        Objects.requireNonNull(bVar, "Null cameraId");
        this.f1716b = bVar;
    }

    @Override // androidx.camera.lifecycle.LifecycleCameraRepository.a
    public c.b a() {
        return this.f1716b;
    }

    @Override // androidx.camera.lifecycle.LifecycleCameraRepository.a
    public h b() {
        return this.f1715a;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof LifecycleCameraRepository.a) {
            LifecycleCameraRepository.a aVar = (LifecycleCameraRepository.a) obj;
            return this.f1715a.equals(aVar.b()) && this.f1716b.equals(aVar.a());
        }
        return false;
    }

    public int hashCode() {
        return ((this.f1715a.hashCode() ^ 1000003) * 1000003) ^ this.f1716b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Key{lifecycleOwner=");
        x.append(this.f1715a);
        x.append(", cameraId=");
        x.append(this.f1716b);
        x.append("}");
        return x.toString();
    }
}