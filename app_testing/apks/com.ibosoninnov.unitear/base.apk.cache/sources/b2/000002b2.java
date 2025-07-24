package b.d.a.e.y1.o;

import android.hardware.camera2.params.OutputConfiguration;
import android.view.Surface;
import java.util.Objects;

/* compiled from: OutputConfigurationCompatApi24Impl.java */
/* loaded from: classes.dex */
public class c extends f {

    /* compiled from: OutputConfigurationCompatApi24Impl.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final OutputConfiguration f1331a;

        public a(OutputConfiguration outputConfiguration) {
            this.f1331a = outputConfiguration;
        }

        public boolean equals(Object obj) {
            return (obj instanceof a) && Objects.equals(this.f1331a, ((a) obj).f1331a) && Objects.equals(null, null);
        }

        public int hashCode() {
            int hashCode = this.f1331a.hashCode() ^ 31;
            int i = ((hashCode << 5) - hashCode) ^ 0;
            return ((i << 5) - i) ^ 0;
        }
    }

    public c(Surface surface) {
        super(new a(new OutputConfiguration(surface)));
    }

    @Override // b.d.a.e.y1.o.f, b.d.a.e.y1.o.b.a
    public Surface a() {
        return ((OutputConfiguration) c()).getSurface();
    }

    @Override // b.d.a.e.y1.o.f, b.d.a.e.y1.o.b.a
    public String b() {
        Objects.requireNonNull((a) this.f1333a);
        return null;
    }

    @Override // b.d.a.e.y1.o.f, b.d.a.e.y1.o.b.a
    public Object c() {
        b.j.b.d.d(this.f1333a instanceof a);
        return ((a) this.f1333a).f1331a;
    }

    public c(Object obj) {
        super(obj);
    }
}