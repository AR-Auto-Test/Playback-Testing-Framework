package b.d.a.e.y1.o;

import android.hardware.camera2.params.OutputConfiguration;
import android.view.Surface;
import java.util.Objects;

/* compiled from: OutputConfigurationCompatApi26Impl.java */
/* loaded from: classes.dex */
public class d extends c {

    /* compiled from: OutputConfigurationCompatApi26Impl.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final OutputConfiguration f1332a;

        public a(OutputConfiguration outputConfiguration) {
            this.f1332a = outputConfiguration;
        }

        public boolean equals(Object obj) {
            return (obj instanceof a) && Objects.equals(this.f1332a, ((a) obj).f1332a) && Objects.equals(null, null);
        }

        public int hashCode() {
            int hashCode = this.f1332a.hashCode() ^ 31;
            return ((hashCode << 5) - hashCode) ^ 0;
        }
    }

    public d(Surface surface) {
        super(new a(new OutputConfiguration(surface)));
    }

    @Override // b.d.a.e.y1.o.c, b.d.a.e.y1.o.f, b.d.a.e.y1.o.b.a
    public String b() {
        Objects.requireNonNull((a) this.f1333a);
        return null;
    }

    @Override // b.d.a.e.y1.o.c, b.d.a.e.y1.o.f, b.d.a.e.y1.o.b.a
    public Object c() {
        b.j.b.d.d(this.f1333a instanceof a);
        return ((a) this.f1333a).f1332a;
    }

    public d(Object obj) {
        super(obj);
    }
}