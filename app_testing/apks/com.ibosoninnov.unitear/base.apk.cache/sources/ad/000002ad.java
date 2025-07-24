package b.d.a.e.y1.o;

import android.hardware.camera2.params.InputConfiguration;
import java.util.Objects;

/* compiled from: InputConfigurationCompat.java */
/* loaded from: classes.dex */
public final class a {

    /* renamed from: a  reason: collision with root package name */
    public final b f1328a;

    /* compiled from: InputConfigurationCompat.java */
    /* renamed from: b.d.a.e.y1.o.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static final class C0019a implements b {

        /* renamed from: a  reason: collision with root package name */
        public final InputConfiguration f1329a;

        public C0019a(Object obj) {
            this.f1329a = (InputConfiguration) obj;
        }

        @Override // b.d.a.e.y1.o.a.b
        public Object a() {
            return this.f1329a;
        }

        public boolean equals(Object obj) {
            if (obj instanceof b) {
                return Objects.equals(this.f1329a, ((b) obj).a());
            }
            return false;
        }

        public int hashCode() {
            return this.f1329a.hashCode();
        }

        public String toString() {
            return this.f1329a.toString();
        }
    }

    /* compiled from: InputConfigurationCompat.java */
    /* loaded from: classes.dex */
    public interface b {
        Object a();
    }

    public a(b bVar) {
        this.f1328a = bVar;
    }

    public boolean equals(Object obj) {
        if (obj instanceof a) {
            return this.f1328a.equals(((a) obj).f1328a);
        }
        return false;
    }

    public int hashCode() {
        return this.f1328a.hashCode();
    }

    public String toString() {
        return this.f1328a.toString();
    }
}