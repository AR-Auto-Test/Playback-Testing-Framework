package b.d.a.e.y1.o;

import android.os.Build;
import android.view.Surface;

/* compiled from: OutputConfigurationCompat.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public final a f1330a;

    /* compiled from: OutputConfigurationCompat.java */
    /* loaded from: classes.dex */
    public interface a {
        Surface a();

        String b();

        Object c();
    }

    public b(Surface surface) {
        int i = Build.VERSION.SDK_INT;
        if (i >= 28) {
            this.f1330a = new e(surface);
        } else if (i >= 26) {
            this.f1330a = new d(surface);
        } else {
            this.f1330a = new c(surface);
        }
    }

    public boolean equals(Object obj) {
        if (obj instanceof b) {
            return this.f1330a.equals(((b) obj).f1330a);
        }
        return false;
    }

    public int hashCode() {
        return this.f1330a.hashCode();
    }

    public b(a aVar) {
        this.f1330a = aVar;
    }
}