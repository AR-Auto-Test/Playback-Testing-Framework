package b.j.j;

import java.util.Objects;

/* compiled from: DisplayCutoutCompat.java */
/* loaded from: classes.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public final Object f2199a;

    public c(Object obj) {
        this.f2199a = obj;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || c.class != obj.getClass()) {
            return false;
        }
        return Objects.equals(this.f2199a, ((c) obj).f2199a);
    }

    public int hashCode() {
        Object obj = this.f2199a;
        if (obj == null) {
            return 0;
        }
        return obj.hashCode();
    }

    public String toString() {
        return c.b.a.a.a.u(c.b.a.a.a.x("DisplayCutoutCompat{"), this.f2199a, "}");
    }
}