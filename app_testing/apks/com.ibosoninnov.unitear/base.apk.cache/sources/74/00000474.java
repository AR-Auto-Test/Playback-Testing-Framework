package b.j.i;

import java.util.Objects;

/* compiled from: Pair.java */
/* loaded from: classes.dex */
public class c<F, S> {

    /* renamed from: a  reason: collision with root package name */
    public final F f2192a;

    /* renamed from: b  reason: collision with root package name */
    public final S f2193b;

    public c(F f2, S s) {
        this.f2192a = f2;
        this.f2193b = s;
    }

    public boolean equals(Object obj) {
        if (obj instanceof c) {
            c cVar = (c) obj;
            return Objects.equals(cVar.f2192a, this.f2192a) && Objects.equals(cVar.f2193b, this.f2193b);
        }
        return false;
    }

    public int hashCode() {
        F f2 = this.f2192a;
        int hashCode = f2 == null ? 0 : f2.hashCode();
        S s = this.f2193b;
        return hashCode ^ (s != null ? s.hashCode() : 0);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Pair{");
        x.append(this.f2192a);
        x.append(" ");
        return c.b.a.a.a.u(x, this.f2193b, "}");
    }
}