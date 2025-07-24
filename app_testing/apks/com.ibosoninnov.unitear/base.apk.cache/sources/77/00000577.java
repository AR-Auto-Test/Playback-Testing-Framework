package b.v;

/* compiled from: NavArgument.java */
/* loaded from: classes.dex */
public final class d {

    /* renamed from: a  reason: collision with root package name */
    public final p f2611a;

    /* renamed from: b  reason: collision with root package name */
    public final boolean f2612b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f2613c;

    /* renamed from: d  reason: collision with root package name */
    public final Object f2614d;

    public d(p<?> pVar, boolean z, Object obj, boolean z2) {
        if (!pVar.l && z) {
            throw new IllegalArgumentException(pVar.b() + " does not allow nullable values");
        } else if (!z && z2 && obj == null) {
            StringBuilder x = c.b.a.a.a.x("Argument with type ");
            x.append(pVar.b());
            x.append(" has null value but is not nullable.");
            throw new IllegalArgumentException(x.toString());
        } else {
            this.f2611a = pVar;
            this.f2612b = z;
            this.f2614d = obj;
            this.f2613c = z2;
        }
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || d.class != obj.getClass()) {
            return false;
        }
        d dVar = (d) obj;
        if (this.f2612b == dVar.f2612b && this.f2613c == dVar.f2613c && this.f2611a.equals(dVar.f2611a)) {
            Object obj2 = this.f2614d;
            return obj2 != null ? obj2.equals(dVar.f2614d) : dVar.f2614d == null;
        }
        return false;
    }

    public int hashCode() {
        int hashCode = ((((this.f2611a.hashCode() * 31) + (this.f2612b ? 1 : 0)) * 31) + (this.f2613c ? 1 : 0)) * 31;
        Object obj = this.f2614d;
        return hashCode + (obj != null ? obj.hashCode() : 0);
    }
}