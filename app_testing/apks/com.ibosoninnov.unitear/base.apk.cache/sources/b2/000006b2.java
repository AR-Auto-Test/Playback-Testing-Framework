package c.a.a.z;

/* compiled from: MutablePair.java */
/* loaded from: classes.dex */
public class i<T> {

    /* renamed from: a  reason: collision with root package name */
    public T f3284a;

    /* renamed from: b  reason: collision with root package name */
    public T f3285b;

    public boolean equals(Object obj) {
        if (obj instanceof b.j.i.c) {
            b.j.i.c cVar = (b.j.i.c) obj;
            F f2 = cVar.f2192a;
            Object obj2 = this.f3284a;
            if (f2 == obj2 || (f2 != 0 && f2.equals(obj2))) {
                S s = cVar.f2193b;
                Object obj3 = this.f3285b;
                return s == obj3 || (s != 0 && s.equals(obj3));
            }
            return false;
        }
        return false;
    }

    public int hashCode() {
        T t = this.f3284a;
        int hashCode = t == null ? 0 : t.hashCode();
        T t2 = this.f3285b;
        return hashCode ^ (t2 != null ? t2.hashCode() : 0);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Pair{");
        x.append(String.valueOf(this.f3284a));
        x.append(" ");
        x.append(String.valueOf(this.f3285b));
        x.append("}");
        return x.toString();
    }
}