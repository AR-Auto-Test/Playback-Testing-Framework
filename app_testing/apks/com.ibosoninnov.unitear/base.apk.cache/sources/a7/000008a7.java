package c.c.a.s;

/* compiled from: MultiClassKey.java */
/* loaded from: classes.dex */
public class i {

    /* renamed from: a  reason: collision with root package name */
    public Class<?> f4194a;

    /* renamed from: b  reason: collision with root package name */
    public Class<?> f4195b;

    /* renamed from: c  reason: collision with root package name */
    public Class<?> f4196c;

    public i() {
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || i.class != obj.getClass()) {
            return false;
        }
        i iVar = (i) obj;
        return this.f4194a.equals(iVar.f4194a) && this.f4195b.equals(iVar.f4195b) && j.b(this.f4196c, iVar.f4196c);
    }

    public int hashCode() {
        int hashCode = (this.f4195b.hashCode() + (this.f4194a.hashCode() * 31)) * 31;
        Class<?> cls = this.f4196c;
        return hashCode + (cls != null ? cls.hashCode() : 0);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("MultiClassKey{first=");
        x.append(this.f4194a);
        x.append(", second=");
        x.append(this.f4195b);
        x.append('}');
        return x.toString();
    }

    public i(Class<?> cls, Class<?> cls2, Class<?> cls3) {
        this.f4194a = cls;
        this.f4195b = cls2;
        this.f4196c = cls3;
    }
}