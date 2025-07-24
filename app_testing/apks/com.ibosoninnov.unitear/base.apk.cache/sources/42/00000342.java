package b.d.b.d1;

import b.d.b.d1.i0;
import java.util.Objects;

/* compiled from: AutoValue_Config_Option.java */
/* loaded from: classes.dex */
public final class n<T> extends i0.a<T> {

    /* renamed from: a  reason: collision with root package name */
    public final String f1571a;

    /* renamed from: b  reason: collision with root package name */
    public final Class<T> f1572b;

    /* renamed from: c  reason: collision with root package name */
    public final Object f1573c;

    public n(String str, Class<T> cls, Object obj) {
        Objects.requireNonNull(str, "Null id");
        this.f1571a = str;
        Objects.requireNonNull(cls, "Null valueClass");
        this.f1572b = cls;
        this.f1573c = obj;
    }

    @Override // b.d.b.d1.i0.a
    public String a() {
        return this.f1571a;
    }

    @Override // b.d.b.d1.i0.a
    public Object b() {
        return this.f1573c;
    }

    @Override // b.d.b.d1.i0.a
    public Class<T> c() {
        return this.f1572b;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof i0.a) {
            i0.a aVar = (i0.a) obj;
            if (this.f1571a.equals(aVar.a()) && this.f1572b.equals(aVar.c())) {
                Object obj2 = this.f1573c;
                if (obj2 == null) {
                    if (aVar.b() == null) {
                        return true;
                    }
                } else if (obj2.equals(aVar.b())) {
                    return true;
                }
            }
            return false;
        }
        return false;
    }

    public int hashCode() {
        int hashCode = (((this.f1571a.hashCode() ^ 1000003) * 1000003) ^ this.f1572b.hashCode()) * 1000003;
        Object obj = this.f1573c;
        return hashCode ^ (obj == null ? 0 : obj.hashCode());
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Option{id=");
        x.append(this.f1571a);
        x.append(", valueClass=");
        x.append(this.f1572b);
        x.append(", token=");
        return c.b.a.a.a.u(x, this.f1573c, "}");
    }
}