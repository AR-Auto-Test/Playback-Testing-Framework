package c.c.a.r;

import c.c.a.m.m;
import java.security.MessageDigest;
import java.util.Objects;

/* compiled from: ObjectKey.java */
/* loaded from: classes.dex */
public final class d implements m {

    /* renamed from: b  reason: collision with root package name */
    public final Object f4172b;

    public d(Object obj) {
        Objects.requireNonNull(obj, "Argument must not be null");
        this.f4172b = obj;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        messageDigest.update(this.f4172b.toString().getBytes(m.f3537a));
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof d) {
            return this.f4172b.equals(((d) obj).f4172b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f4172b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ObjectKey{object=");
        x.append(this.f4172b);
        x.append('}');
        return x.toString();
    }
}