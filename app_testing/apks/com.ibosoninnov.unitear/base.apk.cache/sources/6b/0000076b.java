package c.c.a.m.v;

import java.security.MessageDigest;

/* compiled from: DataCacheKey.java */
/* loaded from: classes.dex */
public final class e implements c.c.a.m.m {

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.m f3678b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.m f3679c;

    public e(c.c.a.m.m mVar, c.c.a.m.m mVar2) {
        this.f3678b = mVar;
        this.f3679c = mVar2;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        this.f3678b.a(messageDigest);
        this.f3679c.a(messageDigest);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof e) {
            e eVar = (e) obj;
            return this.f3678b.equals(eVar.f3678b) && this.f3679c.equals(eVar.f3679c);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f3679c.hashCode() + (this.f3678b.hashCode() * 31);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("DataCacheKey{sourceKey=");
        x.append(this.f3678b);
        x.append(", signature=");
        x.append(this.f3679c);
        x.append('}');
        return x.toString();
    }
}