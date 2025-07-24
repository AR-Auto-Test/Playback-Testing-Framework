package c.c.a.m;

import c.c.a.m.o;
import java.security.MessageDigest;

/* compiled from: Options.java */
/* loaded from: classes.dex */
public final class p implements m {

    /* renamed from: b  reason: collision with root package name */
    public final b.f.a<o<?>, Object> f3544b = new c.c.a.s.b();

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        int i = 0;
        while (true) {
            b.f.a<o<?>, Object> aVar = this.f3544b;
            if (i >= aVar.f1775h) {
                return;
            }
            o<?> h2 = aVar.h(i);
            Object l = this.f3544b.l(i);
            o.b<?> bVar = h2.f3541c;
            if (h2.f3543e == null) {
                h2.f3543e = h2.f3542d.getBytes(m.f3537a);
            }
            bVar.a(h2.f3543e, l, messageDigest);
            i++;
        }
    }

    public <T> T c(o<T> oVar) {
        if (this.f3544b.e(oVar) >= 0) {
            return (T) this.f3544b.getOrDefault(oVar, null);
        }
        return oVar.f3540b;
    }

    public void d(p pVar) {
        this.f3544b.i(pVar.f3544b);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof p) {
            return this.f3544b.equals(((p) obj).f3544b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f3544b.hashCode();
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Options{values=");
        x.append(this.f3544b);
        x.append('}');
        return x.toString();
    }
}