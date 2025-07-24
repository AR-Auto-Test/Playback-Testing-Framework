package c.c.a.m.v;

import java.nio.ByteBuffer;
import java.security.MessageDigest;

/* compiled from: ResourceCacheKey.java */
/* loaded from: classes.dex */
public final class y implements c.c.a.m.m {

    /* renamed from: b  reason: collision with root package name */
    public static final c.c.a.s.g<Class<?>, byte[]> f3812b = new c.c.a.s.g<>(50);

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f3813c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.m.m f3814d;

    /* renamed from: e  reason: collision with root package name */
    public final c.c.a.m.m f3815e;

    /* renamed from: f  reason: collision with root package name */
    public final int f3816f;

    /* renamed from: g  reason: collision with root package name */
    public final int f3817g;

    /* renamed from: h  reason: collision with root package name */
    public final Class<?> f3818h;
    public final c.c.a.m.p i;
    public final c.c.a.m.t<?> j;

    public y(c.c.a.m.v.c0.b bVar, c.c.a.m.m mVar, c.c.a.m.m mVar2, int i, int i2, c.c.a.m.t<?> tVar, Class<?> cls, c.c.a.m.p pVar) {
        this.f3813c = bVar;
        this.f3814d = mVar;
        this.f3815e = mVar2;
        this.f3816f = i;
        this.f3817g = i2;
        this.j = tVar;
        this.f3818h = cls;
        this.i = pVar;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        byte[] bArr = (byte[]) this.f3813c.c(8, byte[].class);
        ByteBuffer.wrap(bArr).putInt(this.f3816f).putInt(this.f3817g).array();
        this.f3815e.a(messageDigest);
        this.f3814d.a(messageDigest);
        messageDigest.update(bArr);
        c.c.a.m.t<?> tVar = this.j;
        if (tVar != null) {
            tVar.a(messageDigest);
        }
        this.i.a(messageDigest);
        c.c.a.s.g<Class<?>, byte[]> gVar = f3812b;
        byte[] a2 = gVar.a(this.f3818h);
        if (a2 == null) {
            a2 = this.f3818h.getName().getBytes(c.c.a.m.m.f3537a);
            gVar.d(this.f3818h, a2);
        }
        messageDigest.update(a2);
        this.f3813c.put(bArr);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof y) {
            y yVar = (y) obj;
            return this.f3817g == yVar.f3817g && this.f3816f == yVar.f3816f && c.c.a.s.j.b(this.j, yVar.j) && this.f3818h.equals(yVar.f3818h) && this.f3814d.equals(yVar.f3814d) && this.f3815e.equals(yVar.f3815e) && this.i.equals(yVar.i);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        int hashCode = ((((this.f3815e.hashCode() + (this.f3814d.hashCode() * 31)) * 31) + this.f3816f) * 31) + this.f3817g;
        c.c.a.m.t<?> tVar = this.j;
        if (tVar != null) {
            hashCode = (hashCode * 31) + tVar.hashCode();
        }
        int hashCode2 = this.f3818h.hashCode();
        return this.i.hashCode() + ((hashCode2 + (hashCode * 31)) * 31);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ResourceCacheKey{sourceKey=");
        x.append(this.f3814d);
        x.append(", signature=");
        x.append(this.f3815e);
        x.append(", width=");
        x.append(this.f3816f);
        x.append(", height=");
        x.append(this.f3817g);
        x.append(", decodedResourceClass=");
        x.append(this.f3818h);
        x.append(", transformation='");
        x.append(this.j);
        x.append('\'');
        x.append(", options=");
        x.append(this.i);
        x.append('}');
        return x.toString();
    }
}