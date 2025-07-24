package c.c.a.m.v;

import java.security.MessageDigest;
import java.util.Map;
import java.util.Objects;

/* compiled from: EngineKey.java */
/* loaded from: classes.dex */
public class o implements c.c.a.m.m {

    /* renamed from: b  reason: collision with root package name */
    public final Object f3773b;

    /* renamed from: c  reason: collision with root package name */
    public final int f3774c;

    /* renamed from: d  reason: collision with root package name */
    public final int f3775d;

    /* renamed from: e  reason: collision with root package name */
    public final Class<?> f3776e;

    /* renamed from: f  reason: collision with root package name */
    public final Class<?> f3777f;

    /* renamed from: g  reason: collision with root package name */
    public final c.c.a.m.m f3778g;

    /* renamed from: h  reason: collision with root package name */
    public final Map<Class<?>, c.c.a.m.t<?>> f3779h;
    public final c.c.a.m.p i;
    public int j;

    public o(Object obj, c.c.a.m.m mVar, int i, int i2, Map<Class<?>, c.c.a.m.t<?>> map, Class<?> cls, Class<?> cls2, c.c.a.m.p pVar) {
        Objects.requireNonNull(obj, "Argument must not be null");
        this.f3773b = obj;
        Objects.requireNonNull(mVar, "Signature must not be null");
        this.f3778g = mVar;
        this.f3774c = i;
        this.f3775d = i2;
        Objects.requireNonNull(map, "Argument must not be null");
        this.f3779h = map;
        Objects.requireNonNull(cls, "Resource class must not be null");
        this.f3776e = cls;
        Objects.requireNonNull(cls2, "Transcode class must not be null");
        this.f3777f = cls2;
        Objects.requireNonNull(pVar, "Argument must not be null");
        this.i = pVar;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        throw new UnsupportedOperationException();
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof o) {
            o oVar = (o) obj;
            return this.f3773b.equals(oVar.f3773b) && this.f3778g.equals(oVar.f3778g) && this.f3775d == oVar.f3775d && this.f3774c == oVar.f3774c && this.f3779h.equals(oVar.f3779h) && this.f3776e.equals(oVar.f3776e) && this.f3777f.equals(oVar.f3777f) && this.i.equals(oVar.i);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        if (this.j == 0) {
            int hashCode = this.f3773b.hashCode();
            this.j = hashCode;
            int hashCode2 = this.f3778g.hashCode() + (hashCode * 31);
            this.j = hashCode2;
            int i = (hashCode2 * 31) + this.f3774c;
            this.j = i;
            int i2 = (i * 31) + this.f3775d;
            this.j = i2;
            int hashCode3 = this.f3779h.hashCode() + (i2 * 31);
            this.j = hashCode3;
            int hashCode4 = this.f3776e.hashCode() + (hashCode3 * 31);
            this.j = hashCode4;
            int hashCode5 = this.f3777f.hashCode() + (hashCode4 * 31);
            this.j = hashCode5;
            this.j = this.i.hashCode() + (hashCode5 * 31);
        }
        return this.j;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("EngineKey{model=");
        x.append(this.f3773b);
        x.append(", width=");
        x.append(this.f3774c);
        x.append(", height=");
        x.append(this.f3775d);
        x.append(", resourceClass=");
        x.append(this.f3776e);
        x.append(", transcodeClass=");
        x.append(this.f3777f);
        x.append(", signature=");
        x.append(this.f3778g);
        x.append(", hashCode=");
        x.append(this.j);
        x.append(", transformations=");
        x.append(this.f3779h);
        x.append(", options=");
        x.append(this.i);
        x.append('}');
        return x.toString();
    }
}