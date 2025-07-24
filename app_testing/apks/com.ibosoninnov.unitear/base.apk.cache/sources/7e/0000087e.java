package c.c.a.q;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import c.c.a.m.m;
import c.c.a.m.n;
import c.c.a.m.o;
import c.c.a.m.p;
import c.c.a.m.t;
import c.c.a.m.v.k;
import c.c.a.m.x.c.l;
import c.c.a.q.a;
import c.c.a.s.j;
import java.util.Map;
import java.util.Objects;
import org.opencv.calib3d.Calib3d;

/* compiled from: BaseRequestOptions.java */
/* loaded from: classes.dex */
public abstract class a<T extends a<T>> implements Cloneable {
    public boolean A;

    /* renamed from: b  reason: collision with root package name */
    public int f4126b;

    /* renamed from: f  reason: collision with root package name */
    public Drawable f4130f;

    /* renamed from: g  reason: collision with root package name */
    public int f4131g;

    /* renamed from: h  reason: collision with root package name */
    public Drawable f4132h;
    public int i;
    public m m;
    public boolean n;
    public boolean o;
    public Drawable p;
    public int q;
    public p r;
    public Map<Class<?>, t<?>> s;
    public Class<?> t;
    public boolean u;
    public Resources.Theme v;
    public boolean w;
    public boolean x;
    public boolean y;
    public boolean z;

    /* renamed from: c  reason: collision with root package name */
    public float f4127c = 1.0f;

    /* renamed from: d  reason: collision with root package name */
    public k f4128d = k.f3733c;

    /* renamed from: e  reason: collision with root package name */
    public c.c.a.f f4129e = c.c.a.f.NORMAL;
    public boolean j = true;
    public int k = -1;
    public int l = -1;

    public a() {
        c.c.a.r.c cVar = c.c.a.r.c.f4171b;
        this.m = c.c.a.r.c.f4171b;
        this.o = true;
        this.r = new p();
        this.s = new c.c.a.s.b();
        this.t = Object.class;
        this.z = true;
    }

    public static boolean g(int i, int i2) {
        return (i & i2) != 0;
    }

    public T a(a<?> aVar) {
        if (this.w) {
            return (T) clone().a(aVar);
        }
        if (g(aVar.f4126b, 2)) {
            this.f4127c = aVar.f4127c;
        }
        if (g(aVar.f4126b, Calib3d.CALIB_TILTED_MODEL)) {
            this.x = aVar.x;
        }
        if (g(aVar.f4126b, 1048576)) {
            this.A = aVar.A;
        }
        if (g(aVar.f4126b, 4)) {
            this.f4128d = aVar.f4128d;
        }
        if (g(aVar.f4126b, 8)) {
            this.f4129e = aVar.f4129e;
        }
        if (g(aVar.f4126b, 16)) {
            this.f4130f = aVar.f4130f;
            this.f4131g = 0;
            this.f4126b &= -33;
        }
        if (g(aVar.f4126b, 32)) {
            this.f4131g = aVar.f4131g;
            this.f4130f = null;
            this.f4126b &= -17;
        }
        if (g(aVar.f4126b, 64)) {
            this.f4132h = aVar.f4132h;
            this.i = 0;
            this.f4126b &= -129;
        }
        if (g(aVar.f4126b, 128)) {
            this.i = aVar.i;
            this.f4132h = null;
            this.f4126b &= -65;
        }
        if (g(aVar.f4126b, 256)) {
            this.j = aVar.j;
        }
        if (g(aVar.f4126b, 512)) {
            this.l = aVar.l;
            this.k = aVar.k;
        }
        if (g(aVar.f4126b, 1024)) {
            this.m = aVar.m;
        }
        if (g(aVar.f4126b, 4096)) {
            this.t = aVar.t;
        }
        if (g(aVar.f4126b, 8192)) {
            this.p = aVar.p;
            this.q = 0;
            this.f4126b &= -16385;
        }
        if (g(aVar.f4126b, Calib3d.CALIB_RATIONAL_MODEL)) {
            this.q = aVar.q;
            this.p = null;
            this.f4126b &= -8193;
        }
        if (g(aVar.f4126b, Calib3d.CALIB_THIN_PRISM_MODEL)) {
            this.v = aVar.v;
        }
        if (g(aVar.f4126b, 65536)) {
            this.o = aVar.o;
        }
        if (g(aVar.f4126b, 131072)) {
            this.n = aVar.n;
        }
        if (g(aVar.f4126b, 2048)) {
            this.s.putAll(aVar.s);
            this.z = aVar.z;
        }
        if (g(aVar.f4126b, 524288)) {
            this.y = aVar.y;
        }
        if (!this.o) {
            this.s.clear();
            int i = this.f4126b & (-2049);
            this.f4126b = i;
            this.n = false;
            this.f4126b = i & (-131073);
            this.z = true;
        }
        this.f4126b |= aVar.f4126b;
        this.r.d(aVar.r);
        l();
        return this;
    }

    public T b() {
        return r(l.f3971c, new c.c.a.m.x.c.i());
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // 
    /* renamed from: c */
    public T clone() {
        try {
            T t = (T) super.clone();
            p pVar = new p();
            t.r = pVar;
            pVar.d(this.r);
            c.c.a.s.b bVar = new c.c.a.s.b();
            t.s = bVar;
            bVar.putAll(this.s);
            t.u = false;
            t.w = false;
            return t;
        } catch (CloneNotSupportedException e2) {
            throw new RuntimeException(e2);
        }
    }

    public T d(Class<?> cls) {
        if (this.w) {
            return (T) clone().d(cls);
        }
        Objects.requireNonNull(cls, "Argument must not be null");
        this.t = cls;
        this.f4126b |= 4096;
        l();
        return this;
    }

    public T e(k kVar) {
        if (this.w) {
            return (T) clone().e(kVar);
        }
        Objects.requireNonNull(kVar, "Argument must not be null");
        this.f4128d = kVar;
        this.f4126b |= 4;
        l();
        return this;
    }

    public boolean equals(Object obj) {
        if (obj instanceof a) {
            a aVar = (a) obj;
            return Float.compare(aVar.f4127c, this.f4127c) == 0 && this.f4131g == aVar.f4131g && j.b(this.f4130f, aVar.f4130f) && this.i == aVar.i && j.b(this.f4132h, aVar.f4132h) && this.q == aVar.q && j.b(this.p, aVar.p) && this.j == aVar.j && this.k == aVar.k && this.l == aVar.l && this.n == aVar.n && this.o == aVar.o && this.x == aVar.x && this.y == aVar.y && this.f4128d.equals(aVar.f4128d) && this.f4129e == aVar.f4129e && this.r.equals(aVar.r) && this.s.equals(aVar.s) && this.t.equals(aVar.t) && j.b(this.m, aVar.m) && j.b(this.v, aVar.v);
        }
        return false;
    }

    public T f(int i) {
        if (this.w) {
            return (T) clone().f(i);
        }
        this.f4131g = i;
        int i2 = this.f4126b | 32;
        this.f4126b = i2;
        this.f4130f = null;
        this.f4126b = i2 & (-17);
        l();
        return this;
    }

    /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: c.c.a.m.o<c.c.a.m.x.c.l>, c.c.a.m.o<Y> */
    public final T h(l lVar, t<Bitmap> tVar) {
        if (this.w) {
            return (T) clone().h(lVar, tVar);
        }
        o oVar = l.f3974f;
        Objects.requireNonNull(lVar, "Argument must not be null");
        m(oVar, lVar);
        return q(tVar, false);
    }

    public int hashCode() {
        float f2 = this.f4127c;
        char[] cArr = j.f4197a;
        return j.g(this.v, j.g(this.m, j.g(this.t, j.g(this.s, j.g(this.r, j.g(this.f4129e, j.g(this.f4128d, (((((((((((((j.g(this.p, (j.g(this.f4132h, (j.g(this.f4130f, ((Float.floatToIntBits(f2) + 527) * 31) + this.f4131g) * 31) + this.i) * 31) + this.q) * 31) + (this.j ? 1 : 0)) * 31) + this.k) * 31) + this.l) * 31) + (this.n ? 1 : 0)) * 31) + (this.o ? 1 : 0)) * 31) + (this.x ? 1 : 0)) * 31) + (this.y ? 1 : 0))))))));
    }

    public T i(int i, int i2) {
        if (this.w) {
            return (T) clone().i(i, i2);
        }
        this.l = i;
        this.k = i2;
        this.f4126b |= 512;
        l();
        return this;
    }

    public T j(int i) {
        if (this.w) {
            return (T) clone().j(i);
        }
        this.i = i;
        int i2 = this.f4126b | 128;
        this.f4126b = i2;
        this.f4132h = null;
        this.f4126b = i2 & (-65);
        l();
        return this;
    }

    public T k(c.c.a.f fVar) {
        if (this.w) {
            return (T) clone().k(fVar);
        }
        Objects.requireNonNull(fVar, "Argument must not be null");
        this.f4129e = fVar;
        this.f4126b |= 8;
        l();
        return this;
    }

    public final T l() {
        if (this.u) {
            throw new IllegalStateException("You cannot modify locked T, consider clone()");
        }
        return this;
    }

    public <Y> T m(o<Y> oVar, Y y) {
        if (this.w) {
            return (T) clone().m(oVar, y);
        }
        Objects.requireNonNull(oVar, "Argument must not be null");
        Objects.requireNonNull(y, "Argument must not be null");
        this.r.f3544b.put(oVar, y);
        l();
        return this;
    }

    public T n(m mVar) {
        if (this.w) {
            return (T) clone().n(mVar);
        }
        Objects.requireNonNull(mVar, "Argument must not be null");
        this.m = mVar;
        this.f4126b |= 1024;
        l();
        return this;
    }

    public T o(boolean z) {
        if (this.w) {
            return (T) clone().o(true);
        }
        this.j = !z;
        this.f4126b |= 256;
        l();
        return this;
    }

    public T p(t<Bitmap> tVar) {
        return q(tVar, true);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.c.a.m.t<android.graphics.Bitmap> */
    /* JADX WARN: Multi-variable type inference failed */
    public T q(t<Bitmap> tVar, boolean z) {
        if (this.w) {
            return (T) clone().q(tVar, z);
        }
        c.c.a.m.x.c.o oVar = new c.c.a.m.x.c.o(tVar, z);
        s(Bitmap.class, tVar, z);
        s(Drawable.class, oVar, z);
        s(BitmapDrawable.class, oVar, z);
        s(c.c.a.m.x.g.c.class, new c.c.a.m.x.g.f(tVar), z);
        l();
        return this;
    }

    /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: c.c.a.m.o<c.c.a.m.x.c.l>, c.c.a.m.o<Y> */
    public final T r(l lVar, t<Bitmap> tVar) {
        if (this.w) {
            return (T) clone().r(lVar, tVar);
        }
        o oVar = l.f3974f;
        Objects.requireNonNull(lVar, "Argument must not be null");
        m(oVar, lVar);
        return q(tVar, true);
    }

    public <Y> T s(Class<Y> cls, t<Y> tVar, boolean z) {
        if (this.w) {
            return (T) clone().s(cls, tVar, z);
        }
        Objects.requireNonNull(cls, "Argument must not be null");
        Objects.requireNonNull(tVar, "Argument must not be null");
        this.s.put(cls, tVar);
        int i = this.f4126b | 2048;
        this.f4126b = i;
        this.o = true;
        int i2 = i | 65536;
        this.f4126b = i2;
        this.z = false;
        if (z) {
            this.f4126b = i2 | 131072;
            this.n = true;
        }
        l();
        return this;
    }

    public T t(t<Bitmap>... tVarArr) {
        if (tVarArr.length > 1) {
            return q(new n(tVarArr), true);
        }
        if (tVarArr.length == 1) {
            return p(tVarArr[0]);
        }
        l();
        return this;
    }

    public T u(boolean z) {
        if (this.w) {
            return (T) clone().u(z);
        }
        this.A = z;
        this.f4126b |= 1048576;
        l();
        return this;
    }
}