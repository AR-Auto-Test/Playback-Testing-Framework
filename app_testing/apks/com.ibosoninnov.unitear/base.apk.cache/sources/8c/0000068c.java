package c.a.a.x.b;

import android.graphics.Path;
import android.graphics.PointF;
import b.d.b.m0;
import c.a.a.x.c.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import java.util.List;

/* compiled from: PolystarContent.java */
/* loaded from: classes.dex */
public class n implements m, a.b, k {

    /* renamed from: b  reason: collision with root package name */
    public final String f3188b;

    /* renamed from: c  reason: collision with root package name */
    public final c.a.a.j f3189c;

    /* renamed from: d  reason: collision with root package name */
    public final int f3190d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f3191e;

    /* renamed from: f  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3192f;

    /* renamed from: g  reason: collision with root package name */
    public final c.a.a.x.c.a<?, PointF> f3193g;

    /* renamed from: h  reason: collision with root package name */
    public final c.a.a.x.c.a<?, Float> f3194h;
    public final c.a.a.x.c.a<?, Float> i;
    public final c.a.a.x.c.a<?, Float> j;
    public final c.a.a.x.c.a<?, Float> k;
    public final c.a.a.x.c.a<?, Float> l;
    public boolean n;

    /* renamed from: a  reason: collision with root package name */
    public final Path f3187a = new Path();
    public b m = new b();

    public n(c.a.a.j jVar, c.a.a.z.l.b bVar, c.a.a.z.k.h hVar) {
        this.f3189c = jVar;
        this.f3188b = hVar.f3338a;
        int i = hVar.f3339b;
        this.f3190d = i;
        this.f3191e = hVar.j;
        c.a.a.x.c.a<Float, Float> a2 = hVar.f3340c.a();
        this.f3192f = a2;
        c.a.a.x.c.a<PointF, PointF> a3 = hVar.f3341d.a();
        this.f3193g = a3;
        c.a.a.x.c.a<Float, Float> a4 = hVar.f3342e.a();
        this.f3194h = a4;
        c.a.a.x.c.a<Float, Float> a5 = hVar.f3344g.a();
        this.j = a5;
        c.a.a.x.c.a<Float, Float> a6 = hVar.i.a();
        this.l = a6;
        if (i == 1) {
            this.i = hVar.f3343f.a();
            this.k = hVar.f3345h.a();
        } else {
            this.i = null;
            this.k = null;
        }
        bVar.e(a2);
        bVar.e(a3);
        bVar.e(a4);
        bVar.e(a5);
        bVar.e(a6);
        if (i == 1) {
            bVar.e(this.i);
            bVar.e(this.k);
        }
        a2.f3223a.add(this);
        a3.f3223a.add(this);
        a4.f3223a.add(this);
        a5.f3223a.add(this);
        a6.f3223a.add(this);
        if (i == 1) {
            this.i.f3223a.add(this);
            this.k.f3223a.add(this);
        }
    }

    @Override // c.a.a.x.c.a.b
    public void a() {
        this.n = false;
        this.f3189c.invalidateSelf();
    }

    @Override // c.a.a.x.b.c
    public void b(List<c> list, List<c> list2) {
        for (int i = 0; i < list.size(); i++) {
            c cVar = list.get(i);
            if (cVar instanceof s) {
                s sVar = (s) cVar;
                if (sVar.f3219c == 1) {
                    this.m.f3149a.add(sVar);
                    sVar.f3218b.add(this);
                }
            }
        }
    }

    @Override // c.a.a.z.f
    public void c(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2) {
        c.a.a.c0.f.f(eVar, i, list, eVar2, this);
    }

    @Override // c.a.a.x.b.m
    public Path g() {
        float f2;
        float f3;
        float sin;
        double d2;
        float f4;
        float f5;
        float f6;
        float f7;
        float f8;
        float f9;
        double d3;
        float f10;
        float f11;
        double d4;
        double d5;
        double d6;
        if (this.n) {
            return this.f3187a;
        }
        this.f3187a.reset();
        if (this.f3191e) {
            this.n = true;
            return this.f3187a;
        }
        int f12 = m0.f(this.f3190d);
        double d7 = ShadowDrawableWrapper.COS_45;
        if (f12 == 0) {
            float floatValue = this.f3192f.e().floatValue();
            c.a.a.x.c.a<?, Float> aVar = this.f3194h;
            if (aVar != null) {
                d7 = aVar.e().floatValue();
            }
            double radians = Math.toRadians(d7 - 90.0d);
            double d8 = floatValue;
            float f13 = (float) (6.283185307179586d / d8);
            float f14 = f13 / 2.0f;
            float f15 = floatValue - ((int) floatValue);
            int i = (f15 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f15 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
            if (i != 0) {
                radians += (1.0f - f15) * f14;
            }
            float floatValue2 = this.j.e().floatValue();
            float floatValue3 = this.i.e().floatValue();
            c.a.a.x.c.a<?, Float> aVar2 = this.k;
            float floatValue4 = aVar2 != null ? aVar2.e().floatValue() / 100.0f : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            c.a.a.x.c.a<?, Float> aVar3 = this.l;
            float floatValue5 = aVar3 != null ? aVar3.e().floatValue() / 100.0f : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            if (i != 0) {
                f5 = c.b.a.a.a.a(floatValue2, floatValue3, f15, floatValue3);
                double d9 = f5;
                f2 = floatValue3;
                f3 = floatValue4;
                f4 = (float) (Math.cos(radians) * d9);
                sin = (float) (d9 * Math.sin(radians));
                this.f3187a.moveTo(f4, sin);
                d2 = radians + ((f13 * f15) / 2.0f);
            } else {
                f2 = floatValue3;
                f3 = floatValue4;
                double d10 = floatValue2;
                float cos = (float) (Math.cos(radians) * d10);
                sin = (float) (Math.sin(radians) * d10);
                this.f3187a.moveTo(cos, sin);
                d2 = radians + f14;
                f4 = cos;
                f5 = 0.0f;
            }
            double ceil = Math.ceil(d8) * 2.0d;
            int i2 = 0;
            boolean z = false;
            while (true) {
                double d11 = i2;
                if (d11 >= ceil) {
                    break;
                }
                float f16 = z ? floatValue2 : f2;
                int i3 = (f5 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
                if (i3 == 0 || d11 != ceil - 2.0d) {
                    f6 = f13;
                    f7 = f14;
                } else {
                    f6 = f13;
                    f7 = (f13 * f15) / 2.0f;
                }
                if (i3 == 0 || d11 != ceil - 1.0d) {
                    f8 = f5;
                    f5 = f16;
                    f9 = f7;
                } else {
                    f9 = f7;
                    f8 = f5;
                }
                double d12 = f5;
                float cos2 = (float) (Math.cos(d2) * d12);
                float sin2 = (float) (d12 * Math.sin(d2));
                if (f3 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && floatValue5 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    this.f3187a.lineTo(cos2, sin2);
                    f10 = sin2;
                    d3 = d2;
                    f11 = floatValue5;
                } else {
                    d3 = d2;
                    float f17 = sin;
                    double atan2 = (float) (Math.atan2(sin, f4) - 1.5707963267948966d);
                    float cos3 = (float) Math.cos(atan2);
                    float sin3 = (float) Math.sin(atan2);
                    f10 = sin2;
                    f11 = floatValue5;
                    double atan22 = (float) (Math.atan2(sin2, cos2) - 1.5707963267948966d);
                    float cos4 = (float) Math.cos(atan22);
                    float sin4 = (float) Math.sin(atan22);
                    float f18 = z ? f3 : f11;
                    float f19 = z ? f11 : f3;
                    float f20 = (z ? f2 : floatValue2) * f18 * 0.47829f;
                    float f21 = cos3 * f20;
                    float f22 = f20 * sin3;
                    float f23 = (z ? floatValue2 : f2) * f19 * 0.47829f;
                    float f24 = cos4 * f23;
                    float f25 = f23 * sin4;
                    if (i != 0) {
                        if (i2 == 0) {
                            f21 *= f15;
                            f22 *= f15;
                        } else if (d11 == ceil - 1.0d) {
                            f24 *= f15;
                            f25 *= f15;
                        }
                    }
                    this.f3187a.cubicTo(f4 - f21, f17 - f22, cos2 + f24, f10 + f25, cos2, f10);
                }
                d2 = d3 + f9;
                z = !z;
                i2++;
                f4 = cos2;
                f5 = f8;
                f13 = f6;
                sin = f10;
                floatValue5 = f11;
            }
            PointF e2 = this.f3193g.e();
            this.f3187a.offset(e2.x, e2.y);
            this.f3187a.close();
        } else if (f12 == 1) {
            int floor = (int) Math.floor(this.f3192f.e().floatValue());
            c.a.a.x.c.a<?, Float> aVar4 = this.f3194h;
            if (aVar4 != null) {
                d7 = aVar4.e().floatValue();
            }
            double radians2 = Math.toRadians(d7 - 90.0d);
            double d13 = floor;
            float floatValue6 = this.l.e().floatValue() / 100.0f;
            float floatValue7 = this.j.e().floatValue();
            double d14 = floatValue7;
            float cos5 = (float) (Math.cos(radians2) * d14);
            float sin5 = (float) (Math.sin(radians2) * d14);
            this.f3187a.moveTo(cos5, sin5);
            double d15 = (float) (6.283185307179586d / d13);
            double d16 = radians2 + d15;
            double ceil2 = Math.ceil(d13);
            int i4 = 0;
            while (i4 < ceil2) {
                float cos6 = (float) (Math.cos(d16) * d14);
                double d17 = ceil2;
                float sin6 = (float) (Math.sin(d16) * d14);
                if (floatValue6 != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    d5 = d14;
                    d4 = d16;
                    double atan23 = (float) (Math.atan2(sin5, cos5) - 1.5707963267948966d);
                    float cos7 = (float) Math.cos(atan23);
                    d6 = d15;
                    double atan24 = (float) (Math.atan2(sin6, cos6) - 1.5707963267948966d);
                    float f26 = floatValue7 * floatValue6 * 0.25f;
                    this.f3187a.cubicTo(cos5 - (cos7 * f26), sin5 - (((float) Math.sin(atan23)) * f26), cos6 + (((float) Math.cos(atan24)) * f26), sin6 + (f26 * ((float) Math.sin(atan24))), cos6, sin6);
                } else {
                    d4 = d16;
                    d5 = d14;
                    d6 = d15;
                    this.f3187a.lineTo(cos6, sin6);
                }
                d16 = d4 + d6;
                i4++;
                sin5 = sin6;
                cos5 = cos6;
                ceil2 = d17;
                d14 = d5;
                d15 = d6;
            }
            PointF e3 = this.f3193g.e();
            this.f3187a.offset(e3.x, e3.y);
            this.f3187a.close();
        }
        this.f3187a.close();
        this.m.a(this.f3187a);
        this.n = true;
        return this.f3187a;
    }

    @Override // c.a.a.x.b.c
    public String getName() {
        return this.f3188b;
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: c.a.a.d0.c<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        c.a.a.x.c.a<?, Float> aVar;
        c.a.a.x.c.a<?, Float> aVar2;
        if (t == c.a.a.o.s) {
            c.a.a.x.c.a<?, Float> aVar3 = this.f3192f;
            c.a.a.d0.c<Float> cVar2 = aVar3.f3227e;
            aVar3.f3227e = cVar;
        } else if (t == c.a.a.o.t) {
            c.a.a.x.c.a<?, Float> aVar4 = this.f3194h;
            c.a.a.d0.c<Float> cVar3 = aVar4.f3227e;
            aVar4.f3227e = cVar;
        } else if (t == c.a.a.o.j) {
            c.a.a.x.c.a<?, PointF> aVar5 = this.f3193g;
            c.a.a.d0.c<PointF> cVar4 = aVar5.f3227e;
            aVar5.f3227e = cVar;
        } else if (t == c.a.a.o.u && (aVar2 = this.i) != null) {
            c.a.a.d0.c<Float> cVar5 = aVar2.f3227e;
            aVar2.f3227e = cVar;
        } else if (t == c.a.a.o.v) {
            c.a.a.x.c.a<?, Float> aVar6 = this.j;
            c.a.a.d0.c<Float> cVar6 = aVar6.f3227e;
            aVar6.f3227e = cVar;
        } else if (t == c.a.a.o.w && (aVar = this.k) != null) {
            c.a.a.d0.c<Float> cVar7 = aVar.f3227e;
            aVar.f3227e = cVar;
        } else if (t == c.a.a.o.x) {
            c.a.a.x.c.a<?, Float> aVar7 = this.l;
            c.a.a.d0.c<Float> cVar8 = aVar7.f3227e;
            aVar7.f3227e = cVar;
        }
    }
}