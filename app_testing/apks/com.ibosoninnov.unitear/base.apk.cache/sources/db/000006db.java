package c.a.a.z.l;

import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;
import b.d.b.m0;
import c.a.a.j;
import c.a.a.o;
import c.a.a.x.c.n;
import c.a.a.x.c.p;
import c.a.a.z.j.k;
import c.a.a.z.k.m;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/* compiled from: TextLayer.java */
/* loaded from: classes.dex */
public class i extends c.a.a.z.l.b {
    public final Paint A;
    public final Paint B;
    public final Map<c.a.a.z.d, List<c.a.a.x.b.d>> C;
    public final b.f.e<String> D;
    public final n E;
    public final j F;
    public final c.a.a.d G;
    public c.a.a.x.c.a<Integer, Integer> H;
    public c.a.a.x.c.a<Integer, Integer> I;
    public c.a.a.x.c.a<Integer, Integer> J;
    public c.a.a.x.c.a<Integer, Integer> K;
    public c.a.a.x.c.a<Float, Float> L;
    public c.a.a.x.c.a<Float, Float> M;
    public c.a.a.x.c.a<Float, Float> N;
    public c.a.a.x.c.a<Float, Float> O;
    public c.a.a.x.c.a<Float, Float> P;
    public final StringBuilder x;
    public final RectF y;
    public final Matrix z;

    /* compiled from: TextLayer.java */
    /* loaded from: classes.dex */
    public class a extends Paint {
        public a(i iVar, int i) {
            super(i);
            setStyle(Paint.Style.FILL);
        }
    }

    /* compiled from: TextLayer.java */
    /* loaded from: classes.dex */
    public class b extends Paint {
        public b(i iVar, int i) {
            super(i);
            setStyle(Paint.Style.STROKE);
        }
    }

    public i(j jVar, e eVar) {
        super(jVar, eVar);
        c.a.a.z.j.b bVar;
        c.a.a.z.j.b bVar2;
        c.a.a.z.j.a aVar;
        c.a.a.z.j.a aVar2;
        this.x = new StringBuilder(2);
        this.y = new RectF();
        this.z = new Matrix();
        this.A = new a(this, 1);
        this.B = new b(this, 1);
        this.C = new HashMap();
        this.D = new b.f.e<>(10);
        this.F = jVar;
        this.G = eVar.f3396b;
        n nVar = new n(eVar.q.f3301a);
        this.E = nVar;
        nVar.f3223a.add(this);
        e(nVar);
        k kVar = eVar.r;
        if (kVar != null && (aVar2 = kVar.f3289a) != null) {
            c.a.a.x.c.a<Integer, Integer> a2 = aVar2.a();
            this.H = a2;
            a2.f3223a.add(this);
            e(this.H);
        }
        if (kVar != null && (aVar = kVar.f3290b) != null) {
            c.a.a.x.c.a<Integer, Integer> a3 = aVar.a();
            this.J = a3;
            a3.f3223a.add(this);
            e(this.J);
        }
        if (kVar != null && (bVar2 = kVar.f3291c) != null) {
            c.a.a.x.c.a<Float, Float> a4 = bVar2.a();
            this.L = a4;
            a4.f3223a.add(this);
            e(this.L);
        }
        if (kVar == null || (bVar = kVar.f3292d) == null) {
            return;
        }
        c.a.a.x.c.a<Float, Float> a5 = bVar.a();
        this.N = a5;
        a5.f3223a.add(this);
        e(this.N);
    }

    @Override // c.a.a.z.l.b, c.a.a.x.b.e
    public void d(RectF rectF, Matrix matrix, boolean z) {
        super.d(rectF, matrix, z);
        rectF.set(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.G.j.width(), this.G.j.height());
    }

    @Override // c.a.a.z.l.b, c.a.a.z.f
    public <T> void h(T t, c.a.a.d0.c<T> cVar) {
        this.v.c(t, cVar);
        if (t == o.f3114a) {
            c.a.a.x.c.a<Integer, Integer> aVar = this.I;
            if (aVar != null) {
                this.u.remove(aVar);
            }
            if (cVar == null) {
                this.I = null;
                return;
            }
            p pVar = new p(cVar, null);
            this.I = pVar;
            pVar.f3223a.add(this);
            e(this.I);
        } else if (t == o.f3115b) {
            c.a.a.x.c.a<Integer, Integer> aVar2 = this.K;
            if (aVar2 != null) {
                this.u.remove(aVar2);
            }
            if (cVar == null) {
                this.K = null;
                return;
            }
            p pVar2 = new p(cVar, null);
            this.K = pVar2;
            pVar2.f3223a.add(this);
            e(this.K);
        } else if (t == o.o) {
            c.a.a.x.c.a<Float, Float> aVar3 = this.M;
            if (aVar3 != null) {
                this.u.remove(aVar3);
            }
            if (cVar == null) {
                this.M = null;
                return;
            }
            p pVar3 = new p(cVar, null);
            this.M = pVar3;
            pVar3.f3223a.add(this);
            e(this.M);
        } else if (t == o.p) {
            c.a.a.x.c.a<Float, Float> aVar4 = this.O;
            if (aVar4 != null) {
                this.u.remove(aVar4);
            }
            if (cVar == null) {
                this.O = null;
                return;
            }
            p pVar4 = new p(cVar, null);
            this.O = pVar4;
            pVar4.f3223a.add(this);
            e(this.O);
        } else if (t == o.B) {
            c.a.a.x.c.a<Float, Float> aVar5 = this.P;
            if (aVar5 != null) {
                this.u.remove(aVar5);
            }
            if (cVar == null) {
                this.P = null;
                return;
            }
            p pVar5 = new p(cVar, null);
            this.P = pVar5;
            pVar5.f3223a.add(this);
            e(this.P);
        }
    }

    /* JADX DEBUG: Multi-variable search result rejected for r8v13, resolved type: java.util.Map<java.lang.String, android.graphics.Typeface> */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r3v4, types: [T, java.lang.String] */
    /* JADX WARN: Type inference failed for: r6v1, types: [T, java.lang.Object, java.lang.String] */
    @Override // c.a.a.z.l.b
    public void k(Canvas canvas, Matrix matrix, int i) {
        c.a.a.y.a aVar;
        float f2;
        String str;
        float floatValue;
        float f3;
        List<String> list;
        int i2;
        String str2;
        List<c.a.a.x.b.d> list2;
        float floatValue2;
        String str3;
        float f4;
        int i3;
        canvas.save();
        if (!(this.F.f3075c.f3043g.i() > 0)) {
            canvas.setMatrix(matrix);
        }
        c.a.a.z.b e2 = this.E.e();
        c.a.a.z.c cVar = this.G.f3041e.get(e2.f3262b);
        if (cVar == null) {
            canvas.restore();
            return;
        }
        c.a.a.x.c.a<Integer, Integer> aVar2 = this.I;
        if (aVar2 != null) {
            this.A.setColor(aVar2.e().intValue());
        } else {
            c.a.a.x.c.a<Integer, Integer> aVar3 = this.H;
            if (aVar3 != null) {
                this.A.setColor(aVar3.e().intValue());
            } else {
                this.A.setColor(e2.f3268h);
            }
        }
        c.a.a.x.c.a<Integer, Integer> aVar4 = this.K;
        if (aVar4 != null) {
            this.B.setColor(aVar4.e().intValue());
        } else {
            c.a.a.x.c.a<Integer, Integer> aVar5 = this.J;
            if (aVar5 != null) {
                this.B.setColor(aVar5.e().intValue());
            } else {
                this.B.setColor(e2.i);
            }
        }
        c.a.a.x.c.a<Integer, Integer> aVar6 = this.v.j;
        int intValue = ((aVar6 == null ? 100 : aVar6.e().intValue()) * 255) / 100;
        this.A.setAlpha(intValue);
        this.B.setAlpha(intValue);
        c.a.a.x.c.a<Float, Float> aVar7 = this.M;
        if (aVar7 != null) {
            this.B.setStrokeWidth(aVar7.e().floatValue());
        } else {
            c.a.a.x.c.a<Float, Float> aVar8 = this.L;
            if (aVar8 != null) {
                this.B.setStrokeWidth(aVar8.e().floatValue());
            } else {
                this.B.setStrokeWidth(c.a.a.c0.g.c() * e2.j * c.a.a.c0.g.d(matrix));
            }
        }
        if (this.F.f3075c.f3043g.i() > 0) {
            c.a.a.x.c.a<Float, Float> aVar9 = this.P;
            if (aVar9 != null) {
                f3 = aVar9.e().floatValue();
            } else {
                f3 = e2.f3263c;
            }
            float f5 = f3 / 100.0f;
            float d2 = c.a.a.c0.g.d(matrix);
            String str4 = e2.f3261a;
            float c2 = c.a.a.c0.g.c() * e2.f3266f;
            List<String> u = u(str4);
            int size = u.size();
            int i4 = 0;
            while (i4 < size) {
                String str5 = u.get(i4);
                float f6 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                int i5 = 0;
                while (i5 < str5.length()) {
                    c.a.a.z.d d3 = this.G.f3043g.d(c.a.a.z.d.a(str5.charAt(i5), cVar.f3269a, cVar.f3271c));
                    if (d3 == null) {
                        f4 = c2;
                        i3 = i4;
                        str3 = str5;
                    } else {
                        str3 = str5;
                        double d4 = d3.f3274c;
                        f4 = c2;
                        i3 = i4;
                        f6 = (float) ((d4 * f5 * c.a.a.c0.g.c() * d2) + f6);
                    }
                    i5++;
                    str5 = str3;
                    c2 = f4;
                    i4 = i3;
                }
                float f7 = c2;
                int i6 = i4;
                String str6 = str5;
                canvas.save();
                r(e2.f3264d, canvas, f6);
                canvas.translate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (i6 * f7) - (((size - 1) * f7) / 2.0f));
                int i7 = 0;
                while (i7 < str6.length()) {
                    String str7 = str6;
                    c.a.a.z.d d5 = this.G.f3043g.d(c.a.a.z.d.a(str7.charAt(i7), cVar.f3269a, cVar.f3271c));
                    if (d5 == null) {
                        list = u;
                        i2 = size;
                        str2 = str7;
                    } else {
                        if (this.C.containsKey(d5)) {
                            list2 = this.C.get(d5);
                            list = u;
                            i2 = size;
                            str2 = str7;
                        } else {
                            List<m> list3 = d5.f3272a;
                            int size2 = list3.size();
                            ArrayList arrayList = new ArrayList(size2);
                            list = u;
                            int i8 = 0;
                            while (i8 < size2) {
                                arrayList.add(new c.a.a.x.b.d(this.F, this, list3.get(i8)));
                                i8++;
                                str7 = str7;
                                size = size;
                                list3 = list3;
                            }
                            i2 = size;
                            str2 = str7;
                            this.C.put(d5, arrayList);
                            list2 = arrayList;
                        }
                        int i9 = 0;
                        while (i9 < list2.size()) {
                            Path g2 = list2.get(i9).g();
                            g2.computeBounds(this.y, false);
                            this.z.set(matrix);
                            List<c.a.a.x.b.d> list4 = list2;
                            this.z.preTranslate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (-e2.f3267g) * c.a.a.c0.g.c());
                            this.z.preScale(f5, f5);
                            g2.transform(this.z);
                            if (e2.k) {
                                t(g2, this.A, canvas);
                                t(g2, this.B, canvas);
                            } else {
                                t(g2, this.B, canvas);
                                t(g2, this.A, canvas);
                            }
                            i9++;
                            list2 = list4;
                        }
                        float c3 = c.a.a.c0.g.c() * ((float) d5.f3274c) * f5 * d2;
                        float f8 = e2.f3265e / 10.0f;
                        c.a.a.x.c.a<Float, Float> aVar10 = this.O;
                        if (aVar10 != null) {
                            floatValue2 = aVar10.e().floatValue();
                        } else {
                            c.a.a.x.c.a<Float, Float> aVar11 = this.N;
                            if (aVar11 != null) {
                                floatValue2 = aVar11.e().floatValue();
                            }
                            canvas.translate((f8 * d2) + c3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        }
                        f8 += floatValue2;
                        canvas.translate((f8 * d2) + c3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    }
                    i7++;
                    u = list;
                    str6 = str2;
                    size = i2;
                }
                canvas.restore();
                i4 = i6 + 1;
                c2 = f7;
            }
        } else {
            float d6 = c.a.a.c0.g.d(matrix);
            j jVar = this.F;
            ?? r6 = cVar.f3269a;
            ?? r3 = cVar.f3271c;
            Typeface typeface = null;
            if (jVar.getCallback() == null) {
                aVar = null;
            } else {
                if (jVar.n == null) {
                    jVar.n = new c.a.a.y.a(jVar.getCallback());
                }
                aVar = jVar.n;
            }
            if (aVar != null) {
                c.a.a.z.i<String> iVar = aVar.f3248a;
                iVar.f3284a = r6;
                iVar.f3285b = r3;
                typeface = aVar.f3249b.get(iVar);
                if (typeface == null) {
                    Typeface typeface2 = aVar.f3250c.get(r6);
                    if (typeface2 == null) {
                        StringBuilder A = c.b.a.a.a.A("fonts/", r6);
                        A.append(aVar.f3252e);
                        typeface2 = Typeface.createFromAsset(aVar.f3251d, A.toString());
                        aVar.f3250c.put(r6, typeface2);
                    }
                    boolean contains = r3.contains("Italic");
                    boolean contains2 = r3.contains("Bold");
                    int i10 = (contains && contains2) ? 3 : contains ? 2 : contains2 ? 1 : 0;
                    typeface = typeface2.getStyle() == i10 ? typeface2 : Typeface.create(typeface2, i10);
                    aVar.f3249b.put(aVar.f3248a, typeface);
                }
            }
            if (typeface != null) {
                String str8 = e2.f3261a;
                Objects.requireNonNull(this.F);
                this.A.setTypeface(typeface);
                c.a.a.x.c.a<Float, Float> aVar12 = this.P;
                if (aVar12 != null) {
                    f2 = aVar12.e().floatValue();
                } else {
                    f2 = e2.f3263c;
                }
                this.A.setTextSize(c.a.a.c0.g.c() * f2);
                this.B.setTypeface(this.A.getTypeface());
                this.B.setTextSize(this.A.getTextSize());
                float c4 = c.a.a.c0.g.c() * e2.f3266f;
                List<String> u2 = u(str8);
                int size3 = u2.size();
                for (int i11 = 0; i11 < size3; i11++) {
                    String str9 = u2.get(i11);
                    r(e2.f3264d, canvas, this.B.measureText(str9));
                    canvas.translate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, (i11 * c4) - (((size3 - 1) * c4) / 2.0f));
                    int i12 = 0;
                    while (i12 < str9.length()) {
                        int codePointAt = str9.codePointAt(i12);
                        int charCount = Character.charCount(codePointAt) + i12;
                        while (charCount < str9.length()) {
                            int codePointAt2 = str9.codePointAt(charCount);
                            if (!(Character.getType(codePointAt2) == 16 || Character.getType(codePointAt2) == 27 || Character.getType(codePointAt2) == 6 || Character.getType(codePointAt2) == 28 || Character.getType(codePointAt2) == 19)) {
                                break;
                            }
                            charCount += Character.charCount(codePointAt2);
                            codePointAt = (codePointAt * 31) + codePointAt2;
                        }
                        b.f.e<String> eVar = this.D;
                        int i13 = size3;
                        float f9 = c4;
                        long j = codePointAt;
                        if (eVar.f1750c) {
                            eVar.c();
                        }
                        if (b.f.d.b(eVar.f1751d, eVar.f1753f, j) >= 0) {
                            str = this.D.d(j);
                        } else {
                            this.x.setLength(0);
                            int i14 = i12;
                            while (i14 < charCount) {
                                int codePointAt3 = str9.codePointAt(i14);
                                this.x.appendCodePoint(codePointAt3);
                                i14 += Character.charCount(codePointAt3);
                            }
                            String sb = this.x.toString();
                            this.D.g(j, sb);
                            str = sb;
                        }
                        i12 += str.length();
                        if (e2.k) {
                            s(str, this.A, canvas);
                            s(str, this.B, canvas);
                        } else {
                            s(str, this.B, canvas);
                            s(str, this.A, canvas);
                        }
                        float measureText = this.A.measureText(str, 0, 1);
                        float f10 = e2.f3265e / 10.0f;
                        c.a.a.x.c.a<Float, Float> aVar13 = this.O;
                        if (aVar13 != null) {
                            floatValue = aVar13.e().floatValue();
                        } else {
                            c.a.a.x.c.a<Float, Float> aVar14 = this.N;
                            if (aVar14 != null) {
                                floatValue = aVar14.e().floatValue();
                            } else {
                                canvas.translate((f10 * d6) + measureText, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                                c4 = f9;
                                size3 = i13;
                            }
                        }
                        f10 += floatValue;
                        canvas.translate((f10 * d6) + measureText, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        c4 = f9;
                        size3 = i13;
                    }
                    canvas.setMatrix(matrix);
                }
            }
        }
        canvas.restore();
    }

    public final void r(int i, Canvas canvas, float f2) {
        int f3 = m0.f(i);
        if (f3 == 1) {
            canvas.translate(-f2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        } else if (f3 != 2) {
        } else {
            canvas.translate((-f2) / 2.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
    }

    public final void s(String str, Paint paint, Canvas canvas) {
        if (paint.getColor() == 0) {
            return;
        }
        if (paint.getStyle() == Paint.Style.STROKE && paint.getStrokeWidth() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return;
        }
        canvas.drawText(str, 0, str.length(), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, paint);
    }

    public final void t(Path path, Paint paint, Canvas canvas) {
        if (paint.getColor() == 0) {
            return;
        }
        if (paint.getStyle() == Paint.Style.STROKE && paint.getStrokeWidth() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            return;
        }
        canvas.drawPath(path, paint);
    }

    public final List<String> u(String str) {
        return Arrays.asList(str.replaceAll("\r\n", "\r").replaceAll("\n", "\r").split("\r"));
    }
}