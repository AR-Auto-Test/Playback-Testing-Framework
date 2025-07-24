package c.a.a.b0;

import android.graphics.Rect;
import c.a.a.b0.h0.c;
import c.a.a.z.l.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.HashMap;

/* compiled from: LottieCompositionMoshiParser.java */
/* loaded from: classes.dex */
public class s {

    /* renamed from: a  reason: collision with root package name */
    public static final c.a f3007a = c.a.a("w", "h", "ip", "op", "fr", "v", "layers", "assets", "fonts", "chars", "markers");

    /* renamed from: b  reason: collision with root package name */
    public static c.a f3008b = c.a.a("id", "layers", "w", "h", "p", "u");

    /* renamed from: c  reason: collision with root package name */
    public static final c.a f3009c = c.a.a("list");

    /* renamed from: d  reason: collision with root package name */
    public static final c.a f3010d = c.a.a("cm", "tm", "dr");

    public static c.a.a.d a(c.a.a.b0.h0.c cVar) {
        ArrayList arrayList;
        ArrayList arrayList2;
        b.f.i<c.a.a.z.d> iVar;
        c.a.a.d dVar;
        ArrayList arrayList3;
        c.a.a.d dVar2;
        float f2;
        float f3;
        c.a.a.d dVar3;
        float c2 = c.a.a.c0.g.c();
        b.f.e<c.a.a.z.l.e> eVar = new b.f.e<>(10);
        ArrayList arrayList4 = new ArrayList();
        HashMap hashMap = new HashMap();
        HashMap hashMap2 = new HashMap();
        HashMap hashMap3 = new HashMap();
        ArrayList arrayList5 = new ArrayList();
        b.f.i<c.a.a.z.d> iVar2 = new b.f.i<>(10);
        c.a.a.d dVar4 = new c.a.a.d();
        cVar.C();
        float f4 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        int i = 0;
        int i2 = 0;
        float f5 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        float f6 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        while (cVar.G()) {
            switch (cVar.O(f3007a)) {
                case 0:
                    arrayList = arrayList5;
                    i = cVar.J();
                    break;
                case 1:
                    arrayList = arrayList5;
                    i2 = cVar.J();
                    break;
                case 2:
                    arrayList2 = arrayList5;
                    iVar = iVar2;
                    dVar = dVar4;
                    f5 = (float) cVar.I();
                    dVar4 = dVar;
                    arrayList5 = arrayList2;
                    iVar2 = iVar;
                case 3:
                    arrayList2 = arrayList5;
                    iVar = iVar2;
                    dVar = dVar4;
                    f4 = ((float) cVar.I()) - 0.01f;
                    dVar4 = dVar;
                    arrayList5 = arrayList2;
                    iVar2 = iVar;
                case 4:
                    arrayList2 = arrayList5;
                    iVar = iVar2;
                    dVar = dVar4;
                    f6 = (float) cVar.I();
                    dVar4 = dVar;
                    arrayList5 = arrayList2;
                    iVar2 = iVar;
                case 5:
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    dVar2 = dVar4;
                    f2 = f4;
                    f3 = f5;
                    String[] split = cVar.L().split("\\.");
                    int parseInt = Integer.parseInt(split[0]);
                    int parseInt2 = Integer.parseInt(split[1]);
                    if (!(parseInt >= 4 && (parseInt > 4 || (parseInt2 >= 4 && (parseInt2 > 4 || Integer.parseInt(split[2]) >= 0))))) {
                        dVar2.a("Lottie only supports bodymovin >= 4.4.0");
                    }
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                case 6:
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    c.a.a.d dVar5 = dVar4;
                    f2 = f4;
                    f3 = f5;
                    cVar.B();
                    int i3 = 0;
                    while (cVar.G()) {
                        c.a.a.d dVar6 = dVar5;
                        c.a.a.z.l.e a2 = r.a(cVar, dVar6);
                        if (a2.f3399e == e.a.IMAGE) {
                            i3++;
                        }
                        arrayList4.add(a2);
                        eVar.g(a2.f3398d, a2);
                        if (i3 > 4) {
                            c.a.a.c0.c.b("You have " + i3 + " images. Lottie should primarily be used with shapes. If you are using Adobe Illustrator, convert the Illustrator layers to shape layers.");
                        }
                        dVar5 = dVar6;
                    }
                    dVar2 = dVar5;
                    cVar.D();
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                case 7:
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    f2 = f4;
                    f3 = f5;
                    cVar.B();
                    while (cVar.G()) {
                        ArrayList arrayList6 = new ArrayList();
                        b.f.e eVar2 = new b.f.e(10);
                        cVar.C();
                        String str = null;
                        String str2 = null;
                        String str3 = null;
                        int i4 = 0;
                        int i5 = 0;
                        while (cVar.G()) {
                            int O = cVar.O(f3008b);
                            if (O != 0) {
                                if (O == 1) {
                                    cVar.B();
                                    while (cVar.G()) {
                                        c.a.a.z.l.e a3 = r.a(cVar, dVar4);
                                        eVar2.g(a3.f3398d, a3);
                                        arrayList6.add(a3);
                                        dVar4 = dVar4;
                                    }
                                    dVar3 = dVar4;
                                    cVar.D();
                                } else if (O == 2) {
                                    i4 = cVar.J();
                                } else if (O == 3) {
                                    i5 = cVar.J();
                                } else if (O == 4) {
                                    str2 = cVar.L();
                                } else if (O != 5) {
                                    cVar.P();
                                    cVar.Q();
                                    dVar3 = dVar4;
                                } else {
                                    str3 = cVar.L();
                                }
                                dVar4 = dVar3;
                            } else {
                                str = cVar.L();
                            }
                        }
                        c.a.a.d dVar7 = dVar4;
                        cVar.E();
                        if (str2 != null) {
                            hashMap2.put(str, new c.a.a.k(i4, i5, str, str2, str3));
                        } else {
                            hashMap.put(str, arrayList6);
                        }
                        dVar4 = dVar7;
                    }
                    cVar.D();
                    dVar2 = dVar4;
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                case 8:
                    f2 = f4;
                    f3 = f5;
                    cVar.C();
                    while (cVar.G()) {
                        if (cVar.O(f3009c) != 0) {
                            cVar.P();
                            cVar.Q();
                        } else {
                            cVar.B();
                            while (cVar.G()) {
                                c.a aVar = j.f2991a;
                                cVar.C();
                                String str4 = null;
                                String str5 = null;
                                String str6 = null;
                                float f7 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                while (cVar.G()) {
                                    ArrayList arrayList7 = arrayList5;
                                    int O2 = cVar.O(j.f2991a);
                                    if (O2 != 0) {
                                        b.f.i<c.a.a.z.d> iVar3 = iVar2;
                                        if (O2 == 1) {
                                            str5 = cVar.L();
                                        } else if (O2 == 2) {
                                            str6 = cVar.L();
                                        } else if (O2 != 3) {
                                            cVar.P();
                                            cVar.Q();
                                        } else {
                                            f7 = (float) cVar.I();
                                        }
                                        arrayList5 = arrayList7;
                                        iVar2 = iVar3;
                                    } else {
                                        str4 = cVar.L();
                                        arrayList5 = arrayList7;
                                    }
                                }
                                cVar.E();
                                hashMap3.put(str5, new c.a.a.z.c(str4, str5, str6, f7));
                                arrayList5 = arrayList5;
                            }
                            cVar.D();
                        }
                    }
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    cVar.E();
                    dVar2 = dVar4;
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                case 9:
                    f2 = f4;
                    f3 = f5;
                    cVar.B();
                    while (cVar.G()) {
                        c.a aVar2 = i.f2989a;
                        ArrayList arrayList8 = new ArrayList();
                        cVar.C();
                        double d2 = 0.0d;
                        double d3 = 0.0d;
                        String str7 = null;
                        String str8 = null;
                        char c3 = 0;
                        while (cVar.G()) {
                            int O3 = cVar.O(i.f2989a);
                            if (O3 == 0) {
                                c3 = cVar.L().charAt(0);
                            } else if (O3 == 1) {
                                d2 = cVar.I();
                            } else if (O3 == 2) {
                                d3 = cVar.I();
                            } else if (O3 == 3) {
                                str7 = cVar.L();
                            } else if (O3 == 4) {
                                str8 = cVar.L();
                            } else if (O3 != 5) {
                                cVar.P();
                                cVar.Q();
                            } else {
                                cVar.C();
                                while (cVar.G()) {
                                    if (cVar.O(i.f2990b) != 0) {
                                        cVar.P();
                                        cVar.Q();
                                    } else {
                                        cVar.B();
                                        while (cVar.G()) {
                                            arrayList8.add((c.a.a.z.k.m) f.a(cVar, dVar4));
                                        }
                                        cVar.D();
                                    }
                                }
                                cVar.E();
                            }
                        }
                        cVar.E();
                        c.a.a.z.d dVar8 = new c.a.a.z.d(arrayList8, c3, d2, d3, str7, str8);
                        iVar2.g(dVar8.hashCode(), dVar8);
                    }
                    cVar.D();
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    dVar2 = dVar4;
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                case 10:
                    cVar.B();
                    while (cVar.G()) {
                        cVar.C();
                        String str9 = null;
                        float f8 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        float f9 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        while (cVar.G()) {
                            int O4 = cVar.O(f3010d);
                            if (O4 != 0) {
                                float f10 = f4;
                                if (O4 == 1) {
                                    f8 = (float) cVar.I();
                                } else if (O4 != 2) {
                                    cVar.P();
                                    cVar.Q();
                                } else {
                                    f9 = (float) cVar.I();
                                }
                                f4 = f10;
                            } else {
                                str9 = cVar.L();
                            }
                        }
                        cVar.E();
                        arrayList5.add(new c.a.a.z.h(str9, f8, f9));
                        f5 = f5;
                        f4 = f4;
                    }
                    f2 = f4;
                    f3 = f5;
                    cVar.D();
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    dVar2 = dVar4;
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
                default:
                    arrayList3 = arrayList5;
                    iVar = iVar2;
                    dVar2 = dVar4;
                    f2 = f4;
                    f3 = f5;
                    cVar.P();
                    cVar.Q();
                    dVar4 = dVar2;
                    f5 = f3;
                    arrayList5 = arrayList3;
                    f4 = f2;
                    iVar2 = iVar;
            }
            arrayList5 = arrayList;
        }
        ArrayList arrayList9 = arrayList5;
        c.a.a.d dVar9 = dVar4;
        dVar9.j = new Rect(0, 0, (int) (i * c2), (int) (i2 * c2));
        dVar9.k = f5;
        dVar9.l = f4;
        dVar9.m = f6;
        dVar9.i = arrayList4;
        dVar9.f3044h = eVar;
        dVar9.f3039c = hashMap;
        dVar9.f3040d = hashMap2;
        dVar9.f3043g = iVar2;
        dVar9.f3041e = hashMap3;
        dVar9.f3042f = arrayList9;
        return dVar9;
    }
}