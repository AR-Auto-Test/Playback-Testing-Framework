package c.a.a.b0;

import android.graphics.Color;
import b.d.b.m0;
import c.a.a.b0.h0.c;
import c.a.a.z.l.e;
import com.google.android.gms.common.GoogleApiAvailabilityLight;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;

/* compiled from: LayerParser.java */
/* loaded from: classes.dex */
public class r {

    /* renamed from: a  reason: collision with root package name */
    public static final c.a f3004a = c.a.a("nm", "ind", "refId", "ty", "parent", "sw", "sh", "sc", "ks", "tt", "masksProperties", "shapes", "t", "ef", "sr", "st", "w", "h", "ip", "op", "tm", "cl", "hd");

    /* renamed from: b  reason: collision with root package name */
    public static final c.a f3005b = c.a.a(GoogleApiAvailabilityLight.TRACKING_SOURCE_DIALOG, "a");

    /* renamed from: c  reason: collision with root package name */
    public static final c.a f3006c = c.a.a("nm");

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:218:0x02d3 */
    /* JADX DEBUG: Multi-variable search result rejected for r6v12, resolved type: c.a.a.z.j.d */
    /* JADX DEBUG: Multi-variable search result rejected for r6v34, resolved type: c.a.a.z.j.d */
    /* JADX DEBUG: Multi-variable search result rejected for r6v41, resolved type: c.a.a.z.j.d */
    /* JADX DEBUG: Multi-variable search result rejected for r6v8, resolved type: c.a.a.z.j.d */
    /* JADX DEBUG: Multi-variable search result rejected for r6v9, resolved type: c.a.a.z.j.d */
    /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r6v11 */
    public static c.a.a.z.l.e a(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        ArrayList arrayList;
        ArrayList arrayList2;
        String str;
        String str2;
        long j;
        char c2;
        int i;
        c.a.a.z.j.d dVar2;
        String str3;
        String str4;
        ArrayList arrayList3 = new ArrayList();
        ArrayList arrayList4 = new ArrayList();
        cVar.C();
        Float valueOf = Float.valueOf(1.0f);
        Float valueOf2 = Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        boolean z = false;
        float f2 = 1.0f;
        int i2 = 0;
        int i3 = 0;
        int i4 = 0;
        int i5 = 0;
        int i6 = 0;
        boolean z2 = false;
        float f3 = 0.0f;
        float f4 = 0.0f;
        long j2 = -1;
        String str5 = null;
        e.a aVar = null;
        String str6 = null;
        c.a.a.z.j.l lVar = null;
        c.a.a.z.j.j jVar = null;
        c.a.a.z.j.k kVar = null;
        int i7 = 1;
        c.a.a.z.j.b bVar = null;
        long j3 = 0;
        String str7 = "UNSET";
        float f5 = 0.0f;
        while (cVar.G()) {
            switch (cVar.O(f3004a)) {
                case 0:
                    str2 = str5;
                    str7 = cVar.L();
                    str5 = str2;
                    z = false;
                case 1:
                    str2 = str5;
                    j3 = cVar.J();
                    str5 = str2;
                    z = false;
                case 2:
                    str2 = str5;
                    str6 = cVar.L();
                    str5 = str2;
                    z = false;
                case 3:
                    str2 = str5;
                    j = j3;
                    int J = cVar.J();
                    aVar = J < 6 ? e.a.values()[J] : e.a.UNKNOWN;
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 4:
                    str2 = str5;
                    j = j3;
                    j2 = cVar.J();
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 5:
                    str2 = str5;
                    i2 = (int) (c.a.a.c0.g.c() * cVar.J());
                    str5 = str2;
                    z = false;
                case 6:
                    str2 = str5;
                    i3 = (int) (c.a.a.c0.g.c() * cVar.J());
                    str5 = str2;
                    z = false;
                case 7:
                    str2 = str5;
                    i4 = Color.parseColor(cVar.L());
                    str5 = str2;
                    z = false;
                case 8:
                    str2 = str5;
                    lVar = c.a(cVar, dVar);
                    str5 = str2;
                    z = false;
                case 9:
                    str2 = str5;
                    j = j3;
                    i7 = m0.com$airbnb$lottie$model$layer$Layer$MatteType$s$values()[cVar.J()];
                    dVar.o++;
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 10:
                    str2 = str5;
                    c.a.a.z.j.h hVar = null;
                    cVar.B();
                    while (cVar.G()) {
                        cVar.C();
                        c.a.a.z.j.h hVar2 = hVar;
                        c.a.a.z.j.d dVar3 = hVar2;
                        int i8 = 0;
                        boolean z3 = false;
                        while (cVar.G()) {
                            String K = cVar.K();
                            K.hashCode();
                            char c3 = 65535;
                            long j4 = j3;
                            switch (K.hashCode()) {
                                case 111:
                                    if (K.equals("o")) {
                                        c2 = 0;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 3588:
                                    if (K.equals("pt")) {
                                        c2 = 1;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 104433:
                                    if (K.equals("inv")) {
                                        c2 = 2;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                case 3357091:
                                    if (K.equals("mode")) {
                                        c2 = 3;
                                        break;
                                    }
                                    c2 = 65535;
                                    break;
                                default:
                                    c2 = 65535;
                                    break;
                            }
                            switch (c2) {
                                case 0:
                                    dVar2 = b.v.u.c.u(cVar, dVar);
                                    i = i8;
                                    dVar3 = dVar2;
                                    i8 = i;
                                    break;
                                case 1:
                                    hVar2 = new c.a.a.z.j.h(q.a(cVar, dVar, c.a.a.c0.g.c(), a0.f2955a));
                                    dVar2 = dVar3;
                                    i = i8;
                                    dVar3 = dVar2;
                                    i8 = i;
                                    break;
                                case 2:
                                    z3 = cVar.H();
                                    dVar2 = dVar3;
                                    i = i8;
                                    dVar3 = dVar2;
                                    i8 = i;
                                    break;
                                case 3:
                                    String L = cVar.L();
                                    L.hashCode();
                                    switch (L.hashCode()) {
                                        case 97:
                                            if (L.equals("a")) {
                                                c3 = 0;
                                                break;
                                            }
                                            break;
                                        case 105:
                                            if (L.equals("i")) {
                                                c3 = 1;
                                                break;
                                            }
                                            break;
                                        case 110:
                                            if (L.equals(GoogleApiAvailabilityLight.TRACKING_SOURCE_NOTIFICATION)) {
                                                c3 = 2;
                                                break;
                                            }
                                            break;
                                        case 115:
                                            if (L.equals("s")) {
                                                c3 = 3;
                                                break;
                                            }
                                            break;
                                    }
                                    switch (c3) {
                                        case 0:
                                            i8 = 1;
                                            break;
                                        case 1:
                                            dVar.a("Animation contains intersect masks. They are not supported but will be treated like add masks.");
                                            i = 3;
                                            dVar3 = dVar3;
                                            i8 = i;
                                            break;
                                        case 2:
                                            i8 = 4;
                                            dVar2 = dVar3;
                                            i = i8;
                                            dVar3 = dVar2;
                                            i8 = i;
                                            break;
                                        case 3:
                                            i = 2;
                                            dVar3 = dVar3;
                                            i8 = i;
                                            break;
                                        default:
                                            c.a.a.c0.c.b("Unknown mask mode " + K + ". Defaulting to Add.");
                                            i8 = 1;
                                            break;
                                    }
                                default:
                                    cVar.Q();
                                    dVar2 = dVar3;
                                    i = i8;
                                    dVar3 = dVar2;
                                    i8 = i;
                                    break;
                            }
                            j3 = j4;
                            dVar3 = dVar3;
                        }
                        cVar.E();
                        arrayList3.add(new c.a.a.z.k.f(i8, hVar2, dVar3, z3));
                        hVar = null;
                    }
                    j = j3;
                    dVar.o += arrayList3.size();
                    cVar.D();
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 11:
                    str2 = str5;
                    cVar.B();
                    while (cVar.G()) {
                        c.a.a.z.k.b a2 = f.a(cVar, dVar);
                        if (a2 != null) {
                            arrayList4.add(a2);
                        }
                    }
                    cVar.D();
                    j = j3;
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 12:
                    cVar.C();
                    while (cVar.G()) {
                        int O = cVar.O(f3005b);
                        if (O == 0) {
                            str3 = str5;
                            jVar = new c.a.a.z.j.j(b.v.u.c.q(cVar, dVar, g.f2970a));
                        } else if (O != 1) {
                            cVar.P();
                            cVar.Q();
                        } else {
                            cVar.B();
                            if (cVar.G()) {
                                c.a aVar2 = b.f2957a;
                                cVar.C();
                                c.a.a.z.j.k kVar2 = null;
                                while (cVar.G()) {
                                    if (cVar.O(b.f2957a) != 0) {
                                        cVar.P();
                                        cVar.Q();
                                    } else {
                                        cVar.C();
                                        c.a.a.z.j.b bVar2 = null;
                                        c.a.a.z.j.b bVar3 = null;
                                        c.a.a.z.j.a aVar3 = null;
                                        c.a.a.z.j.a aVar4 = null;
                                        while (cVar.G()) {
                                            int O2 = cVar.O(b.f2958b);
                                            if (O2 != 0) {
                                                str4 = str5;
                                                if (O2 == 1) {
                                                    aVar4 = b.v.u.c.r(cVar, dVar);
                                                } else if (O2 == 2) {
                                                    bVar2 = b.v.u.c.s(cVar, dVar);
                                                } else if (O2 != 3) {
                                                    cVar.P();
                                                    cVar.Q();
                                                } else {
                                                    bVar3 = b.v.u.c.s(cVar, dVar);
                                                }
                                            } else {
                                                str4 = str5;
                                                aVar3 = b.v.u.c.r(cVar, dVar);
                                            }
                                            str5 = str4;
                                        }
                                        cVar.E();
                                        kVar2 = new c.a.a.z.j.k(aVar3, aVar4, bVar2, bVar3);
                                        str5 = str5;
                                    }
                                }
                                str3 = str5;
                                cVar.E();
                                if (kVar2 == null) {
                                    kVar2 = new c.a.a.z.j.k(null, null, null, null);
                                }
                                kVar = kVar2;
                            } else {
                                str3 = str5;
                            }
                            while (cVar.G()) {
                                cVar.Q();
                            }
                            cVar.D();
                        }
                        str5 = str3;
                    }
                    str2 = str5;
                    cVar.E();
                    str5 = str2;
                    z = false;
                case 13:
                    cVar.B();
                    ArrayList arrayList5 = new ArrayList();
                    while (cVar.G()) {
                        cVar.C();
                        while (cVar.G()) {
                            if (cVar.O(f3006c) != 0) {
                                cVar.P();
                                cVar.Q();
                            } else {
                                arrayList5.add(cVar.L());
                            }
                        }
                        cVar.E();
                    }
                    cVar.D();
                    dVar.a("Lottie doesn't support layer effects. If you are using them for  fills, strokes, trim paths etc. then try adding them directly as contents  in your shape. Found: " + arrayList5);
                    str2 = str5;
                    j = j3;
                    j3 = j;
                    str5 = str2;
                    z = false;
                case 14:
                    f2 = (float) cVar.I();
                    str2 = str5;
                    str5 = str2;
                    z = false;
                case 15:
                    f4 = (float) cVar.I();
                    str2 = str5;
                    str5 = str2;
                    z = false;
                case 16:
                    i5 = (int) (c.a.a.c0.g.c() * cVar.J());
                    str2 = str5;
                    str5 = str2;
                    z = false;
                case 17:
                    i6 = (int) (c.a.a.c0.g.c() * cVar.J());
                    str2 = str5;
                    str5 = str2;
                    z = false;
                case 18:
                    f3 = (float) cVar.I();
                    str2 = str5;
                    str5 = str2;
                    z = false;
                case 19:
                    f5 = (float) cVar.I();
                    z = false;
                case 20:
                    bVar = b.v.u.c.t(cVar, dVar, z);
                    z = false;
                case 21:
                    str5 = cVar.L();
                    z = false;
                case 22:
                    z2 = cVar.H();
                    z = false;
                default:
                    str2 = str5;
                    j = j3;
                    cVar.P();
                    cVar.Q();
                    j3 = j;
                    str5 = str2;
                    z = false;
            }
        }
        String str8 = str5;
        long j5 = j3;
        cVar.E();
        float f6 = f3 / f2;
        float f7 = f5 / f2;
        ArrayList arrayList6 = new ArrayList();
        if (f6 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            arrayList = arrayList4;
            arrayList2 = arrayList3;
            str = str8;
            arrayList6.add(new c.a.a.d0.a(dVar, valueOf2, valueOf2, null, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Float.valueOf(f6)));
        } else {
            arrayList = arrayList4;
            arrayList2 = arrayList3;
            str = str8;
        }
        if (f7 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            f7 = dVar.l;
        }
        arrayList6.add(new c.a.a.d0.a(dVar, valueOf, valueOf, null, f6, Float.valueOf(f7)));
        arrayList6.add(new c.a.a.d0.a(dVar, valueOf2, valueOf2, null, f7, Float.valueOf(Float.MAX_VALUE)));
        if (str7.endsWith(".ai") || "ai".equals(str)) {
            dVar.a("Convert your Illustrator layers to shape layers.");
        }
        return new c.a.a.z.l.e(arrayList, dVar, str7, j5, aVar, j2, str6, arrayList2, lVar, i2, i3, i4, f2, f4, i5, i6, jVar, kVar, arrayList6, i7, bVar, z2);
    }
}