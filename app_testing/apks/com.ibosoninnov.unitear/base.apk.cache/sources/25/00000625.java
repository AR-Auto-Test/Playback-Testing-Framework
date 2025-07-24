package c.a.a.b0;

import android.graphics.PointF;
import c.a.a.b0.h0.c;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: AnimatableTransformParser.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public static c.a f2960a = c.a.a("a", "p", "s", "rz", "r", "o", "so", "eo", "sk", "sa");

    /* renamed from: b  reason: collision with root package name */
    public static c.a f2961b = c.a.a("k");

    /* JADX WARN: Code restructure failed: missing block: B:106:0x020e, code lost:
        if (((java.lang.Float) ((c.a.a.d0.a) r12.f3301a.get(0)).f3046b).floatValue() == com.google.android.material.internal.StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) goto L125;
     */
    /* JADX WARN: Code restructure failed: missing block: B:85:0x01c0, code lost:
        if ((r0.f3057a == 1.0f && r0.f3058b == 1.0f) != false) goto L130;
     */
    /* JADX WARN: Removed duplicated region for block: B:101:0x01f0  */
    /* JADX WARN: Removed duplicated region for block: B:103:0x01f3  */
    /* JADX WARN: Removed duplicated region for block: B:110:0x0214  */
    /* JADX WARN: Removed duplicated region for block: B:113:0x0219  */
    /* JADX WARN: Removed duplicated region for block: B:114:0x021c  */
    /* JADX WARN: Removed duplicated region for block: B:22:0x0068  */
    /* JADX WARN: Removed duplicated region for block: B:23:0x009b  */
    /* JADX WARN: Removed duplicated region for block: B:90:0x01c8  */
    /* JADX WARN: Removed duplicated region for block: B:91:0x01cb  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static c.a.a.z.j.l a(c.a.a.b0.h0.c cVar, c.a.a.d dVar) {
        c.a.a.z.j.b bVar;
        c.a.a.z.j.g gVar;
        boolean z;
        c.a.a.z.j.b bVar2;
        boolean z2;
        c.a.a.z.j.g gVar2;
        c.a.a.z.j.b bVar3;
        c.a.a.z.j.m<PointF, PointF> mVar;
        c.a.a.z.j.b t;
        c.a.a.z.j.b bVar4;
        boolean z3 = false;
        boolean z4 = cVar.M() == c.b.BEGIN_OBJECT;
        if (z4) {
            cVar.C();
        }
        c.a.a.z.j.b bVar5 = null;
        c.a.a.z.j.g gVar3 = null;
        c.a.a.z.j.m<PointF, PointF> mVar2 = null;
        c.a.a.z.j.b bVar6 = null;
        c.a.a.z.j.b bVar7 = null;
        c.a.a.z.j.e eVar = null;
        c.a.a.z.j.d dVar2 = null;
        c.a.a.z.j.b bVar8 = null;
        c.a.a.z.j.b bVar9 = null;
        while (cVar.G()) {
            switch (cVar.O(f2960a)) {
                case 0:
                    gVar2 = gVar3;
                    bVar3 = bVar6;
                    mVar = mVar2;
                    cVar.C();
                    while (cVar.G()) {
                        if (cVar.O(f2961b) != 0) {
                            cVar.P();
                            cVar.Q();
                        } else {
                            eVar = a.a(cVar, dVar);
                        }
                    }
                    cVar.E();
                    gVar3 = gVar2;
                    mVar2 = mVar;
                    break;
                case 1:
                    bVar3 = bVar6;
                    mVar2 = a.b(cVar, dVar);
                    break;
                case 2:
                    bVar3 = bVar6;
                    mVar = mVar2;
                    gVar3 = new c.a.a.z.j.g(b.v.u.c.q(cVar, dVar, z.f3017a));
                    mVar2 = mVar;
                    break;
                case 3:
                    dVar.a("Lottie doesn't support 3D layers.");
                    t = b.v.u.c.t(cVar, dVar, z3);
                    if (!t.f3301a.isEmpty()) {
                        bVar4 = t;
                        gVar2 = gVar3;
                        bVar3 = bVar6;
                        mVar = mVar2;
                        t.f3301a.add(new c.a.a.d0.a(dVar, Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), null, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Float.valueOf(dVar.l)));
                    } else {
                        bVar4 = t;
                        gVar2 = gVar3;
                        bVar3 = bVar6;
                        mVar = mVar2;
                        if (((c.a.a.d0.a) bVar4.f3301a.get(0)).f3046b == 0) {
                            bVar4.f3301a.set(0, new c.a.a.d0.a(dVar, Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), Float.valueOf((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD), null, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Float.valueOf(dVar.l)));
                        }
                    }
                    bVar5 = bVar4;
                    gVar3 = gVar2;
                    mVar2 = mVar;
                    break;
                case 4:
                    t = b.v.u.c.t(cVar, dVar, z3);
                    if (!t.f3301a.isEmpty()) {
                    }
                    bVar5 = bVar4;
                    gVar3 = gVar2;
                    mVar2 = mVar;
                    break;
                case 5:
                    dVar2 = b.v.u.c.u(cVar, dVar);
                    continue;
                case 6:
                    bVar8 = b.v.u.c.t(cVar, dVar, z3);
                    continue;
                case 7:
                    bVar9 = b.v.u.c.t(cVar, dVar, z3);
                    continue;
                case 8:
                    bVar7 = b.v.u.c.t(cVar, dVar, z3);
                    continue;
                case 9:
                    bVar6 = b.v.u.c.t(cVar, dVar, z3);
                    continue;
                default:
                    bVar3 = bVar6;
                    cVar.P();
                    cVar.Q();
                    break;
            }
            bVar6 = bVar3;
            z3 = false;
        }
        c.a.a.z.j.g gVar4 = gVar3;
        c.a.a.z.j.b bVar10 = bVar6;
        c.a.a.z.j.m<PointF, PointF> mVar3 = mVar2;
        if (z4) {
            cVar.E();
        }
        if (eVar == null || (eVar.c() && eVar.f3286a.get(0).f3046b.equals(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
            eVar = null;
        }
        if (mVar3 == null || (!(mVar3 instanceof c.a.a.z.j.i) && mVar3.c() && mVar3.b().get(0).f3046b.equals(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD))) {
            mVar3 = null;
        }
        if (bVar5 == null || (bVar5.c() && ((Float) ((c.a.a.d0.a) bVar5.f3301a.get(0)).f3046b).floatValue() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
            gVar = gVar4;
            bVar = null;
        } else {
            bVar = bVar5;
            gVar = gVar4;
        }
        if (gVar != null) {
            if (gVar.c()) {
                c.a.a.d0.d dVar3 = (c.a.a.d0.d) ((c.a.a.d0.a) gVar.f3301a.get(0)).f3046b;
            }
            z = false;
            c.a.a.z.j.g gVar5 = !z ? null : gVar;
            if (bVar7 != null || (bVar7.c() && ((Float) ((c.a.a.d0.a) bVar7.f3301a.get(0)).f3046b).floatValue() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
                bVar7 = null;
            }
            if (bVar10 == null) {
                bVar2 = bVar10;
                boolean z5 = bVar10.c() ? false : false;
                z2 = z5;
                return new c.a.a.z.j.l(eVar, mVar3, gVar5, bVar, dVar2, bVar8, bVar9, bVar7, z2 ? null : bVar2);
            }
            bVar2 = bVar10;
            z2 = true;
            return new c.a.a.z.j.l(eVar, mVar3, gVar5, bVar, dVar2, bVar8, bVar9, bVar7, z2 ? null : bVar2);
        }
        z = true;
        if (!z) {
        }
        if (bVar7 != null || (bVar7.c() && ((Float) ((c.a.a.d0.a) bVar7.f3301a.get(0)).f3046b).floatValue() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) {
        }
        if (bVar10 == null) {
        }
        z2 = true;
        return new c.a.a.z.j.l(eVar, mVar3, gVar5, bVar, dVar2, bVar8, bVar9, bVar7, z2 ? null : bVar2);
    }
}