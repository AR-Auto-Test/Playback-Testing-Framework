package b.h.b.i;

/* compiled from: Optimizer.java */
/* loaded from: classes.dex */
public class i {

    /* renamed from: a  reason: collision with root package name */
    public static boolean[] f1883a = new boolean[3];

    public static void a(e eVar, b.h.b.d dVar, d dVar2) {
        dVar2.j = -1;
        dVar2.k = -1;
        if (eVar.O[0] != 2 && dVar2.O[0] == 4) {
            int i = dVar2.D.f1866g;
            int r = eVar.r() - dVar2.F.f1866g;
            c cVar = dVar2.D;
            cVar.i = dVar.l(cVar);
            c cVar2 = dVar2.F;
            cVar2.i = dVar.l(cVar2);
            dVar.e(dVar2.D.i, i);
            dVar.e(dVar2.F.i, r);
            dVar2.j = 2;
            dVar2.U = i;
            int i2 = r - i;
            dVar2.Q = i2;
            int i3 = dVar2.X;
            if (i2 < i3) {
                dVar2.Q = i3;
            }
        }
        if (eVar.O[1] == 2 || dVar2.O[1] != 4) {
            return;
        }
        int i4 = dVar2.E.f1866g;
        int l = eVar.l() - dVar2.G.f1866g;
        c cVar3 = dVar2.E;
        cVar3.i = dVar.l(cVar3);
        c cVar4 = dVar2.G;
        cVar4.i = dVar.l(cVar4);
        dVar.e(dVar2.E.i, i4);
        dVar.e(dVar2.G.i, l);
        if (dVar2.W > 0 || dVar2.c0 == 8) {
            c cVar5 = dVar2.H;
            cVar5.i = dVar.l(cVar5);
            dVar.e(dVar2.H.i, dVar2.W + i4);
        }
        dVar2.k = 2;
        dVar2.V = i4;
        int i5 = l - i4;
        dVar2.R = i5;
        int i6 = dVar2.Y;
        if (i5 < i6) {
            dVar2.R = i6;
        }
    }

    public static final boolean b(int i, int i2) {
        return (i & i2) == i2;
    }
}