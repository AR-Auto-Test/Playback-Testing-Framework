package b.h.b.i.l;

import java.util.ArrayList;

/* compiled from: RunGroup.java */
/* loaded from: classes.dex */
public class l {

    /* renamed from: a  reason: collision with root package name */
    public static int f1919a;

    /* renamed from: b  reason: collision with root package name */
    public o f1920b;

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<o> f1921c = new ArrayList<>();

    public l(o oVar, int i) {
        this.f1920b = null;
        f1919a++;
        this.f1920b = oVar;
    }

    public final long a(f fVar, long j) {
        o oVar = fVar.f1906d;
        if (oVar instanceof j) {
            return j;
        }
        int size = fVar.k.size();
        long j2 = j;
        for (int i = 0; i < size; i++) {
            d dVar = fVar.k.get(i);
            if (dVar instanceof f) {
                f fVar2 = (f) dVar;
                if (fVar2.f1906d != oVar) {
                    j2 = Math.min(j2, a(fVar2, fVar2.f1908f + j));
                }
            }
        }
        if (fVar == oVar.i) {
            long j3 = j - oVar.j();
            return Math.min(Math.min(j2, a(oVar.f1935h, j3)), j3 - oVar.f1935h.f1908f);
        }
        return j2;
    }

    public final long b(f fVar, long j) {
        o oVar = fVar.f1906d;
        if (oVar instanceof j) {
            return j;
        }
        int size = fVar.k.size();
        long j2 = j;
        for (int i = 0; i < size; i++) {
            d dVar = fVar.k.get(i);
            if (dVar instanceof f) {
                f fVar2 = (f) dVar;
                if (fVar2.f1906d != oVar) {
                    j2 = Math.max(j2, b(fVar2, fVar2.f1908f + j));
                }
            }
        }
        if (fVar == oVar.f1935h) {
            long j3 = j + oVar.j();
            return Math.max(Math.max(j2, b(oVar.i, j3)), j3 - oVar.i.f1908f);
        }
        return j2;
    }
}