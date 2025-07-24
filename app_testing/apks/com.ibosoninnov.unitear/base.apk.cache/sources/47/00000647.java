package c.a.a.b0;

import c.a.a.b0.h0.c;

/* compiled from: ScaleXYParser.java */
/* loaded from: classes.dex */
public class z implements g0<c.a.a.d0.d> {

    /* renamed from: a  reason: collision with root package name */
    public static final z f3017a = new z();

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public c.a.a.d0.d a(c.a.a.b0.h0.c cVar, float f2) {
        boolean z = cVar.M() == c.b.BEGIN_ARRAY;
        if (z) {
            cVar.B();
        }
        float I = (float) cVar.I();
        float I2 = (float) cVar.I();
        while (cVar.G()) {
            cVar.Q();
        }
        if (z) {
            cVar.D();
        }
        return new c.a.a.d0.d((I / 100.0f) * f2, (I2 / 100.0f) * f2);
    }
}