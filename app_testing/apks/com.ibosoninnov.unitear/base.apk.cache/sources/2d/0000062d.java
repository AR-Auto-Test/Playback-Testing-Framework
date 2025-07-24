package c.a.a.b0;

import b.d.b.m0;
import c.a.a.b0.h0.c;

/* compiled from: DocumentDataParser.java */
/* loaded from: classes.dex */
public class g implements g0<c.a.a.z.b> {

    /* renamed from: a  reason: collision with root package name */
    public static final g f2970a = new g();

    /* renamed from: b  reason: collision with root package name */
    public static final c.a f2971b = c.a.a("t", "f", "s", "j", "tr", "lh", "ls", "fc", "sc", "sw", "of");

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public c.a.a.z.b a(c.a.a.b0.h0.c cVar, float f2) {
        cVar.C();
        int i = 3;
        String str = null;
        String str2 = null;
        int i2 = 0;
        int i3 = 0;
        int i4 = 0;
        float f3 = 0.0f;
        float f4 = 0.0f;
        float f5 = 0.0f;
        float f6 = 0.0f;
        boolean z = true;
        while (cVar.G()) {
            switch (cVar.O(f2971b)) {
                case 0:
                    str = cVar.L();
                    break;
                case 1:
                    str2 = cVar.L();
                    break;
                case 2:
                    f3 = (float) cVar.I();
                    break;
                case 3:
                    int J = cVar.J();
                    if (J <= 2 && J >= 0) {
                        i = m0.com$airbnb$lottie$model$DocumentData$Justification$s$values()[J];
                        break;
                    } else {
                        i = 3;
                        break;
                    }
                case 4:
                    i2 = cVar.J();
                    break;
                case 5:
                    f4 = (float) cVar.I();
                    break;
                case 6:
                    f5 = (float) cVar.I();
                    break;
                case 7:
                    i3 = o.a(cVar);
                    break;
                case 8:
                    i4 = o.a(cVar);
                    break;
                case 9:
                    f6 = (float) cVar.I();
                    break;
                case 10:
                    z = cVar.H();
                    break;
                default:
                    cVar.P();
                    cVar.Q();
                    break;
            }
        }
        cVar.E();
        return new c.a.a.z.b(str, str2, f3, i, i2, f4, f5, i3, i4, f6, z);
    }
}