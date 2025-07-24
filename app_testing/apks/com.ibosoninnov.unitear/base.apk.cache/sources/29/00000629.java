package c.a.a.b0;

import android.graphics.Color;
import c.a.a.b0.h0.c;

/* compiled from: ColorParser.java */
/* loaded from: classes.dex */
public class e implements g0<Integer> {

    /* renamed from: a  reason: collision with root package name */
    public static final e f2965a = new e();

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public Integer a(c.a.a.b0.h0.c cVar, float f2) {
        boolean z = cVar.M() == c.b.BEGIN_ARRAY;
        if (z) {
            cVar.B();
        }
        double I = cVar.I();
        double I2 = cVar.I();
        double I3 = cVar.I();
        double I4 = cVar.I();
        if (z) {
            cVar.D();
        }
        if (I <= 1.0d && I2 <= 1.0d && I3 <= 1.0d) {
            I *= 255.0d;
            I2 *= 255.0d;
            I3 *= 255.0d;
            if (I4 <= 1.0d) {
                I4 *= 255.0d;
            }
        }
        return Integer.valueOf(Color.argb((int) I4, (int) I, (int) I2, (int) I3));
    }
}