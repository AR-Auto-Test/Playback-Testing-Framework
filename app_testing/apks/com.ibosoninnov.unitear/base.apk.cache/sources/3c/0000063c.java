package c.a.a.b0;

import android.graphics.Color;
import android.graphics.PointF;
import c.a.a.b0.h0.c;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: JsonUtils.java */
/* loaded from: classes.dex */
public class o {

    /* renamed from: a  reason: collision with root package name */
    public static final c.a f2999a = c.a.a("x", "y");

    public static int a(c.a.a.b0.h0.c cVar) {
        cVar.B();
        int I = (int) (cVar.I() * 255.0d);
        int I2 = (int) (cVar.I() * 255.0d);
        int I3 = (int) (cVar.I() * 255.0d);
        while (cVar.G()) {
            cVar.Q();
        }
        cVar.D();
        return Color.argb(255, I, I2, I3);
    }

    public static PointF b(c.a.a.b0.h0.c cVar, float f2) {
        int ordinal = cVar.M().ordinal();
        if (ordinal == 0) {
            cVar.B();
            float I = (float) cVar.I();
            float I2 = (float) cVar.I();
            while (cVar.M() != c.b.END_ARRAY) {
                cVar.Q();
            }
            cVar.D();
            return new PointF(I * f2, I2 * f2);
        } else if (ordinal != 2) {
            if (ordinal == 6) {
                float I3 = (float) cVar.I();
                float I4 = (float) cVar.I();
                while (cVar.G()) {
                    cVar.Q();
                }
                return new PointF(I3 * f2, I4 * f2);
            }
            StringBuilder x = c.b.a.a.a.x("Unknown point starts with ");
            x.append(cVar.M());
            throw new IllegalArgumentException(x.toString());
        } else {
            cVar.C();
            float f3 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            float f4 = 0.0f;
            while (cVar.G()) {
                int O = cVar.O(f2999a);
                if (O == 0) {
                    f3 = d(cVar);
                } else if (O != 1) {
                    cVar.P();
                    cVar.Q();
                } else {
                    f4 = d(cVar);
                }
            }
            cVar.E();
            return new PointF(f3 * f2, f4 * f2);
        }
    }

    public static List<PointF> c(c.a.a.b0.h0.c cVar, float f2) {
        ArrayList arrayList = new ArrayList();
        cVar.B();
        while (cVar.M() == c.b.BEGIN_ARRAY) {
            cVar.B();
            arrayList.add(b(cVar, f2));
            cVar.D();
        }
        cVar.D();
        return arrayList;
    }

    public static float d(c.a.a.b0.h0.c cVar) {
        c.b M = cVar.M();
        int ordinal = M.ordinal();
        if (ordinal != 0) {
            if (ordinal == 6) {
                return (float) cVar.I();
            }
            throw new IllegalArgumentException("Unknown value for token of type " + M);
        }
        cVar.B();
        float I = (float) cVar.I();
        while (cVar.G()) {
            cVar.Q();
        }
        cVar.D();
        return I;
    }
}