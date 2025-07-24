package c.a.a.c0;

import android.graphics.PointF;
import c.a.a.x.b.k;
import java.util.List;

/* compiled from: MiscUtils.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public static PointF f3030a = new PointF();

    public static PointF a(PointF pointF, PointF pointF2) {
        return new PointF(pointF.x + pointF2.x, pointF.y + pointF2.y);
    }

    public static float b(float f2, float f3, float f4) {
        return Math.max(f3, Math.min(f4, f2));
    }

    public static int c(int i, int i2, int i3) {
        return Math.max(i2, Math.min(i3, i));
    }

    public static int d(float f2, float f3) {
        int i = (int) f2;
        int i2 = (int) f3;
        int i3 = i / i2;
        int i4 = i % i2;
        if (!((i ^ i2) >= 0) && i4 != 0) {
            i3--;
        }
        return i - (i2 * i3);
    }

    public static float e(float f2, float f3, float f4) {
        return c.b.a.a.a.a(f3, f2, f4, f2);
    }

    public static void f(c.a.a.z.e eVar, int i, List<c.a.a.z.e> list, c.a.a.z.e eVar2, k kVar) {
        if (eVar.c(kVar.getName(), i)) {
            list.add(eVar2.a(kVar.getName()).g(kVar));
        }
    }
}