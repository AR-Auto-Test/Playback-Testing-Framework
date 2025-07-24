package c.a.a.b0;

import android.graphics.PointF;
import c.a.a.b0.h0.c;

/* compiled from: PointFParser.java */
/* loaded from: classes.dex */
public class v implements g0<PointF> {

    /* renamed from: a  reason: collision with root package name */
    public static final v f3013a = new v();

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.a.a.b0.g0
    public PointF a(c.a.a.b0.h0.c cVar, float f2) {
        c.b M = cVar.M();
        if (M == c.b.BEGIN_ARRAY) {
            return o.b(cVar, f2);
        }
        if (M == c.b.BEGIN_OBJECT) {
            return o.b(cVar, f2);
        }
        if (M == c.b.NUMBER) {
            PointF pointF = new PointF(((float) cVar.I()) * f2, ((float) cVar.I()) * f2);
            while (cVar.G()) {
                cVar.Q();
            }
            return pointF;
        }
        throw new IllegalArgumentException("Cannot convert json to point. Next token is " + M);
    }
}