package c.a.a.z.k;

import android.graphics.PointF;
import java.util.ArrayList;
import java.util.List;

/* compiled from: ShapeData.java */
/* loaded from: classes.dex */
public class k {

    /* renamed from: a  reason: collision with root package name */
    public final List<c.a.a.z.a> f3356a;

    /* renamed from: b  reason: collision with root package name */
    public PointF f3357b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f3358c;

    public k(PointF pointF, boolean z, List<c.a.a.z.a> list) {
        this.f3357b = pointF;
        this.f3358c = z;
        this.f3356a = new ArrayList(list);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ShapeData{numCurves=");
        x.append(this.f3356a.size());
        x.append("closed=");
        x.append(this.f3358c);
        x.append('}');
        return x.toString();
    }

    public k() {
        this.f3356a = new ArrayList();
    }
}