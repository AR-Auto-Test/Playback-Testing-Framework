package c.a.a.z.k;

import java.util.Arrays;
import java.util.List;

/* compiled from: ShapeGroup.java */
/* loaded from: classes.dex */
public class m implements b {

    /* renamed from: a  reason: collision with root package name */
    public final String f3365a;

    /* renamed from: b  reason: collision with root package name */
    public final List<b> f3366b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f3367c;

    public m(String str, List<b> list, boolean z) {
        this.f3365a = str;
        this.f3366b = list;
        this.f3367c = z;
    }

    @Override // c.a.a.z.k.b
    public c.a.a.x.b.c a(c.a.a.j jVar, c.a.a.z.l.b bVar) {
        return new c.a.a.x.b.d(jVar, bVar, this);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("ShapeGroup{name='");
        x.append(this.f3365a);
        x.append("' Shapes: ");
        x.append(Arrays.toString(this.f3366b.toArray()));
        x.append('}');
        return x.toString();
    }
}