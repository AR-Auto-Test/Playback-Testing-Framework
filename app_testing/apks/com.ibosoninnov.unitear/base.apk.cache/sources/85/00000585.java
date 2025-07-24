package b.v;

import android.os.Bundle;
import b.v.q;

/* compiled from: NavGraphNavigator.java */
@q.b("navigation")
/* loaded from: classes.dex */
public class l extends q<k> {

    /* renamed from: a  reason: collision with root package name */
    public final r f2658a;

    public l(r rVar) {
        this.f2658a = rVar;
    }

    /* JADX DEBUG: Return type fixed from 'b.v.j' to match base method */
    @Override // b.v.q
    public k a() {
        return new k(this);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [b.v.j, android.os.Bundle, b.v.o, b.v.q$a] */
    @Override // b.v.q
    public j b(k kVar, Bundle bundle, o oVar, q.a aVar) {
        String str;
        k kVar2 = kVar;
        int i = kVar2.k;
        if (i == 0) {
            StringBuilder x = c.b.a.a.a.x("no start destination defined via app:startDestination for ");
            int i2 = kVar2.f2645d;
            if (i2 != 0) {
                if (kVar2.f2646e == null) {
                    kVar2.f2646e = Integer.toString(i2);
                }
                str = kVar2.f2646e;
            } else {
                str = "the root navigation";
            }
            x.append(str);
            throw new IllegalStateException(x.toString());
        }
        j g2 = kVar2.g(i, false);
        if (g2 == null) {
            if (kVar2.l == null) {
                kVar2.l = Integer.toString(kVar2.k);
            }
            throw new IllegalArgumentException(c.b.a.a.a.r("navigation destination ", kVar2.l, " is not a direct child of this NavGraph"));
        }
        return this.f2658a.c(g2.f2643b).b(g2, g2.a(bundle), oVar, aVar);
    }

    @Override // b.v.q
    public boolean e() {
        return true;
    }
}