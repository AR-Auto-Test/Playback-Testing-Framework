package b.d.b.d1;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/* compiled from: Quirks.java */
/* loaded from: classes.dex */
public class z0 {

    /* renamed from: a  reason: collision with root package name */
    public final List<y0> f1590a;

    public z0(List<y0> list) {
        this.f1590a = new ArrayList(list);
    }

    public <T extends y0> T a(Class<T> cls) {
        Iterator<y0> it = this.f1590a.iterator();
        while (it.hasNext()) {
            T t = (T) it.next();
            if (t.getClass() == cls) {
                return t;
            }
        }
        return null;
    }
}