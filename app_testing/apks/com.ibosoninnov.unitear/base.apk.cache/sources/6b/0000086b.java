package c.c.a.n;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.WeakHashMap;

/* compiled from: RequestTracker.java */
/* loaded from: classes.dex */
public class r {

    /* renamed from: a  reason: collision with root package name */
    public final Set<c.c.a.q.c> f4098a = Collections.newSetFromMap(new WeakHashMap());

    /* renamed from: b  reason: collision with root package name */
    public final List<c.c.a.q.c> f4099b = new ArrayList();

    /* renamed from: c  reason: collision with root package name */
    public boolean f4100c;

    public boolean a(c.c.a.q.c cVar) {
        boolean z = true;
        if (cVar == null) {
            return true;
        }
        boolean remove = this.f4098a.remove(cVar);
        if (!this.f4099b.remove(cVar) && !remove) {
            z = false;
        }
        if (z) {
            cVar.clear();
        }
        return z;
    }

    public String toString() {
        return super.toString() + "{numRequests=" + this.f4098a.size() + ", isPaused=" + this.f4100c + "}";
    }
}