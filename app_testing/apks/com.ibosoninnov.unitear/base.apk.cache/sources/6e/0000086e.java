package c.c.a.n;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;
import java.util.WeakHashMap;

/* compiled from: TargetTracker.java */
/* loaded from: classes.dex */
public final class t implements m {

    /* renamed from: b  reason: collision with root package name */
    public final Set<c.c.a.q.j.h<?>> f4108b = Collections.newSetFromMap(new WeakHashMap());

    @Override // c.c.a.n.m
    public void onDestroy() {
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4108b)).iterator();
        while (it.hasNext()) {
            ((c.c.a.q.j.h) it.next()).onDestroy();
        }
    }

    @Override // c.c.a.n.m
    public void onStart() {
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4108b)).iterator();
        while (it.hasNext()) {
            ((c.c.a.q.j.h) it.next()).onStart();
        }
    }

    @Override // c.c.a.n.m
    public void onStop() {
        Iterator it = ((ArrayList) c.c.a.s.j.e(this.f4108b)).iterator();
        while (it.hasNext()) {
            ((c.c.a.q.j.h) it.next()).onStop();
        }
    }
}