package b.q.b;

import android.view.View;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentTransitionImpl.java */
/* loaded from: classes.dex */
public class j0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2481b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Map f2482c;

    public j0(k0 k0Var, ArrayList arrayList, Map map) {
        this.f2481b = arrayList;
        this.f2482c = map;
    }

    @Override // java.lang.Runnable
    public void run() {
        int size = this.f2481b.size();
        for (int i = 0; i < size; i++) {
            View view = (View) this.f2481b.get(i);
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            view.setTransitionName((String) this.f2482c.get(view.getTransitionName()));
        }
    }
}