package b.q.b;

import android.view.View;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FragmentTransitionImpl.java */
/* loaded from: classes.dex */
public class i0 implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2479b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ Map f2480c;

    public i0(k0 k0Var, ArrayList arrayList, Map map) {
        this.f2479b = arrayList;
        this.f2480c = map;
    }

    @Override // java.lang.Runnable
    public void run() {
        String str;
        int size = this.f2479b.size();
        for (int i = 0; i < size; i++) {
            View view = (View) this.f2479b.get(i);
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            String transitionName = view.getTransitionName();
            if (transitionName != null) {
                Iterator it = this.f2480c.entrySet().iterator();
                while (true) {
                    if (!it.hasNext()) {
                        str = null;
                        break;
                    }
                    Map.Entry entry = (Map.Entry) it.next();
                    if (transitionName.equals(entry.getValue())) {
                        str = (String) entry.getKey();
                        break;
                    }
                }
                view.setTransitionName(str);
            }
        }
    }
}