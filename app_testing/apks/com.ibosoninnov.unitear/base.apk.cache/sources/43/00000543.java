package b.q.b;

import androidx.fragment.app.Fragment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

/* compiled from: FragmentStore.java */
/* loaded from: classes.dex */
public class x {

    /* renamed from: a  reason: collision with root package name */
    public final ArrayList<Fragment> f2539a = new ArrayList<>();

    /* renamed from: b  reason: collision with root package name */
    public final HashMap<String, w> f2540b = new HashMap<>();

    public void a(Fragment fragment) {
        if (!this.f2539a.contains(fragment)) {
            synchronized (this.f2539a) {
                this.f2539a.add(fragment);
            }
            fragment.mAdded = true;
            return;
        }
        throw new IllegalStateException("Fragment already added: " + fragment);
    }

    public void b() {
        this.f2540b.values().removeAll(Collections.singleton(null));
    }

    public boolean c(String str) {
        return this.f2540b.containsKey(str);
    }

    public void d(int i) {
        Iterator<Fragment> it = this.f2539a.iterator();
        while (it.hasNext()) {
            w wVar = this.f2540b.get(it.next().mWho);
            if (wVar != null) {
                wVar.f2538c = i;
            }
        }
        for (w wVar2 : this.f2540b.values()) {
            if (wVar2 != null) {
                wVar2.f2538c = i;
            }
        }
    }

    public Fragment e(String str) {
        w wVar = this.f2540b.get(str);
        if (wVar != null) {
            return wVar.f2537b;
        }
        return null;
    }

    public List<Fragment> f() {
        ArrayList arrayList = new ArrayList();
        for (w wVar : this.f2540b.values()) {
            if (wVar != null) {
                arrayList.add(wVar.f2537b);
            } else {
                arrayList.add(null);
            }
        }
        return arrayList;
    }

    public List<Fragment> g() {
        ArrayList arrayList;
        if (this.f2539a.isEmpty()) {
            return Collections.emptyList();
        }
        synchronized (this.f2539a) {
            arrayList = new ArrayList(this.f2539a);
        }
        return arrayList;
    }

    public void h(Fragment fragment) {
        synchronized (this.f2539a) {
            this.f2539a.remove(fragment);
        }
        fragment.mAdded = false;
    }
}