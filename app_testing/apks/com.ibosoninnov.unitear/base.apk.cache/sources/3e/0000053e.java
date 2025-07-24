package b.q.b;

import android.util.Log;
import androidx.fragment.app.Fragment;
import java.util.HashMap;
import java.util.Iterator;

/* compiled from: FragmentManagerViewModel.java */
/* loaded from: classes.dex */
public final class u extends b.t.s {

    /* renamed from: c  reason: collision with root package name */
    public static final b.t.u f2523c = new a();

    /* renamed from: g  reason: collision with root package name */
    public final boolean f2527g;

    /* renamed from: d  reason: collision with root package name */
    public final HashMap<String, Fragment> f2524d = new HashMap<>();

    /* renamed from: e  reason: collision with root package name */
    public final HashMap<String, u> f2525e = new HashMap<>();

    /* renamed from: f  reason: collision with root package name */
    public final HashMap<String, b.t.y> f2526f = new HashMap<>();

    /* renamed from: h  reason: collision with root package name */
    public boolean f2528h = false;

    /* compiled from: FragmentManagerViewModel.java */
    /* loaded from: classes.dex */
    public static class a implements b.t.u {
        @Override // b.t.u
        public <T extends b.t.s> T a(Class<T> cls) {
            return new u(true);
        }
    }

    public u(boolean z) {
        this.f2527g = z;
    }

    @Override // b.t.s
    public void a() {
        if (q.N(3)) {
            Log.d("FragmentManager", "onCleared called for " + this);
        }
        this.f2528h = true;
    }

    public boolean c(Fragment fragment) {
        if (this.f2524d.containsKey(fragment.mWho) && this.f2527g) {
            return this.f2528h;
        }
        return true;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || u.class != obj.getClass()) {
            return false;
        }
        u uVar = (u) obj;
        return this.f2524d.equals(uVar.f2524d) && this.f2525e.equals(uVar.f2525e) && this.f2526f.equals(uVar.f2526f);
    }

    public int hashCode() {
        int hashCode = this.f2525e.hashCode();
        return this.f2526f.hashCode() + ((hashCode + (this.f2524d.hashCode() * 31)) * 31);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("FragmentManagerViewModel{");
        sb.append(Integer.toHexString(System.identityHashCode(this)));
        sb.append("} Fragments (");
        Iterator<Fragment> it = this.f2524d.values().iterator();
        while (it.hasNext()) {
            sb.append(it.next());
            if (it.hasNext()) {
                sb.append(", ");
            }
        }
        sb.append(") Child Non Config (");
        Iterator<String> it2 = this.f2525e.keySet().iterator();
        while (it2.hasNext()) {
            sb.append(it2.next());
            if (it2.hasNext()) {
                sb.append(", ");
            }
        }
        sb.append(") ViewModelStores (");
        Iterator<String> it3 = this.f2526f.keySet().iterator();
        while (it3.hasNext()) {
            sb.append(it3.next());
            if (it3.hasNext()) {
                sb.append(", ");
            }
        }
        sb.append(')');
        return sb.toString();
    }
}