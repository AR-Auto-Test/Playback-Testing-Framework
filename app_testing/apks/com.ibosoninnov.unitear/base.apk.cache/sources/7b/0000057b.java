package b.v;

import b.t.u;
import b.t.v;
import b.t.x;
import b.t.y;
import java.util.HashMap;
import java.util.Iterator;
import java.util.UUID;

/* compiled from: NavControllerViewModel.java */
/* loaded from: classes.dex */
public class g extends b.t.s {

    /* renamed from: c  reason: collision with root package name */
    public static final u f2626c = new a();

    /* renamed from: d  reason: collision with root package name */
    public final HashMap<UUID, y> f2627d = new HashMap<>();

    /* compiled from: NavControllerViewModel.java */
    /* loaded from: classes.dex */
    public class a implements u {
        @Override // b.t.u
        public <T extends b.t.s> T a(Class<T> cls) {
            return new g();
        }
    }

    public static g c(y yVar) {
        b.t.s a2;
        u uVar = f2626c;
        String canonicalName = g.class.getCanonicalName();
        if (canonicalName != null) {
            String q = c.b.a.a.a.q("androidx.lifecycle.ViewModelProvider.DefaultKey:", canonicalName);
            b.t.s sVar = yVar.f2604a.get(q);
            if (g.class.isInstance(sVar)) {
                if (uVar instanceof x) {
                    ((x) uVar).b(sVar);
                }
            } else {
                if (uVar instanceof v) {
                    a2 = ((v) uVar).c(q, g.class);
                } else {
                    a2 = uVar.a(g.class);
                }
                sVar = a2;
                b.t.s put = yVar.f2604a.put(q, sVar);
                if (put != null) {
                    put.a();
                }
            }
            return (g) sVar;
        }
        throw new IllegalArgumentException("Local and anonymous classes can not be ViewModels");
    }

    @Override // b.t.s
    public void a() {
        for (y yVar : this.f2627d.values()) {
            yVar.a();
        }
        this.f2627d.clear();
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("NavControllerViewModel{");
        sb.append(Integer.toHexString(System.identityHashCode(this)));
        sb.append("} ViewModelStores (");
        Iterator<UUID> it = this.f2627d.keySet().iterator();
        while (it.hasNext()) {
            sb.append(it.next());
            if (it.hasNext()) {
                sb.append(", ");
            }
        }
        sb.append(')');
        return sb.toString();
    }
}