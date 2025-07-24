package androidx.lifecycle;

import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.t.q;
import b.t.s;
import b.t.y;
import b.t.z;
import b.x.a;
import b.x.c;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

/* loaded from: classes.dex */
public final class SavedStateHandleController implements f {

    /* renamed from: a  reason: collision with root package name */
    public final String f328a;

    /* renamed from: b  reason: collision with root package name */
    public boolean f329b = false;

    /* renamed from: c  reason: collision with root package name */
    public final q f330c;

    /* loaded from: classes.dex */
    public static final class a implements a.InterfaceC0055a {
        @Override // b.x.a.InterfaceC0055a
        public void a(c cVar) {
            if (cVar instanceof z) {
                y viewModelStore = ((z) cVar).getViewModelStore();
                b.x.a savedStateRegistry = cVar.getSavedStateRegistry();
                Objects.requireNonNull(viewModelStore);
                Iterator it = new HashSet(viewModelStore.f2604a.keySet()).iterator();
                while (it.hasNext()) {
                    SavedStateHandleController.a(viewModelStore.f2604a.get((String) it.next()), savedStateRegistry, cVar.getLifecycle());
                }
                if (new HashSet(viewModelStore.f2604a.keySet()).isEmpty()) {
                    return;
                }
                savedStateRegistry.b(a.class);
                return;
            }
            throw new IllegalStateException("Internal error: OnRecreation should be registered only on componentsthat implement ViewModelStoreOwner");
        }
    }

    public SavedStateHandleController(String str, q qVar) {
        this.f328a = str;
        this.f330c = qVar;
    }

    public static void a(s sVar, b.x.a aVar, e eVar) {
        Object obj;
        Map<String, Object> map = sVar.f2600a;
        if (map == null) {
            obj = null;
        } else {
            synchronized (map) {
                obj = sVar.f2600a.get("androidx.lifecycle.savedstate.vm.tag");
            }
        }
        SavedStateHandleController savedStateHandleController = (SavedStateHandleController) obj;
        if (savedStateHandleController == null || savedStateHandleController.f329b) {
            return;
        }
        savedStateHandleController.b(aVar, eVar);
        g(aVar, eVar);
    }

    public static void g(final b.x.a aVar, final e eVar) {
        e.b bVar = ((i) eVar).f2579b;
        if (bVar != e.b.INITIALIZED) {
            if (!(bVar.compareTo(e.b.STARTED) >= 0)) {
                eVar.a(new f() { // from class: androidx.lifecycle.SavedStateHandleController.1
                    @Override // b.t.f
                    public void e(h hVar, e.a aVar2) {
                        if (aVar2 == e.a.ON_START) {
                            ((i) e.this).f2578a.e(this);
                            aVar.b(a.class);
                        }
                    }
                });
                return;
            }
        }
        aVar.b(a.class);
    }

    public void b(b.x.a aVar, e eVar) {
        if (!this.f329b) {
            this.f329b = true;
            eVar.a(this);
            if (aVar.f2820a.d(this.f328a, this.f330c.f2591c) != null) {
                throw new IllegalArgumentException("SavedStateProvider with the given key is already registered");
            }
            return;
        }
        throw new IllegalStateException("Already attached to lifecycleOwner");
    }

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        if (aVar == e.a.ON_DESTROY) {
            this.f329b = false;
            ((i) hVar.getLifecycle()).f2578a.e(this);
        }
    }
}