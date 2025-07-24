package androidx.savedstate;

import android.annotation.SuppressLint;
import android.os.Bundle;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.x.a;
import b.x.c;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

@SuppressLint({"RestrictedApi"})
/* loaded from: classes.dex */
public final class Recreator implements f {

    /* renamed from: a  reason: collision with root package name */
    public final c f487a;

    /* loaded from: classes.dex */
    public static final class a implements a.b {

        /* renamed from: a  reason: collision with root package name */
        public final Set<String> f488a = new HashSet();

        public a(b.x.a aVar) {
            if (aVar.f2820a.d("androidx.savedstate.Restarter", this) != null) {
                throw new IllegalArgumentException("SavedStateProvider with the given key is already registered");
            }
        }

        @Override // b.x.a.b
        public Bundle a() {
            Bundle bundle = new Bundle();
            bundle.putStringArrayList("classes_to_restore", new ArrayList<>(this.f488a));
            return bundle;
        }
    }

    public Recreator(c cVar) {
        this.f487a = cVar;
    }

    @Override // b.t.f
    public void e(h hVar, e.a aVar) {
        Class cls;
        if (aVar == e.a.ON_CREATE) {
            ((i) hVar.getLifecycle()).f2578a.e(this);
            Bundle a2 = this.f487a.getSavedStateRegistry().a("androidx.savedstate.Restarter");
            if (a2 == null) {
                return;
            }
            ArrayList<String> stringArrayList = a2.getStringArrayList("classes_to_restore");
            if (stringArrayList != null) {
                Iterator<String> it = stringArrayList.iterator();
                while (it.hasNext()) {
                    String next = it.next();
                    try {
                        try {
                            Constructor declaredConstructor = Class.forName(next, false, Recreator.class.getClassLoader()).asSubclass(a.InterfaceC0055a.class).getDeclaredConstructor(new Class[0]);
                            declaredConstructor.setAccessible(true);
                            try {
                                ((a.InterfaceC0055a) declaredConstructor.newInstance(new Object[0])).a(this.f487a);
                            } catch (Exception e2) {
                                throw new RuntimeException(c.b.a.a.a.q("Failed to instantiate ", next), e2);
                            }
                        } catch (NoSuchMethodException e3) {
                            StringBuilder x = c.b.a.a.a.x("Class");
                            x.append(cls.getSimpleName());
                            x.append(" must have default constructor in order to be automatically recreated");
                            throw new IllegalStateException(x.toString(), e3);
                        }
                    } catch (ClassNotFoundException e4) {
                        throw new RuntimeException(c.b.a.a.a.r("Class ", next, " wasn't found"), e4);
                    }
                }
                return;
            }
            throw new IllegalStateException("Bundle with restored state for the component \"androidx.savedstate.Restarter\" must contain list of strings by the key \"classes_to_restore\"");
        }
        throw new AssertionError("Next event must be ON_CREATE");
    }
}