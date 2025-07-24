package b.x;

import android.os.Bundle;
import androidx.savedstate.Recreator;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.x.a;
import java.util.Map;
import java.util.Objects;

/* compiled from: SavedStateRegistryController.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public final c f2825a;

    /* renamed from: b  reason: collision with root package name */
    public final a f2826b = new a();

    public b(c cVar) {
        this.f2825a = cVar;
    }

    public void a(Bundle bundle) {
        e lifecycle = this.f2825a.getLifecycle();
        if (((i) lifecycle).f2579b == e.b.INITIALIZED) {
            lifecycle.a(new Recreator(this.f2825a));
            final a aVar = this.f2826b;
            if (!aVar.f2822c) {
                if (bundle != null) {
                    aVar.f2821b = bundle.getBundle("androidx.lifecycle.BundlableSavedStateRegistry.key");
                }
                lifecycle.a(new f() { // from class: androidx.savedstate.SavedStateRegistry$1
                    @Override // b.t.f
                    public void e(h hVar, e.a aVar2) {
                        if (aVar2 == e.a.ON_START) {
                            a.this.f2824e = true;
                        } else if (aVar2 == e.a.ON_STOP) {
                            a.this.f2824e = false;
                        }
                    }
                });
                aVar.f2822c = true;
                return;
            }
            throw new IllegalStateException("SavedStateRegistry was already restored.");
        }
        throw new IllegalStateException("Restarter must be created only during owner's initialization stage");
    }

    public void b(Bundle bundle) {
        a aVar = this.f2826b;
        Objects.requireNonNull(aVar);
        Bundle bundle2 = new Bundle();
        Bundle bundle3 = aVar.f2821b;
        if (bundle3 != null) {
            bundle2.putAll(bundle3);
        }
        b.c.a.b.b<String, a.b>.d b2 = aVar.f2820a.b();
        while (b2.hasNext()) {
            Map.Entry entry = (Map.Entry) b2.next();
            bundle2.putBundle((String) entry.getKey(), ((a.b) entry.getValue()).a());
        }
        bundle.putBundle("androidx.lifecycle.BundlableSavedStateRegistry.key", bundle2);
    }
}