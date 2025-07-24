package b.v;

import android.content.Context;
import android.os.Bundle;
import b.t.e;
import b.t.y;
import b.t.z;
import java.util.UUID;

/* compiled from: NavBackStackEntry.java */
/* loaded from: classes.dex */
public final class e implements b.t.h, z, b.x.c {

    /* renamed from: b  reason: collision with root package name */
    public final j f2615b;

    /* renamed from: c  reason: collision with root package name */
    public Bundle f2616c;

    /* renamed from: d  reason: collision with root package name */
    public final b.t.i f2617d;

    /* renamed from: e  reason: collision with root package name */
    public final b.x.b f2618e;

    /* renamed from: f  reason: collision with root package name */
    public final UUID f2619f;

    /* renamed from: g  reason: collision with root package name */
    public e.b f2620g;

    /* renamed from: h  reason: collision with root package name */
    public e.b f2621h;
    public g i;

    public e(Context context, j jVar, Bundle bundle, b.t.h hVar, g gVar) {
        this(context, jVar, bundle, hVar, gVar, UUID.randomUUID(), null);
    }

    public void a() {
        if (this.f2620g.ordinal() < this.f2621h.ordinal()) {
            this.f2617d.f(this.f2620g);
        } else {
            this.f2617d.f(this.f2621h);
        }
    }

    @Override // b.t.h
    public b.t.e getLifecycle() {
        return this.f2617d;
    }

    @Override // b.x.c
    public b.x.a getSavedStateRegistry() {
        return this.f2618e.f2826b;
    }

    @Override // b.t.z
    public y getViewModelStore() {
        g gVar = this.i;
        if (gVar != null) {
            UUID uuid = this.f2619f;
            y yVar = gVar.f2627d.get(uuid);
            if (yVar == null) {
                y yVar2 = new y();
                gVar.f2627d.put(uuid, yVar2);
                return yVar2;
            }
            return yVar;
        }
        throw new IllegalStateException("You must call setViewModelStore() on your NavHostController before accessing the ViewModelStore of a navigation graph.");
    }

    public e(Context context, j jVar, Bundle bundle, b.t.h hVar, g gVar, UUID uuid, Bundle bundle2) {
        this.f2617d = new b.t.i(this);
        b.x.b bVar = new b.x.b(this);
        this.f2618e = bVar;
        this.f2620g = e.b.CREATED;
        this.f2621h = e.b.RESUMED;
        this.f2619f = uuid;
        this.f2615b = jVar;
        this.f2616c = bundle;
        this.i = gVar;
        bVar.a(bundle2);
        if (hVar != null) {
            this.f2620g = ((b.t.i) hVar.getLifecycle()).f2579b;
        }
    }
}