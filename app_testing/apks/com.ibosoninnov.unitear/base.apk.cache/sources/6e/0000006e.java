package androidx.camera.lifecycle;

import b.d.b.a1;
import b.d.b.e0;
import b.d.b.e1.c;
import b.d.b.f0;
import b.d.b.i0;
import b.t.e;
import b.t.g;
import b.t.h;
import b.t.i;
import b.t.o;
import java.util.Collections;
import java.util.List;

/* loaded from: classes.dex */
public final class LifecycleCamera implements g, e0 {

    /* renamed from: b  reason: collision with root package name */
    public final h f186b;

    /* renamed from: c  reason: collision with root package name */
    public final c f187c;

    /* renamed from: a  reason: collision with root package name */
    public final Object f185a = new Object();

    /* renamed from: d  reason: collision with root package name */
    public boolean f188d = false;

    public LifecycleCamera(h hVar, c cVar) {
        this.f186b = hVar;
        this.f187c = cVar;
        if (((i) hVar.getLifecycle()).f2579b.compareTo(e.b.STARTED) >= 0) {
            cVar.d();
        } else {
            cVar.f();
        }
        hVar.getLifecycle().a(this);
    }

    @Override // b.d.b.e0
    public f0 a() {
        return this.f187c.f1597a.g();
    }

    @Override // b.d.b.e0
    public i0 b() {
        return this.f187c.b();
    }

    public h k() {
        h hVar;
        synchronized (this.f185a) {
            hVar = this.f186b;
        }
        return hVar;
    }

    public List<a1> l() {
        List<a1> unmodifiableList;
        synchronized (this.f185a) {
            unmodifiableList = Collections.unmodifiableList(this.f187c.k());
        }
        return unmodifiableList;
    }

    public void m() {
        synchronized (this.f185a) {
            if (this.f188d) {
                return;
            }
            onStop(this.f186b);
            this.f188d = true;
        }
    }

    public void n() {
        synchronized (this.f185a) {
            if (this.f188d) {
                this.f188d = false;
                if (((i) this.f186b.getLifecycle()).f2579b.compareTo(e.b.STARTED) >= 0) {
                    onStart(this.f186b);
                }
            }
        }
    }

    @o(e.a.ON_DESTROY)
    public void onDestroy(h hVar) {
        synchronized (this.f185a) {
            c cVar = this.f187c;
            cVar.l(cVar.k());
        }
    }

    @o(e.a.ON_START)
    public void onStart(h hVar) {
        synchronized (this.f185a) {
            if (!this.f188d) {
                this.f187c.d();
            }
        }
    }

    @o(e.a.ON_STOP)
    public void onStop(h hVar) {
        synchronized (this.f185a) {
            if (!this.f188d) {
                this.f187c.f();
            }
        }
    }
}