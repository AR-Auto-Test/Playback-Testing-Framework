package b.b.c;

import android.view.View;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AppCompatDelegateImpl.java */
/* loaded from: classes.dex */
public class o extends b.j.j.u {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ k f598a;

    public o(k kVar) {
        this.f598a = kVar;
    }

    @Override // b.j.j.t
    public void b(View view) {
        this.f598a.t.setAlpha(1.0f);
        this.f598a.w.d(null);
        this.f598a.w = null;
    }

    @Override // b.j.j.u, b.j.j.t
    public void c(View view) {
        this.f598a.t.setVisibility(0);
        this.f598a.t.sendAccessibilityEvent(32);
        if (this.f598a.t.getParent() instanceof View) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            ((View) this.f598a.t.getParent()).requestApplyInsets();
        }
    }
}