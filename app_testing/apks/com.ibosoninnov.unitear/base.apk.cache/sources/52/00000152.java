package b.b.c;

import android.view.View;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: AppCompatDelegateImpl.java */
/* loaded from: classes.dex */
public class n implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ k f596b;

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class a extends b.j.j.u {
        public a() {
        }

        @Override // b.j.j.t
        public void b(View view) {
            n.this.f596b.t.setAlpha(1.0f);
            n.this.f596b.w.d(null);
            n.this.f596b.w = null;
        }

        @Override // b.j.j.u, b.j.j.t
        public void c(View view) {
            n.this.f596b.t.setVisibility(0);
        }
    }

    public n(k kVar) {
        this.f596b = kVar;
    }

    @Override // java.lang.Runnable
    public void run() {
        k kVar = this.f596b;
        kVar.u.showAtLocation(kVar.t, 55, 0, 0);
        this.f596b.G();
        if (this.f596b.T()) {
            this.f596b.t.setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            k kVar2 = this.f596b;
            b.j.j.s b2 = b.j.j.q.b(kVar2.t);
            b2.a(1.0f);
            kVar2.w = b2;
            b.j.j.s sVar = this.f596b.w;
            a aVar = new a();
            View view = sVar.f2231a.get();
            if (view != null) {
                sVar.e(view, aVar);
                return;
            }
            return;
        }
        this.f596b.t.setAlpha(1.0f);
        this.f596b.t.setVisibility(0);
    }
}