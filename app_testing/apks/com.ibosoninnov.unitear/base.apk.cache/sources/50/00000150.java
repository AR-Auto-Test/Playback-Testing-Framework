package b.b.c;

import android.view.View;
import b.j.j.w;

/* compiled from: AppCompatDelegateImpl.java */
/* loaded from: classes.dex */
public class l implements b.j.j.j {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ k f594a;

    public l(k kVar) {
        this.f594a = kVar;
    }

    @Override // b.j.j.j
    public w onApplyWindowInsets(View view, w wVar) {
        int e2 = wVar.e();
        int V = this.f594a.V(wVar, null);
        if (e2 != V) {
            wVar = wVar.h(wVar.c(), V, wVar.d(), wVar.b());
        }
        return b.j.j.q.j(view, wVar);
    }
}