package b.b.h;

import android.annotation.SuppressLint;
import android.view.View;
import b.b.h.w;

/* compiled from: AppCompatSpinner.java */
/* loaded from: classes.dex */
public class v extends h0 {
    public final /* synthetic */ w.d k;
    public final /* synthetic */ w l;

    /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
    public v(w wVar, View view, w.d dVar) {
        super(view);
        this.l = wVar;
        this.k = dVar;
    }

    @Override // b.b.h.h0
    public b.b.g.i.p b() {
        return this.k;
    }

    @Override // b.b.h.h0
    @SuppressLint({"SyntheticAccessor"})
    public boolean c() {
        if (this.l.getInternalPopup().a()) {
            return true;
        }
        this.l.b();
        return true;
    }
}