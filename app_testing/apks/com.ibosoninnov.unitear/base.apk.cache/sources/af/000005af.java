package b.w.b;

import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class e implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2721b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ k f2722c;

    public e(k kVar, ArrayList arrayList) {
        this.f2722c = kVar;
        this.f2721b = arrayList;
    }

    @Override // java.lang.Runnable
    public void run() {
        Iterator it = this.f2721b.iterator();
        while (it.hasNext()) {
            RecyclerView.d0 d0Var = (RecyclerView.d0) it.next();
            k kVar = this.f2722c;
            Objects.requireNonNull(kVar);
            View view = d0Var.itemView;
            ViewPropertyAnimator animate = view.animate();
            kVar.p.add(d0Var);
            animate.alpha(1.0f).setDuration(kVar.f411c).setListener(new g(kVar, d0Var, view, animate)).start();
        }
        this.f2721b.clear();
        this.f2722c.m.remove(this.f2721b);
    }
}