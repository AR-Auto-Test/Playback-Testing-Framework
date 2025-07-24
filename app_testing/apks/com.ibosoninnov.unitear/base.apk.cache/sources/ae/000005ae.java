package b.w.b;

import android.view.View;
import android.view.ViewPropertyAnimator;
import androidx.recyclerview.widget.RecyclerView;
import b.w.b.k;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Objects;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class d implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2719b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ k f2720c;

    public d(k kVar, ArrayList arrayList) {
        this.f2720c = kVar;
        this.f2719b = arrayList;
    }

    @Override // java.lang.Runnable
    public void run() {
        Iterator it = this.f2719b.iterator();
        while (it.hasNext()) {
            k.a aVar = (k.a) it.next();
            k kVar = this.f2720c;
            Objects.requireNonNull(kVar);
            RecyclerView.d0 d0Var = aVar.f2746a;
            View view = d0Var == null ? null : d0Var.itemView;
            RecyclerView.d0 d0Var2 = aVar.f2747b;
            View view2 = d0Var2 != null ? d0Var2.itemView : null;
            if (view != null) {
                ViewPropertyAnimator duration = view.animate().setDuration(kVar.f414f);
                kVar.s.add(aVar.f2746a);
                duration.translationX(aVar.f2750e - aVar.f2748c);
                duration.translationY(aVar.f2751f - aVar.f2749d);
                duration.alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setListener(new i(kVar, aVar, duration, view)).start();
            }
            if (view2 != null) {
                ViewPropertyAnimator animate = view2.animate();
                kVar.s.add(aVar.f2747b);
                animate.translationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).translationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setDuration(kVar.f414f).alpha(1.0f).setListener(new j(kVar, aVar, animate, view2)).start();
            }
        }
        this.f2719b.clear();
        this.f2720c.o.remove(this.f2719b);
    }
}