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
public class c implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ArrayList f2717b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ k f2718c;

    public c(k kVar, ArrayList arrayList) {
        this.f2718c = kVar;
        this.f2717b = arrayList;
    }

    @Override // java.lang.Runnable
    public void run() {
        Iterator it = this.f2717b.iterator();
        while (it.hasNext()) {
            k.b bVar = (k.b) it.next();
            k kVar = this.f2718c;
            RecyclerView.d0 d0Var = bVar.f2752a;
            int i = bVar.f2753b;
            int i2 = bVar.f2754c;
            int i3 = bVar.f2755d;
            int i4 = bVar.f2756e;
            Objects.requireNonNull(kVar);
            View view = d0Var.itemView;
            int i5 = i3 - i;
            int i6 = i4 - i2;
            if (i5 != 0) {
                view.animate().translationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            if (i6 != 0) {
                view.animate().translationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            ViewPropertyAnimator animate = view.animate();
            kVar.q.add(d0Var);
            animate.setDuration(kVar.f413e).setListener(new h(kVar, d0Var, i5, view, i6, animate)).start();
        }
        this.f2717b.clear();
        this.f2718c.n.remove(this.f2717b);
    }
}