package b.w.b;

import androidx.recyclerview.widget.RecyclerView;
import b.w.b.k;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: SimpleItemAnimator.java */
/* loaded from: classes.dex */
public abstract class w extends RecyclerView.l {

    /* renamed from: g  reason: collision with root package name */
    public boolean f2802g = true;

    @Override // androidx.recyclerview.widget.RecyclerView.l
    public boolean a(RecyclerView.d0 d0Var, RecyclerView.d0 d0Var2, RecyclerView.l.c cVar, RecyclerView.l.c cVar2) {
        int i;
        int i2;
        int i3 = cVar.f415a;
        int i4 = cVar.f416b;
        if (d0Var2.shouldIgnore()) {
            int i5 = cVar.f415a;
            i2 = cVar.f416b;
            i = i5;
        } else {
            i = cVar2.f415a;
            i2 = cVar2.f416b;
        }
        k kVar = (k) this;
        if (d0Var == d0Var2) {
            return kVar.i(d0Var, i3, i4, i, i2);
        }
        float translationX = d0Var.itemView.getTranslationX();
        float translationY = d0Var.itemView.getTranslationY();
        float alpha = d0Var.itemView.getAlpha();
        kVar.n(d0Var);
        d0Var.itemView.setTranslationX(translationX);
        d0Var.itemView.setTranslationY(translationY);
        d0Var.itemView.setAlpha(alpha);
        kVar.n(d0Var2);
        d0Var2.itemView.setTranslationX(-((int) ((i - i3) - translationX)));
        d0Var2.itemView.setTranslationY(-((int) ((i2 - i4) - translationY)));
        d0Var2.itemView.setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        kVar.l.add(new k.a(d0Var, d0Var2, i3, i4, i, i2));
        return true;
    }

    public abstract boolean i(RecyclerView.d0 d0Var, int i, int i2, int i3, int i4);
}