package c.e.b.p000if;

import android.content.Context;
import android.graphics.Rect;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: RecycleViewCellSpacing.java */
/* renamed from: c.e.b.if.n  reason: invalid package */
/* loaded from: classes2.dex */
public class n extends RecyclerView.n {

    /* renamed from: a  reason: collision with root package name */
    public int f4894a;

    public n(Context context, int i) {
        this.f4894a = context.getResources().getDimensionPixelSize(i);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.n
    public void getItemOffsets(Rect rect, View view, RecyclerView recyclerView, RecyclerView.a0 a0Var) {
        super.getItemOffsets(rect, view, recyclerView, a0Var);
        int i = this.f4894a;
        rect.set(i, i, i, i);
    }
}