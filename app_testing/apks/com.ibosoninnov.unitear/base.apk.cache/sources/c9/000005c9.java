package b.w.b;

import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: PagerSnapHelper.java */
/* loaded from: classes.dex */
public class u extends x {

    /* renamed from: c  reason: collision with root package name */
    public s f2798c;

    /* renamed from: d  reason: collision with root package name */
    public s f2799d;

    @Override // b.w.b.x
    public int[] a(RecyclerView.o oVar, View view) {
        int[] iArr = new int[2];
        if (oVar.canScrollHorizontally()) {
            iArr[0] = d(view, f(oVar));
        } else {
            iArr[0] = 0;
        }
        if (oVar.canScrollVertically()) {
            iArr[1] = d(view, g(oVar));
        } else {
            iArr[1] = 0;
        }
        return iArr;
    }

    @Override // b.w.b.x
    public View b(RecyclerView.o oVar) {
        if (oVar.canScrollVertically()) {
            return e(oVar, g(oVar));
        }
        if (oVar.canScrollHorizontally()) {
            return e(oVar, f(oVar));
        }
        return null;
    }

    public final int d(View view, s sVar) {
        return ((sVar.c(view) / 2) + sVar.e(view)) - ((sVar.l() / 2) + sVar.k());
    }

    public final View e(RecyclerView.o oVar, s sVar) {
        int childCount = oVar.getChildCount();
        View view = null;
        if (childCount == 0) {
            return null;
        }
        int l = (sVar.l() / 2) + sVar.k();
        int i = Integer.MAX_VALUE;
        for (int i2 = 0; i2 < childCount; i2++) {
            View childAt = oVar.getChildAt(i2);
            int abs = Math.abs(((sVar.c(childAt) / 2) + sVar.e(childAt)) - l);
            if (abs < i) {
                view = childAt;
                i = abs;
            }
        }
        return view;
    }

    public final s f(RecyclerView.o oVar) {
        s sVar = this.f2799d;
        if (sVar == null || sVar.f2794a != oVar) {
            this.f2799d = new q(oVar);
        }
        return this.f2799d;
    }

    public final s g(RecyclerView.o oVar) {
        s sVar = this.f2798c;
        if (sVar == null || sVar.f2794a != oVar) {
            this.f2798c = new r(oVar);
        }
        return this.f2798c;
    }
}