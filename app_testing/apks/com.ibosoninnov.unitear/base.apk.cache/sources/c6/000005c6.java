package b.w.b;

import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: OrientationHelper.java */
/* loaded from: classes.dex */
public final class r extends s {
    public r(RecyclerView.o oVar) {
        super(oVar, null);
    }

    @Override // b.w.b.s
    public int b(View view) {
        return this.f2794a.getDecoratedBottom(view) + ((ViewGroup.MarginLayoutParams) ((RecyclerView.p) view.getLayoutParams())).bottomMargin;
    }

    @Override // b.w.b.s
    public int c(View view) {
        RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
        return this.f2794a.getDecoratedMeasuredHeight(view) + ((ViewGroup.MarginLayoutParams) pVar).topMargin + ((ViewGroup.MarginLayoutParams) pVar).bottomMargin;
    }

    @Override // b.w.b.s
    public int d(View view) {
        RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
        return this.f2794a.getDecoratedMeasuredWidth(view) + ((ViewGroup.MarginLayoutParams) pVar).leftMargin + ((ViewGroup.MarginLayoutParams) pVar).rightMargin;
    }

    @Override // b.w.b.s
    public int e(View view) {
        return this.f2794a.getDecoratedTop(view) - ((ViewGroup.MarginLayoutParams) ((RecyclerView.p) view.getLayoutParams())).topMargin;
    }

    @Override // b.w.b.s
    public int f() {
        return this.f2794a.getHeight();
    }

    @Override // b.w.b.s
    public int g() {
        return this.f2794a.getHeight() - this.f2794a.getPaddingBottom();
    }

    @Override // b.w.b.s
    public int h() {
        return this.f2794a.getPaddingBottom();
    }

    @Override // b.w.b.s
    public int i() {
        return this.f2794a.getHeightMode();
    }

    @Override // b.w.b.s
    public int j() {
        return this.f2794a.getWidthMode();
    }

    @Override // b.w.b.s
    public int k() {
        return this.f2794a.getPaddingTop();
    }

    @Override // b.w.b.s
    public int l() {
        return (this.f2794a.getHeight() - this.f2794a.getPaddingTop()) - this.f2794a.getPaddingBottom();
    }

    @Override // b.w.b.s
    public int n(View view) {
        this.f2794a.getTransformedBoundingBox(view, true, this.f2796c);
        return this.f2796c.bottom;
    }

    @Override // b.w.b.s
    public int o(View view) {
        this.f2794a.getTransformedBoundingBox(view, true, this.f2796c);
        return this.f2796c.top;
    }

    @Override // b.w.b.s
    public void p(int i) {
        this.f2794a.offsetChildrenVertical(i);
    }
}