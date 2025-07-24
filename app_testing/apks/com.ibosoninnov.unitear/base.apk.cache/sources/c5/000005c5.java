package b.w.b;

import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: OrientationHelper.java */
/* loaded from: classes.dex */
public final class q extends s {
    public q(RecyclerView.o oVar) {
        super(oVar, null);
    }

    @Override // b.w.b.s
    public int b(View view) {
        return this.f2794a.getDecoratedRight(view) + ((ViewGroup.MarginLayoutParams) ((RecyclerView.p) view.getLayoutParams())).rightMargin;
    }

    @Override // b.w.b.s
    public int c(View view) {
        RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
        return this.f2794a.getDecoratedMeasuredWidth(view) + ((ViewGroup.MarginLayoutParams) pVar).leftMargin + ((ViewGroup.MarginLayoutParams) pVar).rightMargin;
    }

    @Override // b.w.b.s
    public int d(View view) {
        RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
        return this.f2794a.getDecoratedMeasuredHeight(view) + ((ViewGroup.MarginLayoutParams) pVar).topMargin + ((ViewGroup.MarginLayoutParams) pVar).bottomMargin;
    }

    @Override // b.w.b.s
    public int e(View view) {
        return this.f2794a.getDecoratedLeft(view) - ((ViewGroup.MarginLayoutParams) ((RecyclerView.p) view.getLayoutParams())).leftMargin;
    }

    @Override // b.w.b.s
    public int f() {
        return this.f2794a.getWidth();
    }

    @Override // b.w.b.s
    public int g() {
        return this.f2794a.getWidth() - this.f2794a.getPaddingRight();
    }

    @Override // b.w.b.s
    public int h() {
        return this.f2794a.getPaddingRight();
    }

    @Override // b.w.b.s
    public int i() {
        return this.f2794a.getWidthMode();
    }

    @Override // b.w.b.s
    public int j() {
        return this.f2794a.getHeightMode();
    }

    @Override // b.w.b.s
    public int k() {
        return this.f2794a.getPaddingLeft();
    }

    @Override // b.w.b.s
    public int l() {
        return (this.f2794a.getWidth() - this.f2794a.getPaddingLeft()) - this.f2794a.getPaddingRight();
    }

    @Override // b.w.b.s
    public int n(View view) {
        this.f2794a.getTransformedBoundingBox(view, true, this.f2796c);
        return this.f2796c.right;
    }

    @Override // b.w.b.s
    public int o(View view) {
        this.f2794a.getTransformedBoundingBox(view, true, this.f2796c);
        return this.f2796c.left;
    }

    @Override // b.w.b.s
    public void p(int i) {
        this.f2794a.offsetChildrenHorizontal(i);
    }
}