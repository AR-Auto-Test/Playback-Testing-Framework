package androidx.recyclerview.widget;

import android.content.Context;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import b.j.j.x.b;
import b.w.b.m;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.Objects;

/* loaded from: classes.dex */
public class GridLayoutManager extends LinearLayoutManager {

    /* renamed from: a  reason: collision with root package name */
    public boolean f357a;

    /* renamed from: b  reason: collision with root package name */
    public int f358b;

    /* renamed from: c  reason: collision with root package name */
    public int[] f359c;

    /* renamed from: d  reason: collision with root package name */
    public View[] f360d;

    /* renamed from: e  reason: collision with root package name */
    public final SparseIntArray f361e;

    /* renamed from: f  reason: collision with root package name */
    public final SparseIntArray f362f;

    /* renamed from: g  reason: collision with root package name */
    public c f363g;

    /* renamed from: h  reason: collision with root package name */
    public final Rect f364h;

    /* loaded from: classes.dex */
    public static final class a extends c {
    }

    /* loaded from: classes.dex */
    public static abstract class c {

        /* renamed from: a  reason: collision with root package name */
        public final SparseIntArray f367a = new SparseIntArray();

        /* renamed from: b  reason: collision with root package name */
        public final SparseIntArray f368b = new SparseIntArray();

        public int a(int i, int i2) {
            int i3 = 0;
            int i4 = 0;
            for (int i5 = 0; i5 < i; i5++) {
                i3++;
                if (i3 == i2) {
                    i4++;
                    i3 = 0;
                } else if (i3 > i2) {
                    i4++;
                    i3 = 1;
                }
            }
            return i3 + 1 > i2 ? i4 + 1 : i4;
        }
    }

    public GridLayoutManager(Context context, AttributeSet attributeSet, int i, int i2) {
        super(context, attributeSet, i, i2);
        this.f357a = false;
        this.f358b = -1;
        this.f361e = new SparseIntArray();
        this.f362f = new SparseIntArray();
        this.f363g = new a();
        this.f364h = new Rect();
        i(RecyclerView.o.getProperties(context, attributeSet, i, i2).f421b);
    }

    public final void a(int i) {
        int i2;
        int[] iArr = this.f359c;
        int i3 = this.f358b;
        if (iArr == null || iArr.length != i3 + 1 || iArr[iArr.length - 1] != i) {
            iArr = new int[i3 + 1];
        }
        int i4 = 0;
        iArr[0] = 0;
        int i5 = i / i3;
        int i6 = i % i3;
        int i7 = 0;
        for (int i8 = 1; i8 <= i3; i8++) {
            i4 += i6;
            if (i4 <= 0 || i3 - i4 >= i6) {
                i2 = i5;
            } else {
                i2 = i5 + 1;
                i4 -= i3;
            }
            i7 += i2;
            iArr[i8] = i7;
        }
        this.f359c = iArr;
    }

    public final void b() {
        View[] viewArr = this.f360d;
        if (viewArr == null || viewArr.length != this.f358b) {
            this.f360d = new View[this.f358b];
        }
    }

    public int c(int i, int i2) {
        if (this.mOrientation == 1 && isLayoutRTL()) {
            int[] iArr = this.f359c;
            int i3 = this.f358b;
            return iArr[i3 - i] - iArr[(i3 - i) - i2];
        }
        int[] iArr2 = this.f359c;
        return iArr2[i2 + i] - iArr2[i];
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean checkLayoutParams(RecyclerView.p pVar) {
        return pVar instanceof b;
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager
    public void collectPrefetchPositionsForLayoutState(RecyclerView.a0 a0Var, LinearLayoutManager.c cVar, RecyclerView.o.c cVar2) {
        int i = this.f358b;
        for (int i2 = 0; i2 < this.f358b && cVar.b(a0Var) && i > 0; i2++) {
            ((m.b) cVar2).a(cVar.f381d, Math.max(0, cVar.f384g));
            Objects.requireNonNull(this.f363g);
            i--;
            cVar.f381d += cVar.f382e;
        }
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollOffset(RecyclerView.a0 a0Var) {
        return super.computeHorizontalScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollRange(RecyclerView.a0 a0Var) {
        return super.computeHorizontalScrollRange(a0Var);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollOffset(RecyclerView.a0 a0Var) {
        return super.computeVerticalScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollRange(RecyclerView.a0 a0Var) {
        return super.computeVerticalScrollRange(a0Var);
    }

    public final int d(RecyclerView.v vVar, RecyclerView.a0 a0Var, int i) {
        if (!a0Var.f396g) {
            return this.f363g.a(i, this.f358b);
        }
        int c2 = vVar.c(i);
        if (c2 == -1) {
            Log.w("GridLayoutManager", "Cannot find span size for pre layout position. " + i);
            return 0;
        }
        return this.f363g.a(c2, this.f358b);
    }

    public final int e(RecyclerView.v vVar, RecyclerView.a0 a0Var, int i) {
        if (!a0Var.f396g) {
            c cVar = this.f363g;
            int i2 = this.f358b;
            Objects.requireNonNull(cVar);
            return i % i2;
        }
        int i3 = this.f362f.get(i, -1);
        if (i3 != -1) {
            return i3;
        }
        int c2 = vVar.c(i);
        if (c2 == -1) {
            Log.w("GridLayoutManager", "Cannot find span size for pre layout position. It is not cached, not in the adapter. Pos:" + i);
            return 0;
        }
        c cVar2 = this.f363g;
        int i4 = this.f358b;
        Objects.requireNonNull(cVar2);
        return c2 % i4;
    }

    public final int f(RecyclerView.v vVar, RecyclerView.a0 a0Var, int i) {
        if (!a0Var.f396g) {
            Objects.requireNonNull(this.f363g);
            return 1;
        }
        int i2 = this.f361e.get(i, -1);
        if (i2 != -1) {
            return i2;
        }
        if (vVar.c(i) == -1) {
            Log.w("GridLayoutManager", "Cannot find span size for pre layout position. It is not cached, not in the adapter. Pos:" + i);
            return 1;
        }
        Objects.requireNonNull(this.f363g);
        return 1;
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager
    public View findReferenceChild(RecyclerView.v vVar, RecyclerView.a0 a0Var, int i, int i2, int i3) {
        ensureLayoutState();
        int k = this.mOrientationHelper.k();
        int g2 = this.mOrientationHelper.g();
        int i4 = i2 > i ? 1 : -1;
        View view = null;
        View view2 = null;
        while (i != i2) {
            View childAt = getChildAt(i);
            int position = getPosition(childAt);
            if (position >= 0 && position < i3 && e(vVar, a0Var, position) == 0) {
                if (((RecyclerView.p) childAt.getLayoutParams()).c()) {
                    if (view2 == null) {
                        view2 = childAt;
                    }
                } else if (this.mOrientationHelper.e(childAt) < g2 && this.mOrientationHelper.b(childAt) >= k) {
                    return childAt;
                } else {
                    if (view == null) {
                        view = childAt;
                    }
                }
            }
            i += i4;
        }
        return view != null ? view : view2;
    }

    public final void g(View view, int i, boolean z) {
        int i2;
        int i3;
        b bVar = (b) view.getLayoutParams();
        Rect rect = bVar.f425b;
        int i4 = rect.top + rect.bottom + ((ViewGroup.MarginLayoutParams) bVar).topMargin + ((ViewGroup.MarginLayoutParams) bVar).bottomMargin;
        int i5 = rect.left + rect.right + ((ViewGroup.MarginLayoutParams) bVar).leftMargin + ((ViewGroup.MarginLayoutParams) bVar).rightMargin;
        int c2 = c(bVar.f365e, bVar.f366f);
        if (this.mOrientation == 1) {
            i3 = RecyclerView.o.getChildMeasureSpec(c2, i, i5, ((ViewGroup.MarginLayoutParams) bVar).width, false);
            i2 = RecyclerView.o.getChildMeasureSpec(this.mOrientationHelper.l(), getHeightMode(), i4, ((ViewGroup.MarginLayoutParams) bVar).height, true);
        } else {
            int childMeasureSpec = RecyclerView.o.getChildMeasureSpec(c2, i, i4, ((ViewGroup.MarginLayoutParams) bVar).height, false);
            int childMeasureSpec2 = RecyclerView.o.getChildMeasureSpec(this.mOrientationHelper.l(), getWidthMode(), i5, ((ViewGroup.MarginLayoutParams) bVar).width, true);
            i2 = childMeasureSpec;
            i3 = childMeasureSpec2;
        }
        h(view, i3, i2, z);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateDefaultLayoutParams() {
        if (this.mOrientation == 0) {
            return new b(-2, -1);
        }
        return new b(-1, -2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateLayoutParams(Context context, AttributeSet attributeSet) {
        return new b(context, attributeSet);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int getColumnCountForAccessibility(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.mOrientation == 1) {
            return this.f358b;
        }
        if (a0Var.b() < 1) {
            return 0;
        }
        return d(vVar, a0Var, a0Var.b() - 1) + 1;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int getRowCountForAccessibility(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.mOrientation == 0) {
            return this.f358b;
        }
        if (a0Var.b() < 1) {
            return 0;
        }
        return d(vVar, a0Var, a0Var.b() - 1) + 1;
    }

    public final void h(View view, int i, int i2, boolean z) {
        boolean shouldMeasureChild;
        RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
        if (z) {
            shouldMeasureChild = shouldReMeasureChild(view, i, i2, pVar);
        } else {
            shouldMeasureChild = shouldMeasureChild(view, i, i2, pVar);
        }
        if (shouldMeasureChild) {
            view.measure(i, i2);
        }
    }

    public void i(int i) {
        if (i == this.f358b) {
            return;
        }
        this.f357a = true;
        if (i >= 1) {
            this.f358b = i;
            this.f363g.f367a.clear();
            requestLayout();
            return;
        }
        throw new IllegalArgumentException(c.b.a.a.a.j("Span count should be at least 1. Provided ", i));
    }

    public final void j() {
        int height;
        int paddingTop;
        if (getOrientation() == 1) {
            height = getWidth() - getPaddingRight();
            paddingTop = getPaddingLeft();
        } else {
            height = getHeight() - getPaddingBottom();
            paddingTop = getPaddingTop();
        }
        a(height - paddingTop);
    }

    /* JADX WARN: Code restructure failed: missing block: B:37:0x0086, code lost:
        r21.f375b = true;
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:0x0088, code lost:
        return;
     */
    @Override // androidx.recyclerview.widget.LinearLayoutManager
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void layoutChunk(RecyclerView.v vVar, RecyclerView.a0 a0Var, LinearLayoutManager.c cVar, LinearLayoutManager.b bVar) {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        int i6;
        int i7;
        int i8;
        int i9;
        int i10;
        int i11;
        int i12;
        int i13;
        int i14;
        int childMeasureSpec;
        int i15;
        boolean z;
        View c2;
        int j = this.mOrientationHelper.j();
        boolean z2 = j != 1073741824;
        int i16 = getChildCount() > 0 ? this.f359c[this.f358b] : 0;
        if (z2) {
            j();
        }
        boolean z3 = cVar.f382e == 1;
        int i17 = this.f358b;
        if (!z3) {
            i17 = e(vVar, a0Var, cVar.f381d) + f(vVar, a0Var, cVar.f381d);
        }
        int i18 = 0;
        while (i18 < this.f358b && cVar.b(a0Var) && i17 > 0) {
            int i19 = cVar.f381d;
            int f2 = f(vVar, a0Var, i19);
            if (f2 > this.f358b) {
                throw new IllegalArgumentException(c.b.a.a.a.s(c.b.a.a.a.z("Item at position ", i19, " requires ", f2, " spans but GridLayoutManager has only "), this.f358b, " spans."));
            }
            i17 -= f2;
            if (i17 < 0 || (c2 = cVar.c(vVar)) == null) {
                break;
            }
            this.f360d[i18] = c2;
            i18++;
        }
        if (z3) {
            i3 = 1;
            i2 = i18;
            i = 0;
        } else {
            i = i18 - 1;
            i2 = -1;
            i3 = -1;
        }
        int i20 = 0;
        while (i != i2) {
            View view = this.f360d[i];
            b bVar2 = (b) view.getLayoutParams();
            int f3 = f(vVar, a0Var, getPosition(view));
            bVar2.f366f = f3;
            bVar2.f365e = i20;
            i20 += f3;
            i += i3;
        }
        float f4 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        int i21 = 0;
        for (int i22 = 0; i22 < i18; i22++) {
            View view2 = this.f360d[i22];
            if (cVar.k != null) {
                z = false;
                if (z3) {
                    addDisappearingView(view2);
                } else {
                    addDisappearingView(view2, 0);
                }
            } else if (z3) {
                addView(view2);
                z = false;
            } else {
                z = false;
                addView(view2, 0);
            }
            calculateItemDecorationsForChild(view2, this.f364h);
            g(view2, j, z);
            int c3 = this.mOrientationHelper.c(view2);
            if (c3 > i21) {
                i21 = c3;
            }
            float d2 = (this.mOrientationHelper.d(view2) * 1.0f) / ((b) view2.getLayoutParams()).f366f;
            if (d2 > f4) {
                f4 = d2;
            }
        }
        if (z2) {
            a(Math.max(Math.round(f4 * this.f358b), i16));
            i21 = 0;
            for (int i23 = 0; i23 < i18; i23++) {
                View view3 = this.f360d[i23];
                g(view3, 1073741824, true);
                int c4 = this.mOrientationHelper.c(view3);
                if (c4 > i21) {
                    i21 = c4;
                }
            }
        }
        for (int i24 = 0; i24 < i18; i24++) {
            View view4 = this.f360d[i24];
            if (this.mOrientationHelper.c(view4) != i21) {
                b bVar3 = (b) view4.getLayoutParams();
                Rect rect = bVar3.f425b;
                int i25 = rect.top + rect.bottom + ((ViewGroup.MarginLayoutParams) bVar3).topMargin + ((ViewGroup.MarginLayoutParams) bVar3).bottomMargin;
                int i26 = rect.left + rect.right + ((ViewGroup.MarginLayoutParams) bVar3).leftMargin + ((ViewGroup.MarginLayoutParams) bVar3).rightMargin;
                int c5 = c(bVar3.f365e, bVar3.f366f);
                if (this.mOrientation == 1) {
                    i15 = RecyclerView.o.getChildMeasureSpec(c5, 1073741824, i26, ((ViewGroup.MarginLayoutParams) bVar3).width, false);
                    childMeasureSpec = View.MeasureSpec.makeMeasureSpec(i21 - i25, 1073741824);
                } else {
                    int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i21 - i26, 1073741824);
                    childMeasureSpec = RecyclerView.o.getChildMeasureSpec(c5, 1073741824, i25, ((ViewGroup.MarginLayoutParams) bVar3).height, false);
                    i15 = makeMeasureSpec;
                }
                h(view4, i15, childMeasureSpec, true);
            }
        }
        bVar.f374a = i21;
        if (this.mOrientation == 1) {
            if (cVar.f383f == -1) {
                i14 = cVar.f379b;
                i13 = i14 - i21;
            } else {
                int i27 = cVar.f379b;
                i13 = i27;
                i14 = i21 + i27;
            }
            i7 = i13;
            i6 = 0;
            i8 = i14;
            i5 = 0;
        } else {
            if (cVar.f383f == -1) {
                i5 = cVar.f379b;
                i4 = i5 - i21;
            } else {
                int i28 = cVar.f379b;
                i4 = i28;
                i5 = i21 + i28;
            }
            i6 = i4;
            i7 = 0;
            i8 = 0;
        }
        int i29 = 0;
        while (i29 < i18) {
            View view5 = this.f360d[i29];
            b bVar4 = (b) view5.getLayoutParams();
            if (this.mOrientation == 1) {
                if (isLayoutRTL()) {
                    i5 = getPaddingLeft() + this.f359c[this.f358b - bVar4.f365e];
                    i6 = i5 - this.mOrientationHelper.d(view5);
                } else {
                    int paddingLeft = getPaddingLeft() + this.f359c[bVar4.f365e];
                    i10 = paddingLeft;
                    i11 = i7;
                    i9 = i8;
                    i12 = this.mOrientationHelper.d(view5) + paddingLeft;
                    layoutDecoratedWithMargins(view5, i10, i11, i12, i9);
                    if (!bVar4.c() || bVar4.b()) {
                        bVar.f376c = true;
                    }
                    bVar.f377d |= view5.hasFocusable();
                    i29++;
                    i5 = i12;
                    i7 = i11;
                    i6 = i10;
                    i8 = i9;
                }
            } else {
                i7 = getPaddingTop() + this.f359c[bVar4.f365e];
                i8 = this.mOrientationHelper.d(view5) + i7;
            }
            i11 = i7;
            i10 = i6;
            i9 = i8;
            i12 = i5;
            layoutDecoratedWithMargins(view5, i10, i11, i12, i9);
            if (!bVar4.c()) {
            }
            bVar.f376c = true;
            bVar.f377d |= view5.hasFocusable();
            i29++;
            i5 = i12;
            i7 = i11;
            i6 = i10;
            i8 = i9;
        }
        Arrays.fill(this.f360d, (Object) null);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager
    public void onAnchorReady(RecyclerView.v vVar, RecyclerView.a0 a0Var, LinearLayoutManager.a aVar, int i) {
        super.onAnchorReady(vVar, a0Var, aVar, i);
        j();
        if (a0Var.b() > 0 && !a0Var.f396g) {
            boolean z = i == 1;
            int e2 = e(vVar, a0Var, aVar.f370b);
            if (z) {
                while (e2 > 0) {
                    int i2 = aVar.f370b;
                    if (i2 <= 0) {
                        break;
                    }
                    int i3 = i2 - 1;
                    aVar.f370b = i3;
                    e2 = e(vVar, a0Var, i3);
                }
            } else {
                int b2 = a0Var.b() - 1;
                int i4 = aVar.f370b;
                while (i4 < b2) {
                    int i5 = i4 + 1;
                    int e3 = e(vVar, a0Var, i5);
                    if (e3 <= e2) {
                        break;
                    }
                    i4 = i5;
                    e2 = e3;
                }
                aVar.f370b = i4;
            }
        }
        b();
    }

    /* JADX WARN: Code restructure failed: missing block: B:59:0x00d6, code lost:
        if (r13 == (r2 > r15)) goto L50;
     */
    /* JADX WARN: Code restructure failed: missing block: B:72:0x00f6, code lost:
        if (r13 == (r2 > r7)) goto L51;
     */
    /* JADX WARN: Removed duplicated region for block: B:79:0x0107  */
    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public View onFocusSearchFailed(View view, int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        int childCount;
        int i2;
        int i3;
        View view2;
        View view3;
        int i4;
        int i5;
        boolean z;
        int i6;
        int i7;
        RecyclerView.v vVar2 = vVar;
        RecyclerView.a0 a0Var2 = a0Var;
        View findContainingItemView = findContainingItemView(view);
        View view4 = null;
        if (findContainingItemView == null) {
            return null;
        }
        b bVar = (b) findContainingItemView.getLayoutParams();
        int i8 = bVar.f365e;
        int i9 = bVar.f366f + i8;
        if (super.onFocusSearchFailed(view, i, vVar, a0Var) == null) {
            return null;
        }
        if ((convertFocusDirectionToLayoutDirection(i) == 1) != this.mShouldReverseLayout) {
            i3 = getChildCount() - 1;
            childCount = -1;
            i2 = -1;
        } else {
            childCount = getChildCount();
            i2 = 1;
            i3 = 0;
        }
        boolean z2 = this.mOrientation == 1 && isLayoutRTL();
        int d2 = d(vVar2, a0Var2, i3);
        int i10 = -1;
        int i11 = -1;
        int i12 = 0;
        int i13 = 0;
        int i14 = i3;
        View view5 = null;
        while (i14 != childCount) {
            int d3 = d(vVar2, a0Var2, i14);
            View childAt = getChildAt(i14);
            if (childAt == findContainingItemView) {
                break;
            }
            if (!childAt.hasFocusable() || d3 == d2) {
                b bVar2 = (b) childAt.getLayoutParams();
                int i15 = bVar2.f365e;
                view2 = findContainingItemView;
                int i16 = bVar2.f366f + i15;
                if (childAt.hasFocusable() && i15 == i8 && i16 == i9) {
                    return childAt;
                }
                if (!(childAt.hasFocusable() && view4 == null) && (childAt.hasFocusable() || view5 != null)) {
                    view3 = view5;
                    int min = Math.min(i16, i9) - Math.max(i15, i8);
                    if (childAt.hasFocusable()) {
                        if (min <= i12) {
                            if (min == i12) {
                            }
                        }
                    } else if (view4 == null) {
                        i4 = i12;
                        i5 = childCount;
                        if (isViewPartiallyVisible(childAt, false, true)) {
                            i6 = i13;
                            if (min > i6) {
                                i7 = i11;
                                if (z) {
                                    if (childAt.hasFocusable()) {
                                        i10 = bVar2.f365e;
                                        i11 = i7;
                                        i13 = i6;
                                        view5 = view3;
                                        view4 = childAt;
                                        i12 = Math.min(i16, i9) - Math.max(i15, i8);
                                    } else {
                                        int i17 = bVar2.f365e;
                                        i13 = Math.min(i16, i9) - Math.max(i15, i8);
                                        i11 = i17;
                                        i12 = i4;
                                        view5 = childAt;
                                    }
                                    i14 += i2;
                                    vVar2 = vVar;
                                    a0Var2 = a0Var;
                                    findContainingItemView = view2;
                                    childCount = i5;
                                }
                            } else {
                                if (min == i6) {
                                    i7 = i11;
                                } else {
                                    i7 = i11;
                                }
                                z = false;
                                if (z) {
                                }
                            }
                        }
                        i7 = i11;
                        i6 = i13;
                        z = false;
                        if (z) {
                        }
                    }
                    i4 = i12;
                    i5 = childCount;
                    i7 = i11;
                    i6 = i13;
                    z = false;
                    if (z) {
                    }
                } else {
                    view3 = view5;
                }
                i4 = i12;
                i5 = childCount;
                i7 = i11;
                i6 = i13;
                z = true;
                if (z) {
                }
            } else if (view4 != null) {
                break;
            } else {
                view2 = findContainingItemView;
                view3 = view5;
                i4 = i12;
                i5 = childCount;
                i7 = i11;
                i6 = i13;
            }
            i11 = i7;
            i13 = i6;
            i12 = i4;
            view5 = view3;
            i14 += i2;
            vVar2 = vVar;
            a0Var2 = a0Var;
            findContainingItemView = view2;
            childCount = i5;
        }
        return view4 != null ? view4 : view5;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onInitializeAccessibilityNodeInfoForItem(RecyclerView.v vVar, RecyclerView.a0 a0Var, View view, b.j.j.x.b bVar) {
        ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
        if (!(layoutParams instanceof b)) {
            super.onInitializeAccessibilityNodeInfoForItem(view, bVar);
            return;
        }
        b bVar2 = (b) layoutParams;
        int d2 = d(vVar, a0Var, bVar2.a());
        if (this.mOrientation == 0) {
            bVar.n(b.c.a(bVar2.f365e, bVar2.f366f, d2, 1, false, false));
        } else {
            bVar.n(b.c.a(d2, 1, bVar2.f365e, bVar2.f366f, false, false));
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsAdded(RecyclerView recyclerView, int i, int i2) {
        this.f363g.f367a.clear();
        this.f363g.f368b.clear();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsChanged(RecyclerView recyclerView) {
        this.f363g.f367a.clear();
        this.f363g.f368b.clear();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsMoved(RecyclerView recyclerView, int i, int i2, int i3) {
        this.f363g.f367a.clear();
        this.f363g.f368b.clear();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsRemoved(RecyclerView recyclerView, int i, int i2) {
        this.f363g.f367a.clear();
        this.f363g.f368b.clear();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onItemsUpdated(RecyclerView recyclerView, int i, int i2, Object obj) {
        this.f363g.f367a.clear();
        this.f363g.f368b.clear();
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutChildren(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (a0Var.f396g) {
            int childCount = getChildCount();
            for (int i = 0; i < childCount; i++) {
                b bVar = (b) getChildAt(i).getLayoutParams();
                int a2 = bVar.a();
                this.f361e.put(a2, bVar.f366f);
                this.f362f.put(a2, bVar.f365e);
            }
        }
        super.onLayoutChildren(vVar, a0Var);
        this.f361e.clear();
        this.f362f.clear();
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutCompleted(RecyclerView.a0 a0Var) {
        super.onLayoutCompleted(a0Var);
        this.f357a = false;
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int scrollHorizontallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        j();
        b();
        return super.scrollHorizontallyBy(i, vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public int scrollVerticallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        j();
        b();
        return super.scrollVerticallyBy(i, vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void setMeasuredDimension(Rect rect, int i, int i2) {
        int chooseSize;
        int chooseSize2;
        if (this.f359c == null) {
            super.setMeasuredDimension(rect, i, i2);
        }
        int paddingRight = getPaddingRight() + getPaddingLeft();
        int paddingBottom = getPaddingBottom() + getPaddingTop();
        if (this.mOrientation == 1) {
            chooseSize2 = RecyclerView.o.chooseSize(i2, rect.height() + paddingBottom, getMinimumHeight());
            int[] iArr = this.f359c;
            chooseSize = RecyclerView.o.chooseSize(i, iArr[iArr.length - 1] + paddingRight, getMinimumWidth());
        } else {
            chooseSize = RecyclerView.o.chooseSize(i, rect.width() + paddingRight, getMinimumWidth());
            int[] iArr2 = this.f359c;
            chooseSize2 = RecyclerView.o.chooseSize(i2, iArr2[iArr2.length - 1] + paddingBottom, getMinimumHeight());
        }
        setMeasuredDimension(chooseSize, chooseSize2);
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager
    public void setStackFromEnd(boolean z) {
        if (!z) {
            super.setStackFromEnd(false);
            return;
        }
        throw new UnsupportedOperationException("GridLayoutManager does not support stack from end. Consider using reverse layout");
    }

    @Override // androidx.recyclerview.widget.LinearLayoutManager, androidx.recyclerview.widget.RecyclerView.o
    public boolean supportsPredictiveItemAnimations() {
        return this.mPendingSavedState == null && !this.f357a;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        if (layoutParams instanceof ViewGroup.MarginLayoutParams) {
            return new b((ViewGroup.MarginLayoutParams) layoutParams);
        }
        return new b(layoutParams);
    }

    /* loaded from: classes.dex */
    public static class b extends RecyclerView.p {

        /* renamed from: e  reason: collision with root package name */
        public int f365e;

        /* renamed from: f  reason: collision with root package name */
        public int f366f;

        public b(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f365e = -1;
            this.f366f = 0;
        }

        public b(int i, int i2) {
            super(i, i2);
            this.f365e = -1;
            this.f366f = 0;
        }

        public b(ViewGroup.MarginLayoutParams marginLayoutParams) {
            super(marginLayoutParams);
            this.f365e = -1;
            this.f366f = 0;
        }

        public b(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f365e = -1;
            this.f366f = 0;
        }
    }

    public GridLayoutManager(Context context, int i) {
        super(context);
        this.f357a = false;
        this.f358b = -1;
        this.f361e = new SparseIntArray();
        this.f362f = new SparseIntArray();
        this.f363g = new a();
        this.f364h = new Rect();
        i(i);
    }

    public GridLayoutManager(Context context, int i, int i2, boolean z) {
        super(context, i2, z);
        this.f357a = false;
        this.f358b = -1;
        this.f361e = new SparseIntArray();
        this.f362f = new SparseIntArray();
        this.f363g = new a();
        this.f364h = new Rect();
        i(i);
    }
}