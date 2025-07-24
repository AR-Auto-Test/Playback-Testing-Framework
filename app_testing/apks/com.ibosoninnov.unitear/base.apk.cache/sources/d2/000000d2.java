package androidx.recyclerview.widget;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.PointF;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import android.view.accessibility.AccessibilityEvent;
import androidx.recyclerview.widget.RecyclerView;
import b.w.b.m;
import b.w.b.o;
import b.w.b.s;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.List;
import java.util.Objects;

/* loaded from: classes.dex */
public class LinearLayoutManager extends RecyclerView.o implements RecyclerView.z.b {
    public static final boolean DEBUG = false;
    public static final int HORIZONTAL = 0;
    public static final int INVALID_OFFSET = Integer.MIN_VALUE;
    private static final float MAX_SCROLL_FACTOR = 0.33333334f;
    private static final String TAG = "LinearLayoutManager";
    public static final int VERTICAL = 1;
    public final a mAnchorInfo;
    private int mInitialPrefetchItemCount;
    private boolean mLastStackFromEnd;
    private final b mLayoutChunkResult;
    private c mLayoutState;
    public int mOrientation;
    public s mOrientationHelper;
    public d mPendingSavedState;
    public int mPendingScrollPosition;
    public int mPendingScrollPositionOffset;
    private boolean mRecycleChildrenOnDetach;
    private int[] mReusableIntPair;
    private boolean mReverseLayout;
    public boolean mShouldReverseLayout;
    private boolean mSmoothScrollbarEnabled;
    private boolean mStackFromEnd;

    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public s f369a;

        /* renamed from: b  reason: collision with root package name */
        public int f370b;

        /* renamed from: c  reason: collision with root package name */
        public int f371c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f372d;

        /* renamed from: e  reason: collision with root package name */
        public boolean f373e;

        public a() {
            d();
        }

        public void a() {
            int k;
            if (this.f372d) {
                k = this.f369a.g();
            } else {
                k = this.f369a.k();
            }
            this.f371c = k;
        }

        public void b(View view, int i) {
            if (this.f372d) {
                this.f371c = this.f369a.m() + this.f369a.b(view);
            } else {
                this.f371c = this.f369a.e(view);
            }
            this.f370b = i;
        }

        public void c(View view, int i) {
            int m = this.f369a.m();
            if (m >= 0) {
                b(view, i);
                return;
            }
            this.f370b = i;
            if (this.f372d) {
                int g2 = (this.f369a.g() - m) - this.f369a.b(view);
                this.f371c = this.f369a.g() - g2;
                if (g2 > 0) {
                    int c2 = this.f371c - this.f369a.c(view);
                    int k = this.f369a.k();
                    int min = c2 - (Math.min(this.f369a.e(view) - k, 0) + k);
                    if (min < 0) {
                        this.f371c = Math.min(g2, -min) + this.f371c;
                        return;
                    }
                    return;
                }
                return;
            }
            int e2 = this.f369a.e(view);
            int k2 = e2 - this.f369a.k();
            this.f371c = e2;
            if (k2 > 0) {
                int g3 = (this.f369a.g() - Math.min(0, (this.f369a.g() - m) - this.f369a.b(view))) - (this.f369a.c(view) + e2);
                if (g3 < 0) {
                    this.f371c -= Math.min(k2, -g3);
                }
            }
        }

        public void d() {
            this.f370b = -1;
            this.f371c = Integer.MIN_VALUE;
            this.f372d = false;
            this.f373e = false;
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("AnchorInfo{mPosition=");
            x.append(this.f370b);
            x.append(", mCoordinate=");
            x.append(this.f371c);
            x.append(", mLayoutFromEnd=");
            x.append(this.f372d);
            x.append(", mValid=");
            x.append(this.f373e);
            x.append('}');
            return x.toString();
        }
    }

    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public int f374a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f375b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f376c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f377d;
    }

    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: b  reason: collision with root package name */
        public int f379b;

        /* renamed from: c  reason: collision with root package name */
        public int f380c;

        /* renamed from: d  reason: collision with root package name */
        public int f381d;

        /* renamed from: e  reason: collision with root package name */
        public int f382e;

        /* renamed from: f  reason: collision with root package name */
        public int f383f;

        /* renamed from: g  reason: collision with root package name */
        public int f384g;
        public int j;
        public boolean l;

        /* renamed from: a  reason: collision with root package name */
        public boolean f378a = true;

        /* renamed from: h  reason: collision with root package name */
        public int f385h = 0;
        public int i = 0;
        public List<RecyclerView.d0> k = null;

        public void a(View view) {
            int a2;
            int size = this.k.size();
            View view2 = null;
            int i = Integer.MAX_VALUE;
            for (int i2 = 0; i2 < size; i2++) {
                View view3 = this.k.get(i2).itemView;
                RecyclerView.p pVar = (RecyclerView.p) view3.getLayoutParams();
                if (view3 != view && !pVar.c() && (a2 = (pVar.a() - this.f381d) * this.f382e) >= 0 && a2 < i) {
                    view2 = view3;
                    if (a2 == 0) {
                        break;
                    }
                    i = a2;
                }
            }
            if (view2 == null) {
                this.f381d = -1;
            } else {
                this.f381d = ((RecyclerView.p) view2.getLayoutParams()).a();
            }
        }

        public boolean b(RecyclerView.a0 a0Var) {
            int i = this.f381d;
            return i >= 0 && i < a0Var.b();
        }

        public View c(RecyclerView.v vVar) {
            List<RecyclerView.d0> list = this.k;
            if (list != null) {
                int size = list.size();
                for (int i = 0; i < size; i++) {
                    View view = this.k.get(i).itemView;
                    RecyclerView.p pVar = (RecyclerView.p) view.getLayoutParams();
                    if (!pVar.c() && this.f381d == pVar.a()) {
                        a(view);
                        return view;
                    }
                }
                return null;
            }
            View view2 = vVar.k(this.f381d, false, RecyclerView.FOREVER_NS).itemView;
            this.f381d += this.f382e;
            return view2;
        }
    }

    @SuppressLint({"BanParcelableUsage"})
    /* loaded from: classes.dex */
    public static class d implements Parcelable {
        public static final Parcelable.Creator<d> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f386b;

        /* renamed from: c  reason: collision with root package name */
        public int f387c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f388d;

        /* loaded from: classes.dex */
        public static class a implements Parcelable.Creator<d> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public d createFromParcel(Parcel parcel) {
                return new d(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public d[] newArray(int i) {
                return new d[i];
            }
        }

        public d() {
        }

        public boolean a() {
            return this.f386b >= 0;
        }

        @Override // android.os.Parcelable
        public int describeContents() {
            return 0;
        }

        @Override // android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeInt(this.f386b);
            parcel.writeInt(this.f387c);
            parcel.writeInt(this.f388d ? 1 : 0);
        }

        public d(Parcel parcel) {
            this.f386b = parcel.readInt();
            this.f387c = parcel.readInt();
            this.f388d = parcel.readInt() == 1;
        }

        public d(d dVar) {
            this.f386b = dVar.f386b;
            this.f387c = dVar.f387c;
            this.f388d = dVar.f388d;
        }
    }

    public LinearLayoutManager(Context context) {
        this(context, 1, false);
    }

    private int computeScrollExtent(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        ensureLayoutState();
        return b.v.u.c.e(a0Var, this.mOrientationHelper, findFirstVisibleChildClosestToStart(!this.mSmoothScrollbarEnabled, true), findFirstVisibleChildClosestToEnd(!this.mSmoothScrollbarEnabled, true), this, this.mSmoothScrollbarEnabled);
    }

    private int computeScrollOffset(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        ensureLayoutState();
        return b.v.u.c.f(a0Var, this.mOrientationHelper, findFirstVisibleChildClosestToStart(!this.mSmoothScrollbarEnabled, true), findFirstVisibleChildClosestToEnd(!this.mSmoothScrollbarEnabled, true), this, this.mSmoothScrollbarEnabled, this.mShouldReverseLayout);
    }

    private int computeScrollRange(RecyclerView.a0 a0Var) {
        if (getChildCount() == 0) {
            return 0;
        }
        ensureLayoutState();
        return b.v.u.c.g(a0Var, this.mOrientationHelper, findFirstVisibleChildClosestToStart(!this.mSmoothScrollbarEnabled, true), findFirstVisibleChildClosestToEnd(!this.mSmoothScrollbarEnabled, true), this, this.mSmoothScrollbarEnabled);
    }

    private View findFirstPartiallyOrCompletelyInvisibleChild() {
        return findOnePartiallyOrCompletelyInvisibleChild(0, getChildCount());
    }

    private View findFirstReferenceChild(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return findReferenceChild(vVar, a0Var, 0, getChildCount(), a0Var.b());
    }

    private View findLastPartiallyOrCompletelyInvisibleChild() {
        return findOnePartiallyOrCompletelyInvisibleChild(getChildCount() - 1, -1);
    }

    private View findLastReferenceChild(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return findReferenceChild(vVar, a0Var, getChildCount() - 1, -1, a0Var.b());
    }

    private View findPartiallyOrCompletelyInvisibleChildClosestToEnd() {
        return this.mShouldReverseLayout ? findFirstPartiallyOrCompletelyInvisibleChild() : findLastPartiallyOrCompletelyInvisibleChild();
    }

    private View findPartiallyOrCompletelyInvisibleChildClosestToStart() {
        return this.mShouldReverseLayout ? findLastPartiallyOrCompletelyInvisibleChild() : findFirstPartiallyOrCompletelyInvisibleChild();
    }

    private View findReferenceChildClosestToEnd(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return this.mShouldReverseLayout ? findFirstReferenceChild(vVar, a0Var) : findLastReferenceChild(vVar, a0Var);
    }

    private View findReferenceChildClosestToStart(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        return this.mShouldReverseLayout ? findLastReferenceChild(vVar, a0Var) : findFirstReferenceChild(vVar, a0Var);
    }

    private int fixLayoutEndGap(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var, boolean z) {
        int g2;
        int g3 = this.mOrientationHelper.g() - i;
        if (g3 > 0) {
            int i2 = -scrollBy(-g3, vVar, a0Var);
            int i3 = i + i2;
            if (!z || (g2 = this.mOrientationHelper.g() - i3) <= 0) {
                return i2;
            }
            this.mOrientationHelper.p(g2);
            return g2 + i2;
        }
        return 0;
    }

    private int fixLayoutStartGap(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var, boolean z) {
        int k;
        int k2 = i - this.mOrientationHelper.k();
        if (k2 > 0) {
            int i2 = -scrollBy(k2, vVar, a0Var);
            int i3 = i + i2;
            if (!z || (k = i3 - this.mOrientationHelper.k()) <= 0) {
                return i2;
            }
            this.mOrientationHelper.p(-k);
            return i2 - k;
        }
        return 0;
    }

    private View getChildClosestToEnd() {
        return getChildAt(this.mShouldReverseLayout ? 0 : getChildCount() - 1);
    }

    private View getChildClosestToStart() {
        return getChildAt(this.mShouldReverseLayout ? getChildCount() - 1 : 0);
    }

    private void layoutForPredictiveAnimations(RecyclerView.v vVar, RecyclerView.a0 a0Var, int i, int i2) {
        if (!a0Var.k || getChildCount() == 0 || a0Var.f396g || !supportsPredictiveItemAnimations()) {
            return;
        }
        List<RecyclerView.d0> list = vVar.f437d;
        int size = list.size();
        int position = getPosition(getChildAt(0));
        int i3 = 0;
        int i4 = 0;
        for (int i5 = 0; i5 < size; i5++) {
            RecyclerView.d0 d0Var = list.get(i5);
            if (!d0Var.isRemoved()) {
                if ((d0Var.getLayoutPosition() < position) != this.mShouldReverseLayout ? true : true) {
                    i3 += this.mOrientationHelper.c(d0Var.itemView);
                } else {
                    i4 += this.mOrientationHelper.c(d0Var.itemView);
                }
            }
        }
        this.mLayoutState.k = list;
        if (i3 > 0) {
            updateLayoutStateToFillStart(getPosition(getChildClosestToStart()), i);
            c cVar = this.mLayoutState;
            cVar.f385h = i3;
            cVar.f380c = 0;
            cVar.a(null);
            fill(vVar, this.mLayoutState, a0Var, false);
        }
        if (i4 > 0) {
            updateLayoutStateToFillEnd(getPosition(getChildClosestToEnd()), i2);
            c cVar2 = this.mLayoutState;
            cVar2.f385h = i4;
            cVar2.f380c = 0;
            cVar2.a(null);
            fill(vVar, this.mLayoutState, a0Var, false);
        }
        this.mLayoutState.k = null;
    }

    private void logChildren() {
        Log.d(TAG, "internal representation of views on the screen");
        for (int i = 0; i < getChildCount(); i++) {
            View childAt = getChildAt(i);
            StringBuilder x = c.b.a.a.a.x("item ");
            x.append(getPosition(childAt));
            x.append(", coord:");
            x.append(this.mOrientationHelper.e(childAt));
            Log.d(TAG, x.toString());
        }
        Log.d(TAG, "==============");
    }

    private void recycleByLayoutState(RecyclerView.v vVar, c cVar) {
        if (!cVar.f378a || cVar.l) {
            return;
        }
        int i = cVar.f384g;
        int i2 = cVar.i;
        if (cVar.f383f == -1) {
            recycleViewsFromEnd(vVar, i, i2);
        } else {
            recycleViewsFromStart(vVar, i, i2);
        }
    }

    private void recycleChildren(RecyclerView.v vVar, int i, int i2) {
        if (i == i2) {
            return;
        }
        if (i2 <= i) {
            while (i > i2) {
                removeAndRecycleViewAt(i, vVar);
                i--;
            }
            return;
        }
        for (int i3 = i2 - 1; i3 >= i; i3--) {
            removeAndRecycleViewAt(i3, vVar);
        }
    }

    private void recycleViewsFromEnd(RecyclerView.v vVar, int i, int i2) {
        int childCount = getChildCount();
        if (i < 0) {
            return;
        }
        int f2 = (this.mOrientationHelper.f() - i) + i2;
        if (this.mShouldReverseLayout) {
            for (int i3 = 0; i3 < childCount; i3++) {
                View childAt = getChildAt(i3);
                if (this.mOrientationHelper.e(childAt) < f2 || this.mOrientationHelper.o(childAt) < f2) {
                    recycleChildren(vVar, 0, i3);
                    return;
                }
            }
            return;
        }
        int i4 = childCount - 1;
        for (int i5 = i4; i5 >= 0; i5--) {
            View childAt2 = getChildAt(i5);
            if (this.mOrientationHelper.e(childAt2) < f2 || this.mOrientationHelper.o(childAt2) < f2) {
                recycleChildren(vVar, i4, i5);
                return;
            }
        }
    }

    private void recycleViewsFromStart(RecyclerView.v vVar, int i, int i2) {
        if (i < 0) {
            return;
        }
        int i3 = i - i2;
        int childCount = getChildCount();
        if (!this.mShouldReverseLayout) {
            for (int i4 = 0; i4 < childCount; i4++) {
                View childAt = getChildAt(i4);
                if (this.mOrientationHelper.b(childAt) > i3 || this.mOrientationHelper.n(childAt) > i3) {
                    recycleChildren(vVar, 0, i4);
                    return;
                }
            }
            return;
        }
        int i5 = childCount - 1;
        for (int i6 = i5; i6 >= 0; i6--) {
            View childAt2 = getChildAt(i6);
            if (this.mOrientationHelper.b(childAt2) > i3 || this.mOrientationHelper.n(childAt2) > i3) {
                recycleChildren(vVar, i5, i6);
                return;
            }
        }
    }

    private void resolveShouldLayoutReverse() {
        if (this.mOrientation != 1 && isLayoutRTL()) {
            this.mShouldReverseLayout = !this.mReverseLayout;
        } else {
            this.mShouldReverseLayout = this.mReverseLayout;
        }
    }

    private boolean updateAnchorFromChildren(RecyclerView.v vVar, RecyclerView.a0 a0Var, a aVar) {
        View findReferenceChildClosestToStart;
        int k;
        boolean z = false;
        if (getChildCount() == 0) {
            return false;
        }
        View focusedChild = getFocusedChild();
        if (focusedChild != null) {
            Objects.requireNonNull(aVar);
            RecyclerView.p pVar = (RecyclerView.p) focusedChild.getLayoutParams();
            if (!pVar.c() && pVar.a() >= 0 && pVar.a() < a0Var.b()) {
                aVar.c(focusedChild, getPosition(focusedChild));
                return true;
            }
        }
        if (this.mLastStackFromEnd != this.mStackFromEnd) {
            return false;
        }
        if (aVar.f372d) {
            findReferenceChildClosestToStart = findReferenceChildClosestToEnd(vVar, a0Var);
        } else {
            findReferenceChildClosestToStart = findReferenceChildClosestToStart(vVar, a0Var);
        }
        if (findReferenceChildClosestToStart != null) {
            aVar.b(findReferenceChildClosestToStart, getPosition(findReferenceChildClosestToStart));
            if (!a0Var.f396g && supportsPredictiveItemAnimations()) {
                if (this.mOrientationHelper.e(findReferenceChildClosestToStart) >= this.mOrientationHelper.g() || this.mOrientationHelper.b(findReferenceChildClosestToStart) < this.mOrientationHelper.k()) {
                    z = true;
                }
                if (z) {
                    if (aVar.f372d) {
                        k = this.mOrientationHelper.g();
                    } else {
                        k = this.mOrientationHelper.k();
                    }
                    aVar.f371c = k;
                }
            }
            return true;
        }
        return false;
    }

    private boolean updateAnchorFromPendingData(RecyclerView.a0 a0Var, a aVar) {
        int i;
        int e2;
        if (!a0Var.f396g && (i = this.mPendingScrollPosition) != -1) {
            if (i >= 0 && i < a0Var.b()) {
                aVar.f370b = this.mPendingScrollPosition;
                d dVar = this.mPendingSavedState;
                if (dVar != null && dVar.a()) {
                    boolean z = this.mPendingSavedState.f388d;
                    aVar.f372d = z;
                    if (z) {
                        aVar.f371c = this.mOrientationHelper.g() - this.mPendingSavedState.f387c;
                    } else {
                        aVar.f371c = this.mOrientationHelper.k() + this.mPendingSavedState.f387c;
                    }
                    return true;
                } else if (this.mPendingScrollPositionOffset == Integer.MIN_VALUE) {
                    View findViewByPosition = findViewByPosition(this.mPendingScrollPosition);
                    if (findViewByPosition != null) {
                        if (this.mOrientationHelper.c(findViewByPosition) > this.mOrientationHelper.l()) {
                            aVar.a();
                            return true;
                        } else if (this.mOrientationHelper.e(findViewByPosition) - this.mOrientationHelper.k() < 0) {
                            aVar.f371c = this.mOrientationHelper.k();
                            aVar.f372d = false;
                            return true;
                        } else if (this.mOrientationHelper.g() - this.mOrientationHelper.b(findViewByPosition) < 0) {
                            aVar.f371c = this.mOrientationHelper.g();
                            aVar.f372d = true;
                            return true;
                        } else {
                            if (aVar.f372d) {
                                e2 = this.mOrientationHelper.m() + this.mOrientationHelper.b(findViewByPosition);
                            } else {
                                e2 = this.mOrientationHelper.e(findViewByPosition);
                            }
                            aVar.f371c = e2;
                        }
                    } else {
                        if (getChildCount() > 0) {
                            aVar.f372d = (this.mPendingScrollPosition < getPosition(getChildAt(0))) == this.mShouldReverseLayout;
                        }
                        aVar.a();
                    }
                    return true;
                } else {
                    boolean z2 = this.mShouldReverseLayout;
                    aVar.f372d = z2;
                    if (z2) {
                        aVar.f371c = this.mOrientationHelper.g() - this.mPendingScrollPositionOffset;
                    } else {
                        aVar.f371c = this.mOrientationHelper.k() + this.mPendingScrollPositionOffset;
                    }
                    return true;
                }
            }
            this.mPendingScrollPosition = -1;
            this.mPendingScrollPositionOffset = Integer.MIN_VALUE;
        }
        return false;
    }

    private void updateAnchorInfoForLayout(RecyclerView.v vVar, RecyclerView.a0 a0Var, a aVar) {
        if (updateAnchorFromPendingData(a0Var, aVar) || updateAnchorFromChildren(vVar, a0Var, aVar)) {
            return;
        }
        aVar.a();
        aVar.f370b = this.mStackFromEnd ? a0Var.b() - 1 : 0;
    }

    private void updateLayoutState(int i, int i2, boolean z, RecyclerView.a0 a0Var) {
        int k;
        this.mLayoutState.l = resolveIsInfinite();
        this.mLayoutState.f383f = i;
        int[] iArr = this.mReusableIntPair;
        iArr[0] = 0;
        iArr[1] = 0;
        calculateExtraLayoutSpace(a0Var, iArr);
        int max = Math.max(0, this.mReusableIntPair[0]);
        int max2 = Math.max(0, this.mReusableIntPair[1]);
        boolean z2 = i == 1;
        c cVar = this.mLayoutState;
        int i3 = z2 ? max2 : max;
        cVar.f385h = i3;
        if (!z2) {
            max = max2;
        }
        cVar.i = max;
        if (z2) {
            cVar.f385h = this.mOrientationHelper.h() + i3;
            View childClosestToEnd = getChildClosestToEnd();
            c cVar2 = this.mLayoutState;
            cVar2.f382e = this.mShouldReverseLayout ? -1 : 1;
            int position = getPosition(childClosestToEnd);
            c cVar3 = this.mLayoutState;
            cVar2.f381d = position + cVar3.f382e;
            cVar3.f379b = this.mOrientationHelper.b(childClosestToEnd);
            k = this.mOrientationHelper.b(childClosestToEnd) - this.mOrientationHelper.g();
        } else {
            View childClosestToStart = getChildClosestToStart();
            c cVar4 = this.mLayoutState;
            cVar4.f385h = this.mOrientationHelper.k() + cVar4.f385h;
            c cVar5 = this.mLayoutState;
            cVar5.f382e = this.mShouldReverseLayout ? 1 : -1;
            int position2 = getPosition(childClosestToStart);
            c cVar6 = this.mLayoutState;
            cVar5.f381d = position2 + cVar6.f382e;
            cVar6.f379b = this.mOrientationHelper.e(childClosestToStart);
            k = (-this.mOrientationHelper.e(childClosestToStart)) + this.mOrientationHelper.k();
        }
        c cVar7 = this.mLayoutState;
        cVar7.f380c = i2;
        if (z) {
            cVar7.f380c = i2 - k;
        }
        cVar7.f384g = k;
    }

    private void updateLayoutStateToFillEnd(a aVar) {
        updateLayoutStateToFillEnd(aVar.f370b, aVar.f371c);
    }

    private void updateLayoutStateToFillStart(a aVar) {
        updateLayoutStateToFillStart(aVar.f370b, aVar.f371c);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void assertNotInLayoutOrScroll(String str) {
        if (this.mPendingSavedState == null) {
            super.assertNotInLayoutOrScroll(str);
        }
    }

    public void calculateExtraLayoutSpace(RecyclerView.a0 a0Var, int[] iArr) {
        int i;
        int extraLayoutSpace = getExtraLayoutSpace(a0Var);
        if (this.mLayoutState.f383f == -1) {
            i = 0;
        } else {
            i = extraLayoutSpace;
            extraLayoutSpace = 0;
        }
        iArr[0] = extraLayoutSpace;
        iArr[1] = i;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean canScrollHorizontally() {
        return this.mOrientation == 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean canScrollVertically() {
        return this.mOrientation == 1;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void collectAdjacentPrefetchPositions(int i, int i2, RecyclerView.a0 a0Var, RecyclerView.o.c cVar) {
        if (this.mOrientation != 0) {
            i = i2;
        }
        if (getChildCount() == 0 || i == 0) {
            return;
        }
        ensureLayoutState();
        updateLayoutState(i > 0 ? 1 : -1, Math.abs(i), true, a0Var);
        collectPrefetchPositionsForLayoutState(a0Var, this.mLayoutState, cVar);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void collectInitialPrefetchPositions(int i, RecyclerView.o.c cVar) {
        boolean z;
        int i2;
        d dVar = this.mPendingSavedState;
        if (dVar != null && dVar.a()) {
            d dVar2 = this.mPendingSavedState;
            z = dVar2.f388d;
            i2 = dVar2.f386b;
        } else {
            resolveShouldLayoutReverse();
            z = this.mShouldReverseLayout;
            i2 = this.mPendingScrollPosition;
            if (i2 == -1) {
                i2 = z ? i - 1 : 0;
            }
        }
        int i3 = z ? -1 : 1;
        for (int i4 = 0; i4 < this.mInitialPrefetchItemCount && i2 >= 0 && i2 < i; i4++) {
            ((m.b) cVar).a(i2, 0);
            i2 += i3;
        }
    }

    public void collectPrefetchPositionsForLayoutState(RecyclerView.a0 a0Var, c cVar, RecyclerView.o.c cVar2) {
        int i = cVar.f381d;
        if (i < 0 || i >= a0Var.b()) {
            return;
        }
        ((m.b) cVar2).a(i, Math.max(0, cVar.f384g));
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollExtent(RecyclerView.a0 a0Var) {
        return computeScrollExtent(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollOffset(RecyclerView.a0 a0Var) {
        return computeScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeHorizontalScrollRange(RecyclerView.a0 a0Var) {
        return computeScrollRange(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.z.b
    public PointF computeScrollVectorForPosition(int i) {
        if (getChildCount() == 0) {
            return null;
        }
        int i2 = (i < getPosition(getChildAt(0))) != this.mShouldReverseLayout ? -1 : 1;
        if (this.mOrientation == 0) {
            return new PointF(i2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
        return new PointF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, i2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollExtent(RecyclerView.a0 a0Var) {
        return computeScrollExtent(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollOffset(RecyclerView.a0 a0Var) {
        return computeScrollOffset(a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int computeVerticalScrollRange(RecyclerView.a0 a0Var) {
        return computeScrollRange(a0Var);
    }

    public int convertFocusDirectionToLayoutDirection(int i) {
        return i != 1 ? i != 2 ? i != 17 ? i != 33 ? i != 66 ? (i == 130 && this.mOrientation == 1) ? 1 : Integer.MIN_VALUE : this.mOrientation == 0 ? 1 : Integer.MIN_VALUE : this.mOrientation == 1 ? -1 : Integer.MIN_VALUE : this.mOrientation == 0 ? -1 : Integer.MIN_VALUE : (this.mOrientation != 1 && isLayoutRTL()) ? -1 : 1 : (this.mOrientation != 1 && isLayoutRTL()) ? 1 : -1;
    }

    public c createLayoutState() {
        return new c();
    }

    public void ensureLayoutState() {
        if (this.mLayoutState == null) {
            this.mLayoutState = createLayoutState();
        }
    }

    public int fill(RecyclerView.v vVar, c cVar, RecyclerView.a0 a0Var, boolean z) {
        int i = cVar.f380c;
        int i2 = cVar.f384g;
        if (i2 != Integer.MIN_VALUE) {
            if (i < 0) {
                cVar.f384g = i2 + i;
            }
            recycleByLayoutState(vVar, cVar);
        }
        int i3 = cVar.f380c + cVar.f385h;
        b bVar = this.mLayoutChunkResult;
        while (true) {
            if ((!cVar.l && i3 <= 0) || !cVar.b(a0Var)) {
                break;
            }
            bVar.f374a = 0;
            bVar.f375b = false;
            bVar.f376c = false;
            bVar.f377d = false;
            layoutChunk(vVar, a0Var, cVar, bVar);
            if (!bVar.f375b) {
                int i4 = cVar.f379b;
                int i5 = bVar.f374a;
                cVar.f379b = (cVar.f383f * i5) + i4;
                if (!bVar.f376c || cVar.k != null || !a0Var.f396g) {
                    cVar.f380c -= i5;
                    i3 -= i5;
                }
                int i6 = cVar.f384g;
                if (i6 != Integer.MIN_VALUE) {
                    int i7 = i6 + i5;
                    cVar.f384g = i7;
                    int i8 = cVar.f380c;
                    if (i8 < 0) {
                        cVar.f384g = i7 + i8;
                    }
                    recycleByLayoutState(vVar, cVar);
                }
                if (z && bVar.f377d) {
                    break;
                }
            } else {
                break;
            }
        }
        return i - cVar.f380c;
    }

    public int findFirstCompletelyVisibleItemPosition() {
        View findOneVisibleChild = findOneVisibleChild(0, getChildCount(), true, false);
        if (findOneVisibleChild == null) {
            return -1;
        }
        return getPosition(findOneVisibleChild);
    }

    public View findFirstVisibleChildClosestToEnd(boolean z, boolean z2) {
        if (this.mShouldReverseLayout) {
            return findOneVisibleChild(0, getChildCount(), z, z2);
        }
        return findOneVisibleChild(getChildCount() - 1, -1, z, z2);
    }

    public View findFirstVisibleChildClosestToStart(boolean z, boolean z2) {
        if (this.mShouldReverseLayout) {
            return findOneVisibleChild(getChildCount() - 1, -1, z, z2);
        }
        return findOneVisibleChild(0, getChildCount(), z, z2);
    }

    public int findFirstVisibleItemPosition() {
        View findOneVisibleChild = findOneVisibleChild(0, getChildCount(), false, true);
        if (findOneVisibleChild == null) {
            return -1;
        }
        return getPosition(findOneVisibleChild);
    }

    public int findLastCompletelyVisibleItemPosition() {
        View findOneVisibleChild = findOneVisibleChild(getChildCount() - 1, -1, true, false);
        if (findOneVisibleChild == null) {
            return -1;
        }
        return getPosition(findOneVisibleChild);
    }

    public int findLastVisibleItemPosition() {
        View findOneVisibleChild = findOneVisibleChild(getChildCount() - 1, -1, false, true);
        if (findOneVisibleChild == null) {
            return -1;
        }
        return getPosition(findOneVisibleChild);
    }

    public View findOnePartiallyOrCompletelyInvisibleChild(int i, int i2) {
        int i3;
        int i4;
        ensureLayoutState();
        if ((i2 > i ? (char) 1 : i2 < i ? (char) 65535 : (char) 0) == 0) {
            return getChildAt(i);
        }
        if (this.mOrientationHelper.e(getChildAt(i)) < this.mOrientationHelper.k()) {
            i3 = 16644;
            i4 = 16388;
        } else {
            i3 = 4161;
            i4 = 4097;
        }
        if (this.mOrientation == 0) {
            return this.mHorizontalBoundCheck.a(i, i2, i3, i4);
        }
        return this.mVerticalBoundCheck.a(i, i2, i3, i4);
    }

    public View findOneVisibleChild(int i, int i2, boolean z, boolean z2) {
        ensureLayoutState();
        int i3 = z ? 24579 : 320;
        int i4 = z2 ? 320 : 0;
        if (this.mOrientation == 0) {
            return this.mHorizontalBoundCheck.a(i, i2, i3, i4);
        }
        return this.mVerticalBoundCheck.a(i, i2, i3, i4);
    }

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
            if (position >= 0 && position < i3) {
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

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public View findViewByPosition(int i) {
        int childCount = getChildCount();
        if (childCount == 0) {
            return null;
        }
        int position = i - getPosition(getChildAt(0));
        if (position >= 0 && position < childCount) {
            View childAt = getChildAt(position);
            if (getPosition(childAt) == i) {
                return childAt;
            }
        }
        return super.findViewByPosition(i);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public RecyclerView.p generateDefaultLayoutParams() {
        return new RecyclerView.p(-2, -2);
    }

    @Deprecated
    public int getExtraLayoutSpace(RecyclerView.a0 a0Var) {
        if (a0Var.f390a != -1) {
            return this.mOrientationHelper.l();
        }
        return 0;
    }

    public int getInitialPrefetchItemCount() {
        return this.mInitialPrefetchItemCount;
    }

    public int getOrientation() {
        return this.mOrientation;
    }

    public boolean getRecycleChildrenOnDetach() {
        return this.mRecycleChildrenOnDetach;
    }

    public boolean getReverseLayout() {
        return this.mReverseLayout;
    }

    public boolean getStackFromEnd() {
        return this.mStackFromEnd;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean isAutoMeasureEnabled() {
        return true;
    }

    public boolean isLayoutRTL() {
        return getLayoutDirection() == 1;
    }

    public boolean isSmoothScrollbarEnabled() {
        return this.mSmoothScrollbarEnabled;
    }

    public void layoutChunk(RecyclerView.v vVar, RecyclerView.a0 a0Var, c cVar, b bVar) {
        int i;
        int i2;
        int i3;
        int i4;
        int d2;
        View c2 = cVar.c(vVar);
        if (c2 == null) {
            bVar.f375b = true;
            return;
        }
        RecyclerView.p pVar = (RecyclerView.p) c2.getLayoutParams();
        if (cVar.k == null) {
            if (this.mShouldReverseLayout == (cVar.f383f == -1)) {
                addView(c2);
            } else {
                addView(c2, 0);
            }
        } else {
            if (this.mShouldReverseLayout == (cVar.f383f == -1)) {
                addDisappearingView(c2);
            } else {
                addDisappearingView(c2, 0);
            }
        }
        measureChildWithMargins(c2, 0, 0);
        bVar.f374a = this.mOrientationHelper.c(c2);
        if (this.mOrientation == 1) {
            if (isLayoutRTL()) {
                d2 = getWidth() - getPaddingRight();
                i4 = d2 - this.mOrientationHelper.d(c2);
            } else {
                i4 = getPaddingLeft();
                d2 = this.mOrientationHelper.d(c2) + i4;
            }
            if (cVar.f383f == -1) {
                int i5 = cVar.f379b;
                i3 = i5;
                i2 = d2;
                i = i5 - bVar.f374a;
            } else {
                int i6 = cVar.f379b;
                i = i6;
                i2 = d2;
                i3 = bVar.f374a + i6;
            }
        } else {
            int paddingTop = getPaddingTop();
            int d3 = this.mOrientationHelper.d(c2) + paddingTop;
            if (cVar.f383f == -1) {
                int i7 = cVar.f379b;
                i2 = i7;
                i = paddingTop;
                i3 = d3;
                i4 = i7 - bVar.f374a;
            } else {
                int i8 = cVar.f379b;
                i = paddingTop;
                i2 = bVar.f374a + i8;
                i3 = d3;
                i4 = i8;
            }
        }
        layoutDecoratedWithMargins(c2, i4, i, i2, i3);
        if (pVar.c() || pVar.b()) {
            bVar.f376c = true;
        }
        bVar.f377d = c2.hasFocusable();
    }

    public void onAnchorReady(RecyclerView.v vVar, RecyclerView.a0 a0Var, a aVar, int i) {
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onDetachedFromWindow(RecyclerView recyclerView, RecyclerView.v vVar) {
        super.onDetachedFromWindow(recyclerView, vVar);
        if (this.mRecycleChildrenOnDetach) {
            removeAndRecycleAllViews(vVar);
            vVar.b();
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public View onFocusSearchFailed(View view, int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        int convertFocusDirectionToLayoutDirection;
        View findPartiallyOrCompletelyInvisibleChildClosestToEnd;
        View childClosestToEnd;
        resolveShouldLayoutReverse();
        if (getChildCount() == 0 || (convertFocusDirectionToLayoutDirection = convertFocusDirectionToLayoutDirection(i)) == Integer.MIN_VALUE) {
            return null;
        }
        ensureLayoutState();
        updateLayoutState(convertFocusDirectionToLayoutDirection, (int) (this.mOrientationHelper.l() * MAX_SCROLL_FACTOR), false, a0Var);
        c cVar = this.mLayoutState;
        cVar.f384g = Integer.MIN_VALUE;
        cVar.f378a = false;
        fill(vVar, cVar, a0Var, true);
        if (convertFocusDirectionToLayoutDirection == -1) {
            findPartiallyOrCompletelyInvisibleChildClosestToEnd = findPartiallyOrCompletelyInvisibleChildClosestToStart();
        } else {
            findPartiallyOrCompletelyInvisibleChildClosestToEnd = findPartiallyOrCompletelyInvisibleChildClosestToEnd();
        }
        if (convertFocusDirectionToLayoutDirection == -1) {
            childClosestToEnd = getChildClosestToStart();
        } else {
            childClosestToEnd = getChildClosestToEnd();
        }
        if (childClosestToEnd.hasFocusable()) {
            if (findPartiallyOrCompletelyInvisibleChildClosestToEnd == null) {
                return null;
            }
            return childClosestToEnd;
        }
        return findPartiallyOrCompletelyInvisibleChildClosestToEnd;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onInitializeAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(accessibilityEvent);
        if (getChildCount() > 0) {
            accessibilityEvent.setFromIndex(findFirstVisibleItemPosition());
            accessibilityEvent.setToIndex(findLastVisibleItemPosition());
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutChildren(RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        int i;
        int i2;
        int i3;
        int i4;
        int fixLayoutEndGap;
        int i5;
        View findViewByPosition;
        int e2;
        int i6;
        int i7 = -1;
        if ((this.mPendingSavedState != null || this.mPendingScrollPosition != -1) && a0Var.b() == 0) {
            removeAndRecycleAllViews(vVar);
            return;
        }
        d dVar = this.mPendingSavedState;
        if (dVar != null && dVar.a()) {
            this.mPendingScrollPosition = this.mPendingSavedState.f386b;
        }
        ensureLayoutState();
        this.mLayoutState.f378a = false;
        resolveShouldLayoutReverse();
        View focusedChild = getFocusedChild();
        a aVar = this.mAnchorInfo;
        if (aVar.f373e && this.mPendingScrollPosition == -1 && this.mPendingSavedState == null) {
            if (focusedChild != null && (this.mOrientationHelper.e(focusedChild) >= this.mOrientationHelper.g() || this.mOrientationHelper.b(focusedChild) <= this.mOrientationHelper.k())) {
                this.mAnchorInfo.c(focusedChild, getPosition(focusedChild));
            }
        } else {
            aVar.d();
            a aVar2 = this.mAnchorInfo;
            aVar2.f372d = this.mShouldReverseLayout ^ this.mStackFromEnd;
            updateAnchorInfoForLayout(vVar, a0Var, aVar2);
            this.mAnchorInfo.f373e = true;
        }
        c cVar = this.mLayoutState;
        cVar.f383f = cVar.j >= 0 ? 1 : -1;
        int[] iArr = this.mReusableIntPair;
        iArr[0] = 0;
        iArr[1] = 0;
        calculateExtraLayoutSpace(a0Var, iArr);
        int k = this.mOrientationHelper.k() + Math.max(0, this.mReusableIntPair[0]);
        int h2 = this.mOrientationHelper.h() + Math.max(0, this.mReusableIntPair[1]);
        if (a0Var.f396g && (i5 = this.mPendingScrollPosition) != -1 && this.mPendingScrollPositionOffset != Integer.MIN_VALUE && (findViewByPosition = findViewByPosition(i5)) != null) {
            if (this.mShouldReverseLayout) {
                i6 = this.mOrientationHelper.g() - this.mOrientationHelper.b(findViewByPosition);
                e2 = this.mPendingScrollPositionOffset;
            } else {
                e2 = this.mOrientationHelper.e(findViewByPosition) - this.mOrientationHelper.k();
                i6 = this.mPendingScrollPositionOffset;
            }
            int i8 = i6 - e2;
            if (i8 > 0) {
                k += i8;
            } else {
                h2 -= i8;
            }
        }
        a aVar3 = this.mAnchorInfo;
        if (!aVar3.f372d ? !this.mShouldReverseLayout : this.mShouldReverseLayout) {
            i7 = 1;
        }
        onAnchorReady(vVar, a0Var, aVar3, i7);
        detachAndScrapAttachedViews(vVar);
        this.mLayoutState.l = resolveIsInfinite();
        Objects.requireNonNull(this.mLayoutState);
        this.mLayoutState.i = 0;
        a aVar4 = this.mAnchorInfo;
        if (aVar4.f372d) {
            updateLayoutStateToFillStart(aVar4);
            c cVar2 = this.mLayoutState;
            cVar2.f385h = k;
            fill(vVar, cVar2, a0Var, false);
            c cVar3 = this.mLayoutState;
            i2 = cVar3.f379b;
            int i9 = cVar3.f381d;
            int i10 = cVar3.f380c;
            if (i10 > 0) {
                h2 += i10;
            }
            updateLayoutStateToFillEnd(this.mAnchorInfo);
            c cVar4 = this.mLayoutState;
            cVar4.f385h = h2;
            cVar4.f381d += cVar4.f382e;
            fill(vVar, cVar4, a0Var, false);
            c cVar5 = this.mLayoutState;
            i = cVar5.f379b;
            int i11 = cVar5.f380c;
            if (i11 > 0) {
                updateLayoutStateToFillStart(i9, i2);
                c cVar6 = this.mLayoutState;
                cVar6.f385h = i11;
                fill(vVar, cVar6, a0Var, false);
                i2 = this.mLayoutState.f379b;
            }
        } else {
            updateLayoutStateToFillEnd(aVar4);
            c cVar7 = this.mLayoutState;
            cVar7.f385h = h2;
            fill(vVar, cVar7, a0Var, false);
            c cVar8 = this.mLayoutState;
            i = cVar8.f379b;
            int i12 = cVar8.f381d;
            int i13 = cVar8.f380c;
            if (i13 > 0) {
                k += i13;
            }
            updateLayoutStateToFillStart(this.mAnchorInfo);
            c cVar9 = this.mLayoutState;
            cVar9.f385h = k;
            cVar9.f381d += cVar9.f382e;
            fill(vVar, cVar9, a0Var, false);
            c cVar10 = this.mLayoutState;
            int i14 = cVar10.f379b;
            int i15 = cVar10.f380c;
            if (i15 > 0) {
                updateLayoutStateToFillEnd(i12, i);
                c cVar11 = this.mLayoutState;
                cVar11.f385h = i15;
                fill(vVar, cVar11, a0Var, false);
                i = this.mLayoutState.f379b;
            }
            i2 = i14;
        }
        if (getChildCount() > 0) {
            if (this.mShouldReverseLayout ^ this.mStackFromEnd) {
                int fixLayoutEndGap2 = fixLayoutEndGap(i, vVar, a0Var, true);
                i3 = i2 + fixLayoutEndGap2;
                i4 = i + fixLayoutEndGap2;
                fixLayoutEndGap = fixLayoutStartGap(i3, vVar, a0Var, false);
            } else {
                int fixLayoutStartGap = fixLayoutStartGap(i2, vVar, a0Var, true);
                i3 = i2 + fixLayoutStartGap;
                i4 = i + fixLayoutStartGap;
                fixLayoutEndGap = fixLayoutEndGap(i4, vVar, a0Var, false);
            }
            i2 = i3 + fixLayoutEndGap;
            i = i4 + fixLayoutEndGap;
        }
        layoutForPredictiveAnimations(vVar, a0Var, i2, i);
        if (!a0Var.f396g) {
            s sVar = this.mOrientationHelper;
            sVar.f2795b = sVar.l();
        } else {
            this.mAnchorInfo.d();
        }
        this.mLastStackFromEnd = this.mStackFromEnd;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onLayoutCompleted(RecyclerView.a0 a0Var) {
        super.onLayoutCompleted(a0Var);
        this.mPendingSavedState = null;
        this.mPendingScrollPosition = -1;
        this.mPendingScrollPositionOffset = Integer.MIN_VALUE;
        this.mAnchorInfo.d();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (parcelable instanceof d) {
            this.mPendingSavedState = (d) parcelable;
            requestLayout();
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public Parcelable onSaveInstanceState() {
        d dVar = this.mPendingSavedState;
        if (dVar != null) {
            return new d(dVar);
        }
        d dVar2 = new d();
        if (getChildCount() > 0) {
            ensureLayoutState();
            boolean z = this.mLastStackFromEnd ^ this.mShouldReverseLayout;
            dVar2.f388d = z;
            if (z) {
                View childClosestToEnd = getChildClosestToEnd();
                dVar2.f387c = this.mOrientationHelper.g() - this.mOrientationHelper.b(childClosestToEnd);
                dVar2.f386b = getPosition(childClosestToEnd);
            } else {
                View childClosestToStart = getChildClosestToStart();
                dVar2.f386b = getPosition(childClosestToStart);
                dVar2.f387c = this.mOrientationHelper.e(childClosestToStart) - this.mOrientationHelper.k();
            }
        } else {
            dVar2.f386b = -1;
        }
        return dVar2;
    }

    public void prepareForDrop(View view, View view2, int i, int i2) {
        assertNotInLayoutOrScroll("Cannot drop a view during a scroll or layout calculation");
        ensureLayoutState();
        resolveShouldLayoutReverse();
        int position = getPosition(view);
        int position2 = getPosition(view2);
        boolean z = position < position2 ? true : true;
        if (this.mShouldReverseLayout) {
            if (z) {
                scrollToPositionWithOffset(position2, this.mOrientationHelper.g() - (this.mOrientationHelper.c(view) + this.mOrientationHelper.e(view2)));
            } else {
                scrollToPositionWithOffset(position2, this.mOrientationHelper.g() - this.mOrientationHelper.b(view2));
            }
        } else if (z) {
            scrollToPositionWithOffset(position2, this.mOrientationHelper.e(view2));
        } else {
            scrollToPositionWithOffset(position2, this.mOrientationHelper.b(view2) - this.mOrientationHelper.c(view));
        }
    }

    public boolean resolveIsInfinite() {
        return this.mOrientationHelper.i() == 0 && this.mOrientationHelper.f() == 0;
    }

    public int scrollBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (getChildCount() == 0 || i == 0) {
            return 0;
        }
        ensureLayoutState();
        this.mLayoutState.f378a = true;
        int i2 = i > 0 ? 1 : -1;
        int abs = Math.abs(i);
        updateLayoutState(i2, abs, true, a0Var);
        c cVar = this.mLayoutState;
        int fill = fill(vVar, cVar, a0Var, false) + cVar.f384g;
        if (fill < 0) {
            return 0;
        }
        if (abs > fill) {
            i = i2 * fill;
        }
        this.mOrientationHelper.p(-i);
        this.mLayoutState.j = i;
        return i;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int scrollHorizontallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.mOrientation == 1) {
            return 0;
        }
        return scrollBy(i, vVar, a0Var);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void scrollToPosition(int i) {
        this.mPendingScrollPosition = i;
        this.mPendingScrollPositionOffset = Integer.MIN_VALUE;
        d dVar = this.mPendingSavedState;
        if (dVar != null) {
            dVar.f386b = -1;
        }
        requestLayout();
    }

    public void scrollToPositionWithOffset(int i, int i2) {
        this.mPendingScrollPosition = i;
        this.mPendingScrollPositionOffset = i2;
        d dVar = this.mPendingSavedState;
        if (dVar != null) {
            dVar.f386b = -1;
        }
        requestLayout();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public int scrollVerticallyBy(int i, RecyclerView.v vVar, RecyclerView.a0 a0Var) {
        if (this.mOrientation == 0) {
            return 0;
        }
        return scrollBy(i, vVar, a0Var);
    }

    public void setInitialPrefetchItemCount(int i) {
        this.mInitialPrefetchItemCount = i;
    }

    public void setOrientation(int i) {
        if (i != 0 && i != 1) {
            throw new IllegalArgumentException(c.b.a.a.a.j("invalid orientation:", i));
        }
        assertNotInLayoutOrScroll(null);
        if (i != this.mOrientation || this.mOrientationHelper == null) {
            s a2 = s.a(this, i);
            this.mOrientationHelper = a2;
            this.mAnchorInfo.f369a = a2;
            this.mOrientation = i;
            requestLayout();
        }
    }

    public void setRecycleChildrenOnDetach(boolean z) {
        this.mRecycleChildrenOnDetach = z;
    }

    public void setReverseLayout(boolean z) {
        assertNotInLayoutOrScroll(null);
        if (z == this.mReverseLayout) {
            return;
        }
        this.mReverseLayout = z;
        requestLayout();
    }

    public void setSmoothScrollbarEnabled(boolean z) {
        this.mSmoothScrollbarEnabled = z;
    }

    public void setStackFromEnd(boolean z) {
        assertNotInLayoutOrScroll(null);
        if (this.mStackFromEnd == z) {
            return;
        }
        this.mStackFromEnd = z;
        requestLayout();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean shouldMeasureTwice() {
        return (getHeightMode() == 1073741824 || getWidthMode() == 1073741824 || !hasFlexibleChildInBothOrientations()) ? false : true;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public void smoothScrollToPosition(RecyclerView recyclerView, RecyclerView.a0 a0Var, int i) {
        o oVar = new o(recyclerView.getContext());
        oVar.setTargetPosition(i);
        startSmoothScroll(oVar);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.o
    public boolean supportsPredictiveItemAnimations() {
        return this.mPendingSavedState == null && this.mLastStackFromEnd == this.mStackFromEnd;
    }

    public void validateChildOrder() {
        StringBuilder x = c.b.a.a.a.x("validating child count ");
        x.append(getChildCount());
        Log.d(TAG, x.toString());
        if (getChildCount() < 1) {
            return;
        }
        int position = getPosition(getChildAt(0));
        int e2 = this.mOrientationHelper.e(getChildAt(0));
        if (this.mShouldReverseLayout) {
            for (int i = 1; i < getChildCount(); i++) {
                View childAt = getChildAt(i);
                int position2 = getPosition(childAt);
                int e3 = this.mOrientationHelper.e(childAt);
                if (position2 < position) {
                    logChildren();
                    StringBuilder x2 = c.b.a.a.a.x("detected invalid position. loc invalid? ");
                    x2.append(e3 < e2);
                    throw new RuntimeException(x2.toString());
                } else if (e3 > e2) {
                    logChildren();
                    throw new RuntimeException("detected invalid location");
                }
            }
            return;
        }
        for (int i2 = 1; i2 < getChildCount(); i2++) {
            View childAt2 = getChildAt(i2);
            int position3 = getPosition(childAt2);
            int e4 = this.mOrientationHelper.e(childAt2);
            if (position3 < position) {
                logChildren();
                StringBuilder x3 = c.b.a.a.a.x("detected invalid position. loc invalid? ");
                x3.append(e4 < e2);
                throw new RuntimeException(x3.toString());
            } else if (e4 < e2) {
                logChildren();
                throw new RuntimeException("detected invalid location");
            }
        }
    }

    public LinearLayoutManager(Context context, int i, boolean z) {
        this.mOrientation = 1;
        this.mReverseLayout = false;
        this.mShouldReverseLayout = false;
        this.mStackFromEnd = false;
        this.mSmoothScrollbarEnabled = true;
        this.mPendingScrollPosition = -1;
        this.mPendingScrollPositionOffset = Integer.MIN_VALUE;
        this.mPendingSavedState = null;
        this.mAnchorInfo = new a();
        this.mLayoutChunkResult = new b();
        this.mInitialPrefetchItemCount = 2;
        this.mReusableIntPair = new int[2];
        setOrientation(i);
        setReverseLayout(z);
    }

    private void updateLayoutStateToFillEnd(int i, int i2) {
        this.mLayoutState.f380c = this.mOrientationHelper.g() - i2;
        c cVar = this.mLayoutState;
        cVar.f382e = this.mShouldReverseLayout ? -1 : 1;
        cVar.f381d = i;
        cVar.f383f = 1;
        cVar.f379b = i2;
        cVar.f384g = Integer.MIN_VALUE;
    }

    private void updateLayoutStateToFillStart(int i, int i2) {
        this.mLayoutState.f380c = i2 - this.mOrientationHelper.k();
        c cVar = this.mLayoutState;
        cVar.f381d = i;
        cVar.f382e = this.mShouldReverseLayout ? 1 : -1;
        cVar.f383f = -1;
        cVar.f379b = i2;
        cVar.f384g = Integer.MIN_VALUE;
    }

    public LinearLayoutManager(Context context, AttributeSet attributeSet, int i, int i2) {
        this.mOrientation = 1;
        this.mReverseLayout = false;
        this.mShouldReverseLayout = false;
        this.mStackFromEnd = false;
        this.mSmoothScrollbarEnabled = true;
        this.mPendingScrollPosition = -1;
        this.mPendingScrollPositionOffset = Integer.MIN_VALUE;
        this.mPendingSavedState = null;
        this.mAnchorInfo = new a();
        this.mLayoutChunkResult = new b();
        this.mInitialPrefetchItemCount = 2;
        this.mReusableIntPair = new int[2];
        RecyclerView.o.d properties = RecyclerView.o.getProperties(context, attributeSet, i, i2);
        setOrientation(properties.f420a);
        setReverseLayout(properties.f422c);
        setStackFromEnd(properties.f423d);
    }
}