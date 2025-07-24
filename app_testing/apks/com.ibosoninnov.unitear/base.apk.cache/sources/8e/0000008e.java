package androidx.core.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.FocusFinder;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.accessibility.AccessibilityEvent;
import android.view.animation.AnimationUtils;
import android.widget.EdgeEffect;
import android.widget.FrameLayout;
import android.widget.OverScroller;
import android.widget.ScrollView;
import androidx.appcompat.app.AlertController;
import b.j.j.e;
import b.j.j.f;
import b.j.j.h;
import b.j.j.i;
import b.j.j.q;
import b.j.j.x.b;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class NestedScrollView extends FrameLayout implements h, e {

    /* renamed from: b  reason: collision with root package name */
    public static final a f246b = new a();

    /* renamed from: c  reason: collision with root package name */
    public static final int[] f247c = {16843130};
    public final f A;
    public float B;
    public b C;

    /* renamed from: d  reason: collision with root package name */
    public long f248d;

    /* renamed from: e  reason: collision with root package name */
    public final Rect f249e;

    /* renamed from: f  reason: collision with root package name */
    public OverScroller f250f;

    /* renamed from: g  reason: collision with root package name */
    public EdgeEffect f251g;

    /* renamed from: h  reason: collision with root package name */
    public EdgeEffect f252h;
    public int i;
    public boolean j;
    public boolean k;
    public View l;
    public boolean m;
    public VelocityTracker n;
    public boolean o;
    public boolean p;
    public int q;
    public int r;
    public int s;
    public int t;
    public final int[] u;
    public final int[] v;
    public int w;
    public int x;
    public c y;
    public final i z;

    /* loaded from: classes.dex */
    public static class a extends b.j.j.a {
        @Override // b.j.j.a
        public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            super.onInitializeAccessibilityEvent(view, accessibilityEvent);
            NestedScrollView nestedScrollView = (NestedScrollView) view;
            accessibilityEvent.setClassName(ScrollView.class.getName());
            accessibilityEvent.setScrollable(nestedScrollView.getScrollRange() > 0);
            accessibilityEvent.setScrollX(nestedScrollView.getScrollX());
            accessibilityEvent.setScrollY(nestedScrollView.getScrollY());
            accessibilityEvent.setMaxScrollX(nestedScrollView.getScrollX());
            accessibilityEvent.setMaxScrollY(nestedScrollView.getScrollRange());
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
            int scrollRange;
            super.onInitializeAccessibilityNodeInfo(view, bVar);
            NestedScrollView nestedScrollView = (NestedScrollView) view;
            bVar.f2259b.setClassName(ScrollView.class.getName());
            if (!nestedScrollView.isEnabled() || (scrollRange = nestedScrollView.getScrollRange()) <= 0) {
                return;
            }
            bVar.f2259b.setScrollable(true);
            if (nestedScrollView.getScrollY() > 0) {
                bVar.a(b.a.f2266e);
                bVar.a(b.a.i);
            }
            if (nestedScrollView.getScrollY() < scrollRange) {
                bVar.a(b.a.f2265d);
                bVar.a(b.a.j);
            }
        }

        @Override // b.j.j.a
        public boolean performAccessibilityAction(View view, int i, Bundle bundle) {
            if (super.performAccessibilityAction(view, i, bundle)) {
                return true;
            }
            NestedScrollView nestedScrollView = (NestedScrollView) view;
            if (nestedScrollView.isEnabled()) {
                if (i != 4096) {
                    if (i == 8192 || i == 16908344) {
                        int max = Math.max(nestedScrollView.getScrollY() - ((nestedScrollView.getHeight() - nestedScrollView.getPaddingBottom()) - nestedScrollView.getPaddingTop()), 0);
                        if (max != nestedScrollView.getScrollY()) {
                            nestedScrollView.v(0 - nestedScrollView.getScrollX(), max - nestedScrollView.getScrollY(), BaseTransientBottomBar.ANIMATION_DURATION, true);
                            return true;
                        }
                        return false;
                    } else if (i != 16908346) {
                        return false;
                    }
                }
                int min = Math.min(nestedScrollView.getScrollY() + ((nestedScrollView.getHeight() - nestedScrollView.getPaddingBottom()) - nestedScrollView.getPaddingTop()), nestedScrollView.getScrollRange());
                if (min != nestedScrollView.getScrollY()) {
                    nestedScrollView.v(0 - nestedScrollView.getScrollX(), min - nestedScrollView.getScrollY(), BaseTransientBottomBar.ANIMATION_DURATION, true);
                    return true;
                }
                return false;
            }
            return false;
        }
    }

    /* loaded from: classes.dex */
    public interface b {
    }

    /* loaded from: classes.dex */
    public static class c extends View.BaseSavedState {
        public static final Parcelable.Creator<c> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f253b;

        /* loaded from: classes.dex */
        public class a implements Parcelable.Creator<c> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public c createFromParcel(Parcel parcel) {
                return new c(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public c[] newArray(int i) {
                return new c[i];
            }
        }

        public c(Parcelable parcelable) {
            super(parcelable);
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("HorizontalScrollView.SavedState{");
            x.append(Integer.toHexString(System.identityHashCode(this)));
            x.append(" scrollPosition=");
            return c.b.a.a.a.s(x, this.f253b, "}");
        }

        @Override // android.view.View.BaseSavedState, android.view.AbsSavedState, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeInt(this.f253b);
        }

        public c(Parcel parcel) {
            super(parcel);
            this.f253b = parcel.readInt();
        }
    }

    public NestedScrollView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 0);
        this.f249e = new Rect();
        this.j = true;
        this.k = false;
        this.l = null;
        this.m = false;
        this.p = true;
        this.t = -1;
        this.u = new int[2];
        this.v = new int[2];
        this.f250f = new OverScroller(getContext());
        setFocusable(true);
        setDescendantFocusability(Calib3d.CALIB_TILTED_MODEL);
        setWillNotDraw(false);
        ViewConfiguration viewConfiguration = ViewConfiguration.get(getContext());
        this.q = viewConfiguration.getScaledTouchSlop();
        this.r = viewConfiguration.getScaledMinimumFlingVelocity();
        this.s = viewConfiguration.getScaledMaximumFlingVelocity();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, f247c, 0, 0);
        setFillViewport(obtainStyledAttributes.getBoolean(0, false));
        obtainStyledAttributes.recycle();
        this.z = new i();
        this.A = new f(this);
        setNestedScrollingEnabled(true);
        q.n(this, f246b);
    }

    public static int c(int i, int i2, int i3) {
        if (i2 >= i3 || i < 0) {
            return 0;
        }
        return i2 + i > i3 ? i3 - i2 : i;
    }

    private float getVerticalScrollFactorCompat() {
        if (this.B == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            TypedValue typedValue = new TypedValue();
            Context context = getContext();
            if (context.getTheme().resolveAttribute(16842829, typedValue, true)) {
                this.B = typedValue.getDimension(context.getResources().getDisplayMetrics());
            } else {
                throw new IllegalStateException("Expected theme to define listPreferredItemHeight.");
            }
        }
        return this.B;
    }

    public static boolean m(View view, View view2) {
        if (view == view2) {
            return true;
        }
        ViewParent parent = view.getParent();
        return (parent instanceof ViewGroup) && m((View) parent, view2);
    }

    public final void a() {
        this.f250f.abortAnimation();
        this.A.i(1);
    }

    @Override // android.view.ViewGroup
    public void addView(View view) {
        if (getChildCount() <= 0) {
            super.addView(view);
            return;
        }
        throw new IllegalStateException("ScrollView can host only one direct child");
    }

    public boolean b(int i) {
        View findFocus = findFocus();
        if (findFocus == this) {
            findFocus = null;
        }
        View findNextFocus = FocusFinder.getInstance().findNextFocus(this, findFocus, i);
        int maxScrollAmount = getMaxScrollAmount();
        if (findNextFocus != null && n(findNextFocus, maxScrollAmount, getHeight())) {
            findNextFocus.getDrawingRect(this.f249e);
            offsetDescendantRectToMyCoords(findNextFocus, this.f249e);
            f(d(this.f249e));
            findNextFocus.requestFocus(i);
        } else {
            if (i == 33 && getScrollY() < maxScrollAmount) {
                maxScrollAmount = getScrollY();
            } else if (i == 130 && getChildCount() > 0) {
                View childAt = getChildAt(0);
                maxScrollAmount = Math.min((childAt.getBottom() + ((FrameLayout.LayoutParams) childAt.getLayoutParams()).bottomMargin) - ((getHeight() + getScrollY()) - getPaddingBottom()), maxScrollAmount);
            }
            if (maxScrollAmount == 0) {
                return false;
            }
            if (i != 130) {
                maxScrollAmount = -maxScrollAmount;
            }
            f(maxScrollAmount);
        }
        if (findFocus != null && findFocus.isFocused() && (!n(findFocus, 0, getHeight()))) {
            int descendantFocusability = getDescendantFocusability();
            setDescendantFocusability(131072);
            requestFocus();
            setDescendantFocusability(descendantFocusability);
        }
        return true;
    }

    @Override // android.view.View
    public int computeHorizontalScrollExtent() {
        return super.computeHorizontalScrollExtent();
    }

    @Override // android.view.View
    public int computeHorizontalScrollOffset() {
        return super.computeHorizontalScrollOffset();
    }

    @Override // android.view.View
    public int computeHorizontalScrollRange() {
        return super.computeHorizontalScrollRange();
    }

    @Override // android.view.View
    public void computeScroll() {
        if (this.f250f.isFinished()) {
            return;
        }
        this.f250f.computeScrollOffset();
        int currY = this.f250f.getCurrY();
        int i = currY - this.x;
        this.x = currY;
        int[] iArr = this.v;
        boolean z = false;
        iArr[1] = 0;
        e(0, i, iArr, null, 1);
        int i2 = i - this.v[1];
        int scrollRange = getScrollRange();
        if (i2 != 0) {
            int scrollY = getScrollY();
            q(0, i2, getScrollX(), scrollY, 0, scrollRange, 0, 0);
            int scrollY2 = getScrollY() - scrollY;
            int i3 = i2 - scrollY2;
            int[] iArr2 = this.v;
            iArr2[1] = 0;
            this.A.e(0, scrollY2, 0, i3, this.u, 1, iArr2);
            i2 = i3 - this.v[1];
        }
        if (i2 != 0) {
            int overScrollMode = getOverScrollMode();
            if (overScrollMode == 0 || (overScrollMode == 1 && scrollRange > 0)) {
                z = true;
            }
            if (z) {
                h();
                if (i2 < 0) {
                    if (this.f251g.isFinished()) {
                        this.f251g.onAbsorb((int) this.f250f.getCurrVelocity());
                    }
                } else if (this.f252h.isFinished()) {
                    this.f252h.onAbsorb((int) this.f250f.getCurrVelocity());
                }
            }
            a();
        }
        if (!this.f250f.isFinished()) {
            AtomicInteger atomicInteger = q.f2214a;
            postInvalidateOnAnimation();
            return;
        }
        this.A.i(1);
    }

    @Override // android.view.View
    public int computeVerticalScrollExtent() {
        return super.computeVerticalScrollExtent();
    }

    @Override // android.view.View
    public int computeVerticalScrollOffset() {
        return Math.max(0, super.computeVerticalScrollOffset());
    }

    @Override // android.view.View
    public int computeVerticalScrollRange() {
        int childCount = getChildCount();
        int height = (getHeight() - getPaddingBottom()) - getPaddingTop();
        if (childCount == 0) {
            return height;
        }
        View childAt = getChildAt(0);
        int bottom = childAt.getBottom() + ((FrameLayout.LayoutParams) childAt.getLayoutParams()).bottomMargin;
        int scrollY = getScrollY();
        int max = Math.max(0, bottom - height);
        return scrollY < 0 ? bottom - scrollY : scrollY > max ? bottom + (scrollY - max) : bottom;
    }

    public int d(Rect rect) {
        int i;
        int i2;
        if (getChildCount() == 0) {
            return 0;
        }
        int height = getHeight();
        int scrollY = getScrollY();
        int i3 = scrollY + height;
        int verticalFadingEdgeLength = getVerticalFadingEdgeLength();
        if (rect.top > 0) {
            scrollY += verticalFadingEdgeLength;
        }
        View childAt = getChildAt(0);
        FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
        int i4 = rect.bottom < (childAt.getHeight() + layoutParams.topMargin) + layoutParams.bottomMargin ? i3 - verticalFadingEdgeLength : i3;
        int i5 = rect.bottom;
        if (i5 > i4 && rect.top > scrollY) {
            if (rect.height() > height) {
                i2 = rect.top - scrollY;
            } else {
                i2 = rect.bottom - i4;
            }
            return Math.min(i2 + 0, (childAt.getBottom() + layoutParams.bottomMargin) - i3);
        } else if (rect.top >= scrollY || i5 >= i4) {
            return 0;
        } else {
            if (rect.height() > height) {
                i = 0 - (i4 - rect.bottom);
            } else {
                i = 0 - (scrollY - rect.top);
            }
            return Math.max(i, -getScrollY());
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public boolean dispatchKeyEvent(KeyEvent keyEvent) {
        return super.dispatchKeyEvent(keyEvent) || i(keyEvent);
    }

    @Override // android.view.View
    public boolean dispatchNestedFling(float f2, float f3, boolean z) {
        return this.A.a(f2, f3, z);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreFling(float f2, float f3) {
        return this.A.b(f2, f3);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreScroll(int i, int i2, int[] iArr, int[] iArr2) {
        return e(i, i2, iArr, iArr2, 0);
    }

    @Override // android.view.View
    public boolean dispatchNestedScroll(int i, int i2, int i3, int i4, int[] iArr) {
        return this.A.d(i, i2, i3, i4, iArr);
    }

    @Override // android.view.View
    public void draw(Canvas canvas) {
        int i;
        super.draw(canvas);
        if (this.f251g != null) {
            int scrollY = getScrollY();
            int i2 = 0;
            if (!this.f251g.isFinished()) {
                int save = canvas.save();
                int width = getWidth();
                int height = getHeight();
                int min = Math.min(0, scrollY);
                if (getClipToPadding()) {
                    width -= getPaddingRight() + getPaddingLeft();
                    i = getPaddingLeft() + 0;
                } else {
                    i = 0;
                }
                if (getClipToPadding()) {
                    height -= getPaddingBottom() + getPaddingTop();
                    min += getPaddingTop();
                }
                canvas.translate(i, min);
                this.f251g.setSize(width, height);
                if (this.f251g.draw(canvas)) {
                    AtomicInteger atomicInteger = q.f2214a;
                    postInvalidateOnAnimation();
                }
                canvas.restoreToCount(save);
            }
            if (this.f252h.isFinished()) {
                return;
            }
            int save2 = canvas.save();
            int width2 = getWidth();
            int height2 = getHeight();
            int max = Math.max(getScrollRange(), scrollY) + height2;
            if (getClipToPadding()) {
                width2 -= getPaddingRight() + getPaddingLeft();
                i2 = 0 + getPaddingLeft();
            }
            if (getClipToPadding()) {
                height2 -= getPaddingBottom() + getPaddingTop();
                max -= getPaddingBottom();
            }
            canvas.translate(i2 - width2, max);
            canvas.rotate(180.0f, width2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            this.f252h.setSize(width2, height2);
            if (this.f252h.draw(canvas)) {
                AtomicInteger atomicInteger2 = q.f2214a;
                postInvalidateOnAnimation();
            }
            canvas.restoreToCount(save2);
        }
    }

    public boolean e(int i, int i2, int[] iArr, int[] iArr2, int i3) {
        return this.A.c(i, i2, iArr, iArr2, i3);
    }

    public final void f(int i) {
        if (i != 0) {
            if (this.p) {
                v(0, i, BaseTransientBottomBar.ANIMATION_DURATION, false);
            } else {
                scrollBy(0, i);
            }
        }
    }

    public final void g() {
        this.m = false;
        r();
        this.A.i(0);
        EdgeEffect edgeEffect = this.f251g;
        if (edgeEffect != null) {
            edgeEffect.onRelease();
            this.f252h.onRelease();
        }
    }

    @Override // android.view.View
    public float getBottomFadingEdgeStrength() {
        if (getChildCount() == 0) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        View childAt = getChildAt(0);
        int verticalFadingEdgeLength = getVerticalFadingEdgeLength();
        int bottom = ((childAt.getBottom() + ((FrameLayout.LayoutParams) childAt.getLayoutParams()).bottomMargin) - getScrollY()) - (getHeight() - getPaddingBottom());
        if (bottom < verticalFadingEdgeLength) {
            return bottom / verticalFadingEdgeLength;
        }
        return 1.0f;
    }

    public int getMaxScrollAmount() {
        return (int) (getHeight() * 0.5f);
    }

    @Override // android.view.ViewGroup
    public int getNestedScrollAxes() {
        return this.z.a();
    }

    public int getScrollRange() {
        if (getChildCount() > 0) {
            View childAt = getChildAt(0);
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
            return Math.max(0, ((childAt.getHeight() + layoutParams.topMargin) + layoutParams.bottomMargin) - ((getHeight() - getPaddingTop()) - getPaddingBottom()));
        }
        return 0;
    }

    @Override // android.view.View
    public float getTopFadingEdgeStrength() {
        if (getChildCount() == 0) {
            return StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        int verticalFadingEdgeLength = getVerticalFadingEdgeLength();
        int scrollY = getScrollY();
        if (scrollY < verticalFadingEdgeLength) {
            return scrollY / verticalFadingEdgeLength;
        }
        return 1.0f;
    }

    public final void h() {
        if (getOverScrollMode() != 2) {
            if (this.f251g == null) {
                Context context = getContext();
                this.f251g = new EdgeEffect(context);
                this.f252h = new EdgeEffect(context);
                return;
            }
            return;
        }
        this.f251g = null;
        this.f252h = null;
    }

    @Override // android.view.View
    public boolean hasNestedScrollingParent() {
        return l(0);
    }

    /* JADX WARN: Removed duplicated region for block: B:10:0x0038  */
    /* JADX WARN: Removed duplicated region for block: B:26:0x0062  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean i(KeyEvent keyEvent) {
        boolean z;
        this.f249e.setEmpty();
        if (getChildCount() > 0) {
            View childAt = getChildAt(0);
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
            if (childAt.getHeight() + layoutParams.topMargin + layoutParams.bottomMargin > (getHeight() - getPaddingTop()) - getPaddingBottom()) {
                z = true;
                if (z) {
                    if (!isFocused() || keyEvent.getKeyCode() == 4) {
                        return false;
                    }
                    View findFocus = findFocus();
                    if (findFocus == this) {
                        findFocus = null;
                    }
                    View findNextFocus = FocusFinder.getInstance().findNextFocus(this, findFocus, 130);
                    return (findNextFocus == null || findNextFocus == this || !findNextFocus.requestFocus(130)) ? false : true;
                } else if (keyEvent.getAction() == 0) {
                    int keyCode = keyEvent.getKeyCode();
                    if (keyCode == 19) {
                        if (!keyEvent.isAltPressed()) {
                            return b(33);
                        }
                        return k(33);
                    } else if (keyCode == 20) {
                        if (!keyEvent.isAltPressed()) {
                            return b(130);
                        }
                        return k(130);
                    } else if (keyCode != 62) {
                        return false;
                    } else {
                        int i = keyEvent.isShiftPressed() ? 33 : 130;
                        boolean z2 = i == 130;
                        int height = getHeight();
                        if (z2) {
                            this.f249e.top = getScrollY() + height;
                            int childCount = getChildCount();
                            if (childCount > 0) {
                                View childAt2 = getChildAt(childCount - 1);
                                int paddingBottom = getPaddingBottom() + childAt2.getBottom() + ((FrameLayout.LayoutParams) childAt2.getLayoutParams()).bottomMargin;
                                Rect rect = this.f249e;
                                if (rect.top + height > paddingBottom) {
                                    rect.top = paddingBottom - height;
                                }
                            }
                        } else {
                            this.f249e.top = getScrollY() - height;
                            Rect rect2 = this.f249e;
                            if (rect2.top < 0) {
                                rect2.top = 0;
                            }
                        }
                        Rect rect3 = this.f249e;
                        int i2 = rect3.top;
                        int i3 = height + i2;
                        rect3.bottom = i3;
                        t(i, i2, i3);
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        z = false;
        if (z) {
        }
    }

    @Override // android.view.View
    public boolean isNestedScrollingEnabled() {
        return this.A.f2207d;
    }

    public void j(int i) {
        if (getChildCount() > 0) {
            this.f250f.fling(getScrollX(), getScrollY(), 0, i, 0, 0, Integer.MIN_VALUE, Integer.MAX_VALUE, 0, 0);
            s(true);
        }
    }

    public boolean k(int i) {
        int childCount;
        boolean z = i == 130;
        int height = getHeight();
        Rect rect = this.f249e;
        rect.top = 0;
        rect.bottom = height;
        if (z && (childCount = getChildCount()) > 0) {
            View childAt = getChildAt(childCount - 1);
            this.f249e.bottom = getPaddingBottom() + childAt.getBottom() + ((FrameLayout.LayoutParams) childAt.getLayoutParams()).bottomMargin;
            Rect rect2 = this.f249e;
            rect2.top = rect2.bottom - height;
        }
        Rect rect3 = this.f249e;
        return t(i, rect3.top, rect3.bottom);
    }

    public boolean l(int i) {
        return this.A.f(i) != null;
    }

    @Override // android.view.ViewGroup
    public void measureChild(View view, int i, int i2) {
        ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
        view.measure(FrameLayout.getChildMeasureSpec(i, getPaddingRight() + getPaddingLeft(), layoutParams.width), View.MeasureSpec.makeMeasureSpec(0, 0));
    }

    @Override // android.view.ViewGroup
    public void measureChildWithMargins(View view, int i, int i2, int i3, int i4) {
        ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) view.getLayoutParams();
        view.measure(FrameLayout.getChildMeasureSpec(i, getPaddingRight() + getPaddingLeft() + marginLayoutParams.leftMargin + marginLayoutParams.rightMargin + i2, marginLayoutParams.width), View.MeasureSpec.makeMeasureSpec(marginLayoutParams.topMargin + marginLayoutParams.bottomMargin, 0));
    }

    public final boolean n(View view, int i, int i2) {
        view.getDrawingRect(this.f249e);
        offsetDescendantRectToMyCoords(view, this.f249e);
        return this.f249e.bottom + i >= getScrollY() && this.f249e.top - i <= getScrollY() + i2;
    }

    public final void o(int i, int i2, int[] iArr) {
        int scrollY = getScrollY();
        scrollBy(0, i);
        int scrollY2 = getScrollY() - scrollY;
        if (iArr != null) {
            iArr[1] = iArr[1] + scrollY2;
        }
        this.A.e(0, scrollY2, 0, i - scrollY2, null, i2, iArr);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        this.k = false;
    }

    @Override // android.view.View
    public boolean onGenericMotionEvent(MotionEvent motionEvent) {
        if ((motionEvent.getSource() & 2) != 0 && motionEvent.getAction() == 8 && !this.m) {
            float axisValue = motionEvent.getAxisValue(9);
            if (axisValue != StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                int scrollRange = getScrollRange();
                int scrollY = getScrollY();
                int verticalScrollFactorCompat = scrollY - ((int) (axisValue * getVerticalScrollFactorCompat()));
                if (verticalScrollFactorCompat < 0) {
                    scrollRange = 0;
                } else if (verticalScrollFactorCompat <= scrollRange) {
                    scrollRange = verticalScrollFactorCompat;
                }
                if (scrollRange != scrollY) {
                    super.scrollTo(getScrollX(), scrollRange);
                    return true;
                }
            }
        }
        return false;
    }

    /* JADX WARN: Removed duplicated region for block: B:51:0x00e5  */
    /* JADX WARN: Removed duplicated region for block: B:52:0x00eb  */
    @Override // android.view.ViewGroup
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        boolean z;
        int action = motionEvent.getAction();
        if (action == 2 && this.m) {
            return true;
        }
        int i = action & 255;
        if (i != 0) {
            if (i != 1) {
                if (i == 2) {
                    int i2 = this.t;
                    if (i2 != -1) {
                        int findPointerIndex = motionEvent.findPointerIndex(i2);
                        if (findPointerIndex == -1) {
                            Log.e("NestedScrollView", "Invalid pointerId=" + i2 + " in onInterceptTouchEvent");
                        } else {
                            int y = (int) motionEvent.getY(findPointerIndex);
                            if (Math.abs(y - this.i) > this.q && (2 & getNestedScrollAxes()) == 0) {
                                this.m = true;
                                this.i = y;
                                if (this.n == null) {
                                    this.n = VelocityTracker.obtain();
                                }
                                this.n.addMovement(motionEvent);
                                this.w = 0;
                                ViewParent parent = getParent();
                                if (parent != null) {
                                    parent.requestDisallowInterceptTouchEvent(true);
                                }
                            }
                        }
                    }
                } else if (i != 3) {
                    if (i == 6) {
                        p(motionEvent);
                    }
                }
            }
            this.m = false;
            this.t = -1;
            r();
            if (this.f250f.springBack(getScrollX(), getScrollY(), 0, 0, 0, getScrollRange())) {
                AtomicInteger atomicInteger = q.f2214a;
                postInvalidateOnAnimation();
            }
            this.A.i(0);
        } else {
            int y2 = (int) motionEvent.getY();
            int x = (int) motionEvent.getX();
            if (getChildCount() > 0) {
                int scrollY = getScrollY();
                View childAt = getChildAt(0);
                if (y2 >= childAt.getTop() - scrollY && y2 < childAt.getBottom() - scrollY && x >= childAt.getLeft() && x < childAt.getRight()) {
                    z = true;
                    if (z) {
                        this.m = false;
                        r();
                    } else {
                        this.i = y2;
                        this.t = motionEvent.getPointerId(0);
                        VelocityTracker velocityTracker = this.n;
                        if (velocityTracker == null) {
                            this.n = VelocityTracker.obtain();
                        } else {
                            velocityTracker.clear();
                        }
                        this.n.addMovement(motionEvent);
                        this.f250f.computeScrollOffset();
                        this.m = !this.f250f.isFinished();
                        w(2, 0);
                    }
                }
            }
            z = false;
            if (z) {
            }
        }
        return this.m;
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        int i5 = 0;
        this.j = false;
        View view = this.l;
        if (view != null && m(view, this)) {
            u(this.l);
        }
        this.l = null;
        if (!this.k) {
            if (this.y != null) {
                scrollTo(getScrollX(), this.y.f253b);
                this.y = null;
            }
            if (getChildCount() > 0) {
                View childAt = getChildAt(0);
                FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
                i5 = childAt.getMeasuredHeight() + layoutParams.topMargin + layoutParams.bottomMargin;
            }
            int paddingTop = ((i4 - i2) - getPaddingTop()) - getPaddingBottom();
            int scrollY = getScrollY();
            int c2 = c(scrollY, paddingTop, i5);
            if (c2 != scrollY) {
                scrollTo(getScrollX(), c2);
            }
        }
        scrollTo(getScrollX(), getScrollY());
        this.k = true;
    }

    @Override // android.widget.FrameLayout, android.view.View
    public void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        if (this.o && View.MeasureSpec.getMode(i2) != 0 && getChildCount() > 0) {
            View childAt = getChildAt(0);
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
            int measuredHeight = childAt.getMeasuredHeight();
            int measuredHeight2 = (((getMeasuredHeight() - getPaddingTop()) - getPaddingBottom()) - layoutParams.topMargin) - layoutParams.bottomMargin;
            if (measuredHeight < measuredHeight2) {
                childAt.measure(FrameLayout.getChildMeasureSpec(i, getPaddingRight() + getPaddingLeft() + layoutParams.leftMargin + layoutParams.rightMargin, layoutParams.width), View.MeasureSpec.makeMeasureSpec(measuredHeight2, 1073741824));
            }
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedFling(View view, float f2, float f3, boolean z) {
        if (z) {
            return false;
        }
        dispatchNestedFling(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f3, true);
        j((int) f3);
        return true;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedPreFling(View view, float f2, float f3) {
        return dispatchNestedPreFling(f2, f3);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr) {
        e(i, i2, iArr, null, 0);
    }

    @Override // b.j.j.h
    public void onNestedScroll(View view, int i, int i2, int i3, int i4, int i5, int[] iArr) {
        o(i4, i5, iArr);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScrollAccepted(View view, View view2, int i) {
        this.z.f2209a = i;
        w(2, 0);
    }

    @Override // android.view.View
    public void onOverScrolled(int i, int i2, boolean z, boolean z2) {
        super.scrollTo(i, i2);
    }

    @Override // android.view.ViewGroup
    public boolean onRequestFocusInDescendants(int i, Rect rect) {
        View findNextFocusFromRect;
        if (i == 2) {
            i = 130;
        } else if (i == 1) {
            i = 33;
        }
        if (rect == null) {
            findNextFocusFromRect = FocusFinder.getInstance().findNextFocus(this, null, i);
        } else {
            findNextFocusFromRect = FocusFinder.getInstance().findNextFocusFromRect(this, rect, i);
        }
        if (findNextFocusFromRect == null || (true ^ n(findNextFocusFromRect, 0, getHeight()))) {
            return false;
        }
        return findNextFocusFromRect.requestFocus(i, rect);
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof c)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        c cVar = (c) parcelable;
        super.onRestoreInstanceState(cVar.getSuperState());
        this.y = cVar;
        requestLayout();
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        c cVar = new c(super.onSaveInstanceState());
        cVar.f253b = getScrollY();
        return cVar;
    }

    @Override // android.view.View
    public void onScrollChanged(int i, int i2, int i3, int i4) {
        super.onScrollChanged(i, i2, i3, i4);
        b bVar = this.C;
        if (bVar != null) {
            b.b.c.b bVar2 = (b.b.c.b) bVar;
            AlertController.c(this, bVar2.f550a, bVar2.f551b);
        }
    }

    @Override // android.view.View
    public void onSizeChanged(int i, int i2, int i3, int i4) {
        super.onSizeChanged(i, i2, i3, i4);
        View findFocus = findFocus();
        if (findFocus == null || this == findFocus || !n(findFocus, 0, i4)) {
            return;
        }
        findFocus.getDrawingRect(this.f249e);
        offsetDescendantRectToMyCoords(findFocus, this.f249e);
        f(d(this.f249e));
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onStartNestedScroll(View view, View view2, int i) {
        return (i & 2) != 0;
    }

    @Override // b.j.j.g
    public boolean onStartNestedScroll(View view, View view2, int i, int i2) {
        return (i & 2) != 0;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onStopNestedScroll(View view) {
        this.z.f2209a = 0;
        x(0);
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        ViewParent parent;
        if (this.n == null) {
            this.n = VelocityTracker.obtain();
        }
        int actionMasked = motionEvent.getActionMasked();
        if (actionMasked == 0) {
            this.w = 0;
        }
        MotionEvent obtain = MotionEvent.obtain(motionEvent);
        obtain.offsetLocation(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.w);
        if (actionMasked != 0) {
            if (actionMasked == 1) {
                VelocityTracker velocityTracker = this.n;
                velocityTracker.computeCurrentVelocity(1000, this.s);
                int yVelocity = (int) velocityTracker.getYVelocity(this.t);
                if (Math.abs(yVelocity) >= this.r) {
                    int i = -yVelocity;
                    float f2 = i;
                    if (!dispatchNestedPreFling(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f2)) {
                        dispatchNestedFling(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f2, true);
                        j(i);
                    }
                } else if (this.f250f.springBack(getScrollX(), getScrollY(), 0, 0, 0, getScrollRange())) {
                    AtomicInteger atomicInteger = q.f2214a;
                    postInvalidateOnAnimation();
                }
                this.t = -1;
                g();
            } else if (actionMasked == 2) {
                int findPointerIndex = motionEvent.findPointerIndex(this.t);
                if (findPointerIndex == -1) {
                    StringBuilder x = c.b.a.a.a.x("Invalid pointerId=");
                    x.append(this.t);
                    x.append(" in onTouchEvent");
                    Log.e("NestedScrollView", x.toString());
                } else {
                    int y = (int) motionEvent.getY(findPointerIndex);
                    int i2 = this.i - y;
                    if (!this.m && Math.abs(i2) > this.q) {
                        ViewParent parent2 = getParent();
                        if (parent2 != null) {
                            parent2.requestDisallowInterceptTouchEvent(true);
                        }
                        this.m = true;
                        i2 = i2 > 0 ? i2 - this.q : i2 + this.q;
                    }
                    int i3 = i2;
                    if (this.m) {
                        if (e(0, i3, this.v, this.u, 0)) {
                            i3 -= this.v[1];
                            this.w += this.u[1];
                        }
                        int i4 = i3;
                        this.i = y - this.u[1];
                        int scrollY = getScrollY();
                        int scrollRange = getScrollRange();
                        int overScrollMode = getOverScrollMode();
                        boolean z = overScrollMode == 0 || (overScrollMode == 1 && scrollRange > 0);
                        if (q(0, i4, 0, getScrollY(), 0, scrollRange, 0, 0) && !l(0)) {
                            this.n.clear();
                        }
                        int scrollY2 = getScrollY() - scrollY;
                        int[] iArr = this.v;
                        iArr[1] = 0;
                        this.A.e(0, scrollY2, 0, i4 - scrollY2, this.u, 0, iArr);
                        int i5 = this.i;
                        int[] iArr2 = this.u;
                        this.i = i5 - iArr2[1];
                        this.w += iArr2[1];
                        if (z) {
                            int i6 = i4 - this.v[1];
                            h();
                            int i7 = scrollY + i6;
                            if (i7 < 0) {
                                this.f251g.onPull(i6 / getHeight(), motionEvent.getX(findPointerIndex) / getWidth());
                                if (!this.f252h.isFinished()) {
                                    this.f252h.onRelease();
                                }
                            } else if (i7 > scrollRange) {
                                this.f252h.onPull(i6 / getHeight(), 1.0f - (motionEvent.getX(findPointerIndex) / getWidth()));
                                if (!this.f251g.isFinished()) {
                                    this.f251g.onRelease();
                                }
                            }
                            EdgeEffect edgeEffect = this.f251g;
                            if (edgeEffect != null && (!edgeEffect.isFinished() || !this.f252h.isFinished())) {
                                AtomicInteger atomicInteger2 = q.f2214a;
                                postInvalidateOnAnimation();
                            }
                        }
                    }
                }
            } else if (actionMasked == 3) {
                if (this.m && getChildCount() > 0 && this.f250f.springBack(getScrollX(), getScrollY(), 0, 0, 0, getScrollRange())) {
                    AtomicInteger atomicInteger3 = q.f2214a;
                    postInvalidateOnAnimation();
                }
                this.t = -1;
                g();
            } else if (actionMasked == 5) {
                int actionIndex = motionEvent.getActionIndex();
                this.i = (int) motionEvent.getY(actionIndex);
                this.t = motionEvent.getPointerId(actionIndex);
            } else if (actionMasked == 6) {
                p(motionEvent);
                this.i = (int) motionEvent.getY(motionEvent.findPointerIndex(this.t));
            }
        } else if (getChildCount() == 0) {
            return false;
        } else {
            boolean z2 = !this.f250f.isFinished();
            this.m = z2;
            if (z2 && (parent = getParent()) != null) {
                parent.requestDisallowInterceptTouchEvent(true);
            }
            if (!this.f250f.isFinished()) {
                a();
            }
            this.i = (int) motionEvent.getY();
            this.t = motionEvent.getPointerId(0);
            w(2, 0);
        }
        VelocityTracker velocityTracker2 = this.n;
        if (velocityTracker2 != null) {
            velocityTracker2.addMovement(obtain);
        }
        obtain.recycle();
        return true;
    }

    public final void p(MotionEvent motionEvent) {
        int actionIndex = motionEvent.getActionIndex();
        if (motionEvent.getPointerId(actionIndex) == this.t) {
            int i = actionIndex == 0 ? 1 : 0;
            this.i = (int) motionEvent.getY(i);
            this.t = motionEvent.getPointerId(i);
            VelocityTracker velocityTracker = this.n;
            if (velocityTracker != null) {
                velocityTracker.clear();
            }
        }
    }

    public boolean q(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8) {
        boolean z;
        boolean z2;
        int overScrollMode = getOverScrollMode();
        boolean z3 = computeHorizontalScrollRange() > computeHorizontalScrollExtent();
        boolean z4 = computeVerticalScrollRange() > computeVerticalScrollExtent();
        boolean z5 = overScrollMode == 0 || (overScrollMode == 1 && z3);
        boolean z6 = overScrollMode == 0 || (overScrollMode == 1 && z4);
        int i9 = i3 + i;
        int i10 = !z5 ? 0 : i7;
        int i11 = i4 + i2;
        int i12 = !z6 ? 0 : i8;
        int i13 = -i10;
        int i14 = i10 + i5;
        int i15 = -i12;
        int i16 = i12 + i6;
        if (i9 > i14) {
            i9 = i14;
            z = true;
        } else if (i9 < i13) {
            z = true;
            i9 = i13;
        } else {
            z = false;
        }
        if (i11 > i16) {
            i11 = i16;
            z2 = true;
        } else if (i11 < i15) {
            z2 = true;
            i11 = i15;
        } else {
            z2 = false;
        }
        if (z2 && !l(1)) {
            this.f250f.springBack(i9, i11, 0, 0, 0, getScrollRange());
        }
        onOverScrolled(i9, i11, z, z2);
        return z || z2;
    }

    public final void r() {
        VelocityTracker velocityTracker = this.n;
        if (velocityTracker != null) {
            velocityTracker.recycle();
            this.n = null;
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestChildFocus(View view, View view2) {
        if (!this.j) {
            u(view2);
        } else {
            this.l = view2;
        }
        super.requestChildFocus(view, view2);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean requestChildRectangleOnScreen(View view, Rect rect, boolean z) {
        rect.offset(view.getLeft() - view.getScrollX(), view.getTop() - view.getScrollY());
        int d2 = d(rect);
        boolean z2 = d2 != 0;
        if (z2) {
            if (z) {
                scrollBy(0, d2);
            } else {
                v(0, d2, BaseTransientBottomBar.ANIMATION_DURATION, false);
            }
        }
        return z2;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestDisallowInterceptTouchEvent(boolean z) {
        if (z) {
            r();
        }
        super.requestDisallowInterceptTouchEvent(z);
    }

    @Override // android.view.View, android.view.ViewParent
    public void requestLayout() {
        this.j = true;
        super.requestLayout();
    }

    public final void s(boolean z) {
        if (z) {
            w(2, 1);
        } else {
            this.A.i(1);
        }
        this.x = getScrollY();
        AtomicInteger atomicInteger = q.f2214a;
        postInvalidateOnAnimation();
    }

    @Override // android.view.View
    public void scrollTo(int i, int i2) {
        if (getChildCount() > 0) {
            View childAt = getChildAt(0);
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
            int c2 = c(i, (getWidth() - getPaddingLeft()) - getPaddingRight(), childAt.getWidth() + layoutParams.leftMargin + layoutParams.rightMargin);
            int c3 = c(i2, (getHeight() - getPaddingTop()) - getPaddingBottom(), childAt.getHeight() + layoutParams.topMargin + layoutParams.bottomMargin);
            if (c2 == getScrollX() && c3 == getScrollY()) {
                return;
            }
            super.scrollTo(c2, c3);
        }
    }

    public void setFillViewport(boolean z) {
        if (z != this.o) {
            this.o = z;
            requestLayout();
        }
    }

    @Override // android.view.View
    public void setNestedScrollingEnabled(boolean z) {
        f fVar = this.A;
        if (fVar.f2207d) {
            View view = fVar.f2206c;
            AtomicInteger atomicInteger = q.f2214a;
            view.stopNestedScroll();
        }
        fVar.f2207d = z;
    }

    public void setOnScrollChangeListener(b bVar) {
        this.C = bVar;
    }

    public void setSmoothScrollingEnabled(boolean z) {
        this.p = z;
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup
    public boolean shouldDelayChildPressedState() {
        return true;
    }

    @Override // android.view.View
    public boolean startNestedScroll(int i) {
        return this.A.h(i, 0);
    }

    @Override // android.view.View
    public void stopNestedScroll() {
        this.A.i(0);
    }

    public final boolean t(int i, int i2, int i3) {
        boolean z;
        int height = getHeight();
        int scrollY = getScrollY();
        int i4 = height + scrollY;
        boolean z2 = i == 33;
        ArrayList focusables = getFocusables(2);
        int size = focusables.size();
        View view = null;
        boolean z3 = false;
        for (int i5 = 0; i5 < size; i5++) {
            View view2 = (View) focusables.get(i5);
            int top = view2.getTop();
            int bottom = view2.getBottom();
            if (i2 < bottom && top < i3) {
                boolean z4 = i2 < top && bottom < i3;
                if (view == null) {
                    view = view2;
                    z3 = z4;
                } else {
                    boolean z5 = (z2 && top < view.getTop()) || (!z2 && bottom > view.getBottom());
                    if (z3) {
                        if (z4) {
                            if (!z5) {
                            }
                            view = view2;
                        }
                    } else if (z4) {
                        view = view2;
                        z3 = true;
                    } else {
                        if (!z5) {
                        }
                        view = view2;
                    }
                }
            }
        }
        if (view == null) {
            view = this;
        }
        if (i2 < scrollY || i3 > i4) {
            f(z2 ? i2 - scrollY : i3 - i4);
            z = true;
        } else {
            z = false;
        }
        if (view != findFocus()) {
            view.requestFocus(i);
        }
        return z;
    }

    public final void u(View view) {
        view.getDrawingRect(this.f249e);
        offsetDescendantRectToMyCoords(view, this.f249e);
        int d2 = d(this.f249e);
        if (d2 != 0) {
            scrollBy(0, d2);
        }
    }

    public final void v(int i, int i2, int i3, boolean z) {
        if (getChildCount() == 0) {
            return;
        }
        if (AnimationUtils.currentAnimationTimeMillis() - this.f248d > 250) {
            View childAt = getChildAt(0);
            FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) childAt.getLayoutParams();
            int scrollY = getScrollY();
            OverScroller overScroller = this.f250f;
            int scrollX = getScrollX();
            overScroller.startScroll(scrollX, scrollY, 0, Math.max(0, Math.min(i2 + scrollY, Math.max(0, ((childAt.getHeight() + layoutParams.topMargin) + layoutParams.bottomMargin) - ((getHeight() - getPaddingTop()) - getPaddingBottom())))) - scrollY, i3);
            s(z);
        } else {
            if (!this.f250f.isFinished()) {
                a();
            }
            scrollBy(i, i2);
        }
        this.f248d = AnimationUtils.currentAnimationTimeMillis();
    }

    public boolean w(int i, int i2) {
        return this.A.h(i, i2);
    }

    public void x(int i) {
        this.A.i(i);
    }

    @Override // b.j.j.g
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr, int i3) {
        e(i, i2, iArr, null, i3);
    }

    @Override // b.j.j.g
    public void onNestedScroll(View view, int i, int i2, int i3, int i4, int i5) {
        o(i4, i5, null);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScroll(View view, int i, int i2, int i3, int i4) {
        o(i4, 0, null);
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i) {
        if (getChildCount() <= 0) {
            super.addView(view, i);
            return;
        }
        throw new IllegalStateException("ScrollView can host only one direct child");
    }

    @Override // b.j.j.g
    public void onNestedScrollAccepted(View view, View view2, int i, int i2) {
        i iVar = this.z;
        if (i2 == 1) {
            iVar.f2210b = i;
        } else {
            iVar.f2209a = i;
        }
        w(2, i2);
    }

    @Override // b.j.j.g
    public void onStopNestedScroll(View view, int i) {
        i iVar = this.z;
        if (i == 1) {
            iVar.f2210b = 0;
        } else {
            iVar.f2209a = 0;
        }
        x(i);
    }

    @Override // android.view.ViewGroup, android.view.ViewManager
    public void addView(View view, ViewGroup.LayoutParams layoutParams) {
        if (getChildCount() <= 0) {
            super.addView(view, layoutParams);
            return;
        }
        throw new IllegalStateException("ScrollView can host only one direct child");
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i, ViewGroup.LayoutParams layoutParams) {
        if (getChildCount() <= 0) {
            super.addView(view, i, layoutParams);
            return;
        }
        throw new IllegalStateException("ScrollView can host only one direct child");
    }
}