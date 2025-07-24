package androidx.appcompat.widget;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Configuration;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.util.AttributeSet;
import android.view.Menu;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewPropertyAnimator;
import android.view.Window;
import android.view.WindowInsets;
import android.widget.OverScroller;
import b.b.c.u;
import b.b.g.i.m;
import b.b.h.c0;
import b.b.h.d0;
import b.j.j.g;
import b.j.j.h;
import b.j.j.i;
import b.j.j.q;
import b.j.j.w;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

@SuppressLint({"UnknownNullness"})
/* loaded from: classes.dex */
public class ActionBarOverlayLayout extends ViewGroup implements c0, g, h {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f103b = {R.attr.actionBarSize, 16842841};
    public d A;
    public OverScroller B;
    public ViewPropertyAnimator C;
    public final AnimatorListenerAdapter D;
    public final Runnable E;
    public final Runnable F;
    public final i G;

    /* renamed from: c  reason: collision with root package name */
    public int f104c;

    /* renamed from: d  reason: collision with root package name */
    public int f105d;

    /* renamed from: e  reason: collision with root package name */
    public ContentFrameLayout f106e;

    /* renamed from: f  reason: collision with root package name */
    public ActionBarContainer f107f;

    /* renamed from: g  reason: collision with root package name */
    public d0 f108g;

    /* renamed from: h  reason: collision with root package name */
    public Drawable f109h;
    public boolean i;
    public boolean j;
    public boolean k;
    public boolean l;
    public boolean m;
    public int n;
    public int o;
    public final Rect p;
    public final Rect q;
    public final Rect r;
    public final Rect s;
    public final Rect t;
    public final Rect u;
    public final Rect v;
    public w w;
    public w x;
    public w y;
    public w z;

    /* loaded from: classes.dex */
    public class a extends AnimatorListenerAdapter {
        public a() {
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationCancel(Animator animator) {
            ActionBarOverlayLayout actionBarOverlayLayout = ActionBarOverlayLayout.this;
            actionBarOverlayLayout.C = null;
            actionBarOverlayLayout.m = false;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            ActionBarOverlayLayout actionBarOverlayLayout = ActionBarOverlayLayout.this;
            actionBarOverlayLayout.C = null;
            actionBarOverlayLayout.m = false;
        }
    }

    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            ActionBarOverlayLayout.this.k();
            ActionBarOverlayLayout actionBarOverlayLayout = ActionBarOverlayLayout.this;
            actionBarOverlayLayout.C = actionBarOverlayLayout.f107f.animate().translationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setListener(ActionBarOverlayLayout.this.D);
        }
    }

    /* loaded from: classes.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            ActionBarOverlayLayout.this.k();
            ActionBarOverlayLayout actionBarOverlayLayout = ActionBarOverlayLayout.this;
            actionBarOverlayLayout.C = actionBarOverlayLayout.f107f.animate().translationY(-ActionBarOverlayLayout.this.f107f.getHeight()).setListener(ActionBarOverlayLayout.this.D);
        }
    }

    /* loaded from: classes.dex */
    public interface d {
    }

    /* loaded from: classes.dex */
    public static class e extends ViewGroup.MarginLayoutParams {
        public e(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
        }

        public e(int i, int i2) {
            super(i, i2);
        }

        public e(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
        }
    }

    public ActionBarOverlayLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.f105d = 0;
        this.p = new Rect();
        this.q = new Rect();
        this.r = new Rect();
        this.s = new Rect();
        this.t = new Rect();
        this.u = new Rect();
        this.v = new Rect();
        w wVar = w.f2237a;
        this.w = wVar;
        this.x = wVar;
        this.y = wVar;
        this.z = wVar;
        this.D = new a();
        this.E = new b();
        this.F = new c();
        l(context);
        this.G = new i();
    }

    @Override // b.b.h.c0
    public void a(Menu menu, m.a aVar) {
        m();
        this.f108g.a(menu, aVar);
    }

    @Override // b.b.h.c0
    public boolean b() {
        m();
        return this.f108g.b();
    }

    @Override // b.b.h.c0
    public void c() {
        m();
        this.f108g.c();
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return layoutParams instanceof e;
    }

    @Override // b.b.h.c0
    public boolean d() {
        m();
        return this.f108g.d();
    }

    @Override // android.view.View
    public void draw(Canvas canvas) {
        int i;
        super.draw(canvas);
        if (this.f109h == null || this.i) {
            return;
        }
        if (this.f107f.getVisibility() == 0) {
            i = (int) (this.f107f.getTranslationY() + this.f107f.getBottom() + 0.5f);
        } else {
            i = 0;
        }
        this.f109h.setBounds(0, i, getWidth(), this.f109h.getIntrinsicHeight() + i);
        this.f109h.draw(canvas);
    }

    @Override // b.b.h.c0
    public boolean e() {
        m();
        return this.f108g.e();
    }

    @Override // b.b.h.c0
    public boolean f() {
        m();
        return this.f108g.f();
    }

    @Override // android.view.View
    public boolean fitSystemWindows(Rect rect) {
        return super.fitSystemWindows(rect);
    }

    @Override // b.b.h.c0
    public boolean g() {
        m();
        return this.f108g.g();
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateDefaultLayoutParams() {
        return new e(-1, -1);
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(AttributeSet attributeSet) {
        return new e(getContext(), attributeSet);
    }

    public int getActionBarHideOffset() {
        ActionBarContainer actionBarContainer = this.f107f;
        if (actionBarContainer != null) {
            return -((int) actionBarContainer.getTranslationY());
        }
        return 0;
    }

    @Override // android.view.ViewGroup
    public int getNestedScrollAxes() {
        return this.G.a();
    }

    public CharSequence getTitle() {
        m();
        return this.f108g.getTitle();
    }

    @Override // b.b.h.c0
    public void h(int i) {
        m();
        if (i == 2) {
            this.f108g.t();
        } else if (i == 5) {
            this.f108g.u();
        } else if (i != 109) {
        } else {
            setOverlayMode(true);
        }
    }

    @Override // b.b.h.c0
    public void i() {
        m();
        this.f108g.h();
    }

    /* JADX WARN: Removed duplicated region for block: B:13:0x0021  */
    /* JADX WARN: Removed duplicated region for block: B:17:0x002c  */
    /* JADX WARN: Removed duplicated region for block: B:9:0x0016  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean j(View view, Rect rect, boolean z, boolean z2, boolean z3, boolean z4) {
        boolean z5;
        e eVar = (e) view.getLayoutParams();
        if (z) {
            int i = ((ViewGroup.MarginLayoutParams) eVar).leftMargin;
            int i2 = rect.left;
            if (i != i2) {
                ((ViewGroup.MarginLayoutParams) eVar).leftMargin = i2;
                z5 = true;
                if (z2) {
                    int i3 = ((ViewGroup.MarginLayoutParams) eVar).topMargin;
                    int i4 = rect.top;
                    if (i3 != i4) {
                        ((ViewGroup.MarginLayoutParams) eVar).topMargin = i4;
                        z5 = true;
                    }
                }
                if (z4) {
                    int i5 = ((ViewGroup.MarginLayoutParams) eVar).rightMargin;
                    int i6 = rect.right;
                    if (i5 != i6) {
                        ((ViewGroup.MarginLayoutParams) eVar).rightMargin = i6;
                        z5 = true;
                    }
                }
                if (z3) {
                    int i7 = ((ViewGroup.MarginLayoutParams) eVar).bottomMargin;
                    int i8 = rect.bottom;
                    if (i7 != i8) {
                        ((ViewGroup.MarginLayoutParams) eVar).bottomMargin = i8;
                        return true;
                    }
                }
                return z5;
            }
        }
        z5 = false;
        if (z2) {
        }
        if (z4) {
        }
        if (z3) {
        }
        return z5;
    }

    public void k() {
        removeCallbacks(this.E);
        removeCallbacks(this.F);
        ViewPropertyAnimator viewPropertyAnimator = this.C;
        if (viewPropertyAnimator != null) {
            viewPropertyAnimator.cancel();
        }
    }

    public final void l(Context context) {
        TypedArray obtainStyledAttributes = getContext().getTheme().obtainStyledAttributes(f103b);
        this.f104c = obtainStyledAttributes.getDimensionPixelSize(0, 0);
        Drawable drawable = obtainStyledAttributes.getDrawable(1);
        this.f109h = drawable;
        setWillNotDraw(drawable == null);
        obtainStyledAttributes.recycle();
        this.i = context.getApplicationInfo().targetSdkVersion < 19;
        this.B = new OverScroller(context);
    }

    public void m() {
        d0 wrapper;
        if (this.f106e == null) {
            this.f106e = (ContentFrameLayout) findViewById(R.id.action_bar_activity_content);
            this.f107f = (ActionBarContainer) findViewById(R.id.action_bar_container);
            View findViewById = findViewById(R.id.action_bar);
            if (findViewById instanceof d0) {
                wrapper = (d0) findViewById;
            } else if (findViewById instanceof Toolbar) {
                wrapper = ((Toolbar) findViewById).getWrapper();
            } else {
                StringBuilder x = c.b.a.a.a.x("Can't make a decor toolbar out of ");
                x.append(findViewById.getClass().getSimpleName());
                throw new IllegalStateException(x.toString());
            }
            this.f108g = wrapper;
        }
    }

    @Override // android.view.View
    public WindowInsets onApplyWindowInsets(WindowInsets windowInsets) {
        m();
        w k = w.k(windowInsets, null);
        boolean j = j(this.f107f, new Rect(k.c(), k.e(), k.d(), k.b()), true, true, false, true);
        Rect rect = this.p;
        AtomicInteger atomicInteger = q.f2214a;
        q.b.b(this, k, rect);
        Rect rect2 = this.p;
        w j2 = k.f2238b.j(rect2.left, rect2.top, rect2.right, rect2.bottom);
        this.w = j2;
        boolean z = true;
        if (!this.x.equals(j2)) {
            this.x = this.w;
            j = true;
        }
        if (this.q.equals(this.p)) {
            z = j;
        } else {
            this.q.set(this.p);
        }
        if (z) {
            requestLayout();
        }
        return k.f2238b.a().a().f2238b.b().i();
    }

    @Override // android.view.View
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        l(getContext());
        AtomicInteger atomicInteger = q.f2214a;
        requestApplyInsets();
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        k();
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        int childCount = getChildCount();
        int paddingLeft = getPaddingLeft();
        int paddingTop = getPaddingTop();
        for (int i5 = 0; i5 < childCount; i5++) {
            View childAt = getChildAt(i5);
            if (childAt.getVisibility() != 8) {
                e eVar = (e) childAt.getLayoutParams();
                int measuredWidth = childAt.getMeasuredWidth();
                int measuredHeight = childAt.getMeasuredHeight();
                int i6 = ((ViewGroup.MarginLayoutParams) eVar).leftMargin + paddingLeft;
                int i7 = ((ViewGroup.MarginLayoutParams) eVar).topMargin + paddingTop;
                childAt.layout(i6, i7, measuredWidth + i6, measuredHeight + i7);
            }
        }
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        int measuredHeight;
        w.d aVar;
        m();
        measureChildWithMargins(this.f107f, i, 0, i2, 0);
        e eVar = (e) this.f107f.getLayoutParams();
        int max = Math.max(0, this.f107f.getMeasuredWidth() + ((ViewGroup.MarginLayoutParams) eVar).leftMargin + ((ViewGroup.MarginLayoutParams) eVar).rightMargin);
        int max2 = Math.max(0, this.f107f.getMeasuredHeight() + ((ViewGroup.MarginLayoutParams) eVar).topMargin + ((ViewGroup.MarginLayoutParams) eVar).bottomMargin);
        int combineMeasuredStates = View.combineMeasuredStates(0, this.f107f.getMeasuredState());
        AtomicInteger atomicInteger = q.f2214a;
        boolean z = (getWindowSystemUiVisibility() & 256) != 0;
        if (z) {
            measuredHeight = this.f104c;
            if (this.k && this.f107f.getTabContainer() != null) {
                measuredHeight += this.f104c;
            }
        } else {
            measuredHeight = this.f107f.getVisibility() != 8 ? this.f107f.getMeasuredHeight() : 0;
        }
        this.r.set(this.p);
        w wVar = this.w;
        this.y = wVar;
        if (!this.j && !z) {
            Rect rect = this.r;
            rect.top += measuredHeight;
            rect.bottom += 0;
            this.y = wVar.f2238b.j(0, measuredHeight, 0, 0);
        } else {
            b.j.d.b a2 = b.j.d.b.a(wVar.c(), this.y.e() + measuredHeight, this.y.d(), this.y.b() + 0);
            w wVar2 = this.y;
            int i3 = Build.VERSION.SDK_INT;
            if (i3 >= 30) {
                aVar = new w.c(wVar2);
            } else if (i3 >= 29) {
                aVar = new w.b(wVar2);
            } else {
                aVar = new w.a(wVar2);
            }
            aVar.d(a2);
            this.y = aVar.b();
        }
        j(this.f106e, this.r, true, true, true, true);
        if (!this.z.equals(this.y)) {
            w wVar3 = this.y;
            this.z = wVar3;
            q.c(this.f106e, wVar3);
        }
        measureChildWithMargins(this.f106e, i, 0, i2, 0);
        e eVar2 = (e) this.f106e.getLayoutParams();
        int max3 = Math.max(max, this.f106e.getMeasuredWidth() + ((ViewGroup.MarginLayoutParams) eVar2).leftMargin + ((ViewGroup.MarginLayoutParams) eVar2).rightMargin);
        int max4 = Math.max(max2, this.f106e.getMeasuredHeight() + ((ViewGroup.MarginLayoutParams) eVar2).topMargin + ((ViewGroup.MarginLayoutParams) eVar2).bottomMargin);
        int combineMeasuredStates2 = View.combineMeasuredStates(combineMeasuredStates, this.f106e.getMeasuredState());
        setMeasuredDimension(View.resolveSizeAndState(Math.max(getPaddingRight() + getPaddingLeft() + max3, getSuggestedMinimumWidth()), i, combineMeasuredStates2), View.resolveSizeAndState(Math.max(getPaddingBottom() + getPaddingTop() + max4, getSuggestedMinimumHeight()), i2, combineMeasuredStates2 << 16));
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedFling(View view, float f2, float f3, boolean z) {
        if (this.l && z) {
            this.B.fling(0, 0, 0, (int) f3, 0, 0, Integer.MIN_VALUE, Integer.MAX_VALUE);
            if (this.B.getFinalY() > this.f107f.getHeight()) {
                k();
                this.F.run();
            } else {
                k();
                this.E.run();
            }
            this.m = true;
            return true;
        }
        return false;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedPreFling(View view, float f2, float f3) {
        return false;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr) {
    }

    @Override // b.j.j.g
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr, int i3) {
        if (i3 == 0) {
            onNestedPreScroll(view, i, i2, iArr);
        }
    }

    @Override // b.j.j.g
    public void onNestedScroll(View view, int i, int i2, int i3, int i4, int i5) {
        if (i5 == 0) {
            onNestedScroll(view, i, i2, i3, i4);
        }
    }

    @Override // b.j.j.g
    public void onNestedScrollAccepted(View view, View view2, int i, int i2) {
        if (i2 == 0) {
            onNestedScrollAccepted(view, view2, i);
        }
    }

    @Override // b.j.j.g
    public boolean onStartNestedScroll(View view, View view2, int i, int i2) {
        return i2 == 0 && onStartNestedScroll(view, view2, i);
    }

    @Override // b.j.j.g
    public void onStopNestedScroll(View view, int i) {
        if (i == 0) {
            onStopNestedScroll(view);
        }
    }

    @Override // android.view.View
    public void onWindowSystemUiVisibilityChanged(int i) {
        super.onWindowSystemUiVisibilityChanged(i);
        m();
        int i2 = this.o ^ i;
        this.o = i;
        boolean z = (i & 4) == 0;
        boolean z2 = (i & 256) != 0;
        d dVar = this.A;
        if (dVar != null) {
            ((u) dVar).r = !z2;
            if (!z && z2) {
                u uVar = (u) dVar;
                if (!uVar.s) {
                    uVar.s = true;
                    uVar.g(true);
                }
            } else {
                u uVar2 = (u) dVar;
                if (uVar2.s) {
                    uVar2.s = false;
                    uVar2.g(true);
                }
            }
        }
        if ((i2 & 256) == 0 || this.A == null) {
            return;
        }
        AtomicInteger atomicInteger = q.f2214a;
        requestApplyInsets();
    }

    @Override // android.view.View
    public void onWindowVisibilityChanged(int i) {
        super.onWindowVisibilityChanged(i);
        this.f105d = i;
        d dVar = this.A;
        if (dVar != null) {
            ((u) dVar).q = i;
        }
    }

    public void setActionBarHideOffset(int i) {
        k();
        this.f107f.setTranslationY(-Math.max(0, Math.min(i, this.f107f.getHeight())));
    }

    public void setActionBarVisibilityCallback(d dVar) {
        this.A = dVar;
        if (getWindowToken() != null) {
            ((u) this.A).q = this.f105d;
            int i = this.o;
            if (i != 0) {
                onWindowSystemUiVisibilityChanged(i);
                AtomicInteger atomicInteger = q.f2214a;
                requestApplyInsets();
            }
        }
    }

    public void setHasNonEmbeddedTabs(boolean z) {
        this.k = z;
    }

    public void setHideOnContentScrollEnabled(boolean z) {
        if (z != this.l) {
            this.l = z;
            if (z) {
                return;
            }
            k();
            setActionBarHideOffset(0);
        }
    }

    public void setIcon(int i) {
        m();
        this.f108g.setIcon(i);
    }

    public void setLogo(int i) {
        m();
        this.f108g.l(i);
    }

    public void setOverlayMode(boolean z) {
        this.j = z;
        this.i = z && getContext().getApplicationInfo().targetSdkVersion < 19;
    }

    public void setShowingForActionMode(boolean z) {
    }

    public void setUiOptions(int i) {
    }

    @Override // b.b.h.c0
    public void setWindowCallback(Window.Callback callback) {
        m();
        this.f108g.setWindowCallback(callback);
    }

    @Override // b.b.h.c0
    public void setWindowTitle(CharSequence charSequence) {
        m();
        this.f108g.setWindowTitle(charSequence);
    }

    @Override // android.view.ViewGroup
    public boolean shouldDelayChildPressedState() {
        return false;
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return new e(layoutParams);
    }

    @Override // b.j.j.h
    public void onNestedScroll(View view, int i, int i2, int i3, int i4, int i5, int[] iArr) {
        if (i5 == 0) {
            onNestedScroll(view, i, i2, i3, i4);
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScrollAccepted(View view, View view2, int i) {
        u uVar;
        b.b.g.g gVar;
        this.G.f2209a = i;
        this.n = getActionBarHideOffset();
        k();
        d dVar = this.A;
        if (dVar == null || (gVar = (uVar = (u) dVar).v) == null) {
            return;
        }
        gVar.a();
        uVar.v = null;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onStartNestedScroll(View view, View view2, int i) {
        if ((i & 2) == 0 || this.f107f.getVisibility() != 0) {
            return false;
        }
        return this.l;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onStopNestedScroll(View view) {
        if (this.l && !this.m) {
            if (this.n <= this.f107f.getHeight()) {
                k();
                postDelayed(this.E, 600L);
            } else {
                k();
                postDelayed(this.F, 600L);
            }
        }
        d dVar = this.A;
        if (dVar != null) {
            Objects.requireNonNull((u) dVar);
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScroll(View view, int i, int i2, int i3, int i4) {
        int i5 = this.n + i2;
        this.n = i5;
        setActionBarHideOffset(i5);
    }

    public void setIcon(Drawable drawable) {
        m();
        this.f108g.setIcon(drawable);
    }
}