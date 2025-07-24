package androidx.swiperefreshlayout.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.Transformation;
import android.widget.ListView;
import b.j.j.i;
import b.j.j.q;
import b.y.a.d;
import c.e.b.c0;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.ibosoninnov.unitear.ARGalleryActivity;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* loaded from: classes.dex */
public class SwipeRefreshLayout extends ViewGroup implements b.j.j.e {

    /* renamed from: b  reason: collision with root package name */
    public static final String f490b = SwipeRefreshLayout.class.getSimpleName();

    /* renamed from: c  reason: collision with root package name */
    public static final int[] f491c = {16842766};
    public int A;
    public b.y.a.d B;
    public Animation C;
    public Animation D;
    public Animation E;
    public Animation F;
    public boolean G;
    public int H;
    public g I;
    public Animation.AnimationListener J;
    public final Animation K;
    public final Animation L;

    /* renamed from: d  reason: collision with root package name */
    public View f492d;

    /* renamed from: e  reason: collision with root package name */
    public h f493e;

    /* renamed from: f  reason: collision with root package name */
    public boolean f494f;

    /* renamed from: g  reason: collision with root package name */
    public int f495g;

    /* renamed from: h  reason: collision with root package name */
    public float f496h;
    public float i;
    public final i j;
    public final b.j.j.f k;
    public final int[] l;
    public final int[] m;
    public boolean n;
    public int o;
    public int p;
    public float q;
    public float r;
    public boolean s;
    public int t;
    public final DecelerateInterpolator u;
    public b.y.a.a v;
    public int w;
    public int x;
    public int y;
    public int z;

    /* loaded from: classes.dex */
    public class a implements Animation.AnimationListener {
        public a() {
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationEnd(Animation animation) {
            h hVar;
            SwipeRefreshLayout swipeRefreshLayout = SwipeRefreshLayout.this;
            if (swipeRefreshLayout.f494f) {
                swipeRefreshLayout.B.setAlpha(255);
                SwipeRefreshLayout.this.B.start();
                SwipeRefreshLayout swipeRefreshLayout2 = SwipeRefreshLayout.this;
                if (swipeRefreshLayout2.G && (hVar = swipeRefreshLayout2.f493e) != null) {
                    ARGalleryActivity aRGalleryActivity = ((c0) hVar).f4585a;
                    Objects.requireNonNull(aRGalleryActivity);
                    Log.d("Pull to refresh", "Called");
                    aRGalleryActivity.w.clear();
                    aRGalleryActivity.x.clear();
                    aRGalleryActivity.z.notifyDataSetChanged();
                    aRGalleryActivity.w();
                }
                SwipeRefreshLayout swipeRefreshLayout3 = SwipeRefreshLayout.this;
                swipeRefreshLayout3.p = swipeRefreshLayout3.v.getTop();
                return;
            }
            swipeRefreshLayout.h();
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationRepeat(Animation animation) {
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationStart(Animation animation) {
        }
    }

    /* loaded from: classes.dex */
    public class b extends Animation {
        public b() {
        }

        @Override // android.view.animation.Animation
        public void applyTransformation(float f2, Transformation transformation) {
            SwipeRefreshLayout.this.setAnimationProgress(1.0f - f2);
        }
    }

    /* loaded from: classes.dex */
    public class c extends Animation {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f499b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ int f500c;

        public c(int i, int i2) {
            this.f499b = i;
            this.f500c = i2;
        }

        @Override // android.view.animation.Animation
        public void applyTransformation(float f2, Transformation transformation) {
            b.y.a.d dVar = SwipeRefreshLayout.this.B;
            int i = this.f499b;
            dVar.setAlpha((int) (((this.f500c - i) * f2) + i));
        }
    }

    /* loaded from: classes.dex */
    public class d implements Animation.AnimationListener {
        public d() {
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationEnd(Animation animation) {
            Objects.requireNonNull(SwipeRefreshLayout.this);
            SwipeRefreshLayout.this.l(null);
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationRepeat(Animation animation) {
        }

        @Override // android.view.animation.Animation.AnimationListener
        public void onAnimationStart(Animation animation) {
        }
    }

    /* loaded from: classes.dex */
    public class e extends Animation {
        public e() {
        }

        @Override // android.view.animation.Animation
        public void applyTransformation(float f2, Transformation transformation) {
            Objects.requireNonNull(SwipeRefreshLayout.this);
            SwipeRefreshLayout swipeRefreshLayout = SwipeRefreshLayout.this;
            int abs = swipeRefreshLayout.z - Math.abs(swipeRefreshLayout.y);
            SwipeRefreshLayout swipeRefreshLayout2 = SwipeRefreshLayout.this;
            int i = swipeRefreshLayout2.x;
            SwipeRefreshLayout.this.setTargetOffsetTopAndBottom((i + ((int) ((abs - i) * f2))) - swipeRefreshLayout2.v.getTop());
            b.y.a.d dVar = SwipeRefreshLayout.this.B;
            float f3 = 1.0f - f2;
            d.a aVar = dVar.f2836e;
            if (f3 != aVar.p) {
                aVar.p = f3;
            }
            dVar.invalidateSelf();
        }
    }

    /* loaded from: classes.dex */
    public class f extends Animation {
        public f() {
        }

        @Override // android.view.animation.Animation
        public void applyTransformation(float f2, Transformation transformation) {
            SwipeRefreshLayout.this.f(f2);
        }
    }

    /* loaded from: classes.dex */
    public interface g {
        boolean a(SwipeRefreshLayout swipeRefreshLayout, View view);
    }

    /* loaded from: classes.dex */
    public interface h {
    }

    public SwipeRefreshLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.f494f = false;
        this.f496h = -1.0f;
        this.l = new int[2];
        this.m = new int[2];
        this.t = -1;
        this.w = -1;
        this.J = new a();
        this.K = new e();
        this.L = new f();
        this.f495g = ViewConfiguration.get(context).getScaledTouchSlop();
        this.o = getResources().getInteger(17694721);
        setWillNotDraw(false);
        this.u = new DecelerateInterpolator(2.0f);
        DisplayMetrics displayMetrics = getResources().getDisplayMetrics();
        this.H = (int) (displayMetrics.density * 40.0f);
        this.v = new b.y.a.a(getContext(), -328966);
        b.y.a.d dVar = new b.y.a.d(getContext());
        this.B = dVar;
        dVar.c(1);
        this.v.setImageDrawable(this.B);
        this.v.setVisibility(8);
        addView(this.v);
        setChildrenDrawingOrderEnabled(true);
        int i = (int) (displayMetrics.density * 64.0f);
        this.z = i;
        this.f496h = i;
        this.j = new i();
        this.k = new b.j.j.f(this);
        setNestedScrollingEnabled(true);
        int i2 = -this.H;
        this.p = i2;
        this.y = i2;
        f(1.0f);
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, f491c);
        setEnabled(obtainStyledAttributes.getBoolean(0, true));
        obtainStyledAttributes.recycle();
    }

    private void setColorViewAlpha(int i) {
        this.v.getBackground().setAlpha(i);
        b.y.a.d dVar = this.B;
        dVar.f2836e.t = i;
        dVar.invalidateSelf();
    }

    public boolean a() {
        g gVar = this.I;
        if (gVar != null) {
            return gVar.a(this, this.f492d);
        }
        View view = this.f492d;
        if (view instanceof ListView) {
            return ((ListView) view).canScrollList(-1);
        }
        return view.canScrollVertically(-1);
    }

    public final void b() {
        if (this.f492d == null) {
            for (int i = 0; i < getChildCount(); i++) {
                View childAt = getChildAt(i);
                if (!childAt.equals(this.v)) {
                    this.f492d = childAt;
                    return;
                }
            }
        }
    }

    public final void c(float f2) {
        if (f2 > this.f496h) {
            i(true, true);
            return;
        }
        this.f494f = false;
        b.y.a.d dVar = this.B;
        d.a aVar = dVar.f2836e;
        aVar.f2844e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar.f2845f = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        dVar.invalidateSelf();
        d dVar2 = new d();
        this.x = this.p;
        this.L.reset();
        this.L.setDuration(200L);
        this.L.setInterpolator(this.u);
        b.y.a.a aVar2 = this.v;
        aVar2.f2827b = dVar2;
        aVar2.clearAnimation();
        this.v.startAnimation(this.L);
        b.y.a.d dVar3 = this.B;
        d.a aVar3 = dVar3.f2836e;
        if (aVar3.n) {
            aVar3.n = false;
        }
        dVar3.invalidateSelf();
    }

    public final boolean d(Animation animation) {
        return (animation == null || !animation.hasStarted() || animation.hasEnded()) ? false : true;
    }

    @Override // android.view.View
    public boolean dispatchNestedFling(float f2, float f3, boolean z) {
        return this.k.a(f2, f3, z);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreFling(float f2, float f3) {
        return this.k.b(f2, f3);
    }

    @Override // android.view.View
    public boolean dispatchNestedPreScroll(int i, int i2, int[] iArr, int[] iArr2) {
        return this.k.c(i, i2, iArr, iArr2, 0);
    }

    @Override // android.view.View
    public boolean dispatchNestedScroll(int i, int i2, int i3, int i4, int[] iArr) {
        return this.k.d(i, i2, i3, i4, iArr);
    }

    public final void e(float f2) {
        b.y.a.d dVar = this.B;
        d.a aVar = dVar.f2836e;
        if (!aVar.n) {
            aVar.n = true;
        }
        dVar.invalidateSelf();
        float min = Math.min(1.0f, Math.abs(f2 / this.f496h));
        float max = (((float) Math.max(min - 0.4d, (double) ShadowDrawableWrapper.COS_45)) * 5.0f) / 3.0f;
        float abs = Math.abs(f2) - this.f496h;
        int i = this.A;
        if (i <= 0) {
            i = this.z;
        }
        float f3 = i;
        double max2 = Math.max((float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, Math.min(abs, f3 * 2.0f) / f3) / 4.0f;
        float pow = ((float) (max2 - Math.pow(max2, 2.0d))) * 2.0f;
        int i2 = this.y + ((int) ((f3 * min) + (f3 * pow * 2.0f)));
        if (this.v.getVisibility() != 0) {
            this.v.setVisibility(0);
        }
        this.v.setScaleX(1.0f);
        this.v.setScaleY(1.0f);
        if (f2 < this.f496h) {
            if (this.B.f2836e.t > 76 && !d(this.E)) {
                this.E = j(this.B.f2836e.t, 76);
            }
        } else if (this.B.f2836e.t < 255 && !d(this.F)) {
            this.F = j(this.B.f2836e.t, 255);
        }
        b.y.a.d dVar2 = this.B;
        float min2 = Math.min(0.8f, max * 0.8f);
        d.a aVar2 = dVar2.f2836e;
        aVar2.f2844e = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        aVar2.f2845f = min2;
        dVar2.invalidateSelf();
        b.y.a.d dVar3 = this.B;
        float min3 = Math.min(1.0f, max);
        d.a aVar3 = dVar3.f2836e;
        if (min3 != aVar3.p) {
            aVar3.p = min3;
        }
        dVar3.invalidateSelf();
        b.y.a.d dVar4 = this.B;
        dVar4.f2836e.f2846g = ((pow * 2.0f) + ((max * 0.4f) - 0.25f)) * 0.5f;
        dVar4.invalidateSelf();
        setTargetOffsetTopAndBottom(i2 - this.p);
    }

    public void f(float f2) {
        int i = this.x;
        setTargetOffsetTopAndBottom((i + ((int) ((this.y - i) * f2))) - this.v.getTop());
    }

    public final void g(MotionEvent motionEvent) {
        int actionIndex = motionEvent.getActionIndex();
        if (motionEvent.getPointerId(actionIndex) == this.t) {
            this.t = motionEvent.getPointerId(actionIndex == 0 ? 1 : 0);
        }
    }

    @Override // android.view.ViewGroup
    public int getChildDrawingOrder(int i, int i2) {
        int i3 = this.w;
        return i3 < 0 ? i2 : i2 == i + (-1) ? i3 : i2 >= i3 ? i2 + 1 : i2;
    }

    @Override // android.view.ViewGroup
    public int getNestedScrollAxes() {
        return this.j.a();
    }

    public int getProgressCircleDiameter() {
        return this.H;
    }

    public int getProgressViewEndOffset() {
        return this.z;
    }

    public int getProgressViewStartOffset() {
        return this.y;
    }

    public void h() {
        this.v.clearAnimation();
        this.B.stop();
        this.v.setVisibility(8);
        setColorViewAlpha(255);
        setTargetOffsetTopAndBottom(this.y - this.p);
        this.p = this.v.getTop();
    }

    @Override // android.view.View
    public boolean hasNestedScrollingParent() {
        return this.k.g(0);
    }

    public final void i(boolean z, boolean z2) {
        if (this.f494f != z) {
            this.G = z2;
            b();
            this.f494f = z;
            if (z) {
                int i = this.p;
                Animation.AnimationListener animationListener = this.J;
                this.x = i;
                this.K.reset();
                this.K.setDuration(200L);
                this.K.setInterpolator(this.u);
                if (animationListener != null) {
                    this.v.f2827b = animationListener;
                }
                this.v.clearAnimation();
                this.v.startAnimation(this.K);
                return;
            }
            l(this.J);
        }
    }

    @Override // android.view.View
    public boolean isNestedScrollingEnabled() {
        return this.k.f2207d;
    }

    public final Animation j(int i, int i2) {
        c cVar = new c(i, i2);
        cVar.setDuration(300L);
        b.y.a.a aVar = this.v;
        aVar.f2827b = null;
        aVar.clearAnimation();
        this.v.startAnimation(cVar);
        return cVar;
    }

    public final void k(float f2) {
        float f3 = this.r;
        int i = this.f495g;
        if (f2 - f3 <= i || this.s) {
            return;
        }
        this.q = f3 + i;
        this.s = true;
        this.B.setAlpha(76);
    }

    public void l(Animation.AnimationListener animationListener) {
        b bVar = new b();
        this.D = bVar;
        bVar.setDuration(150L);
        b.y.a.a aVar = this.v;
        aVar.f2827b = animationListener;
        aVar.clearAnimation();
        this.v.startAnimation(this.D);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        h();
    }

    @Override // android.view.ViewGroup
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        b();
        int actionMasked = motionEvent.getActionMasked();
        if (!isEnabled() || a() || this.f494f || this.n) {
            return false;
        }
        if (actionMasked != 0) {
            if (actionMasked != 1) {
                if (actionMasked == 2) {
                    int i = this.t;
                    if (i == -1) {
                        Log.e(f490b, "Got ACTION_MOVE event but don't have an active pointer id.");
                        return false;
                    }
                    int findPointerIndex = motionEvent.findPointerIndex(i);
                    if (findPointerIndex < 0) {
                        return false;
                    }
                    k(motionEvent.getY(findPointerIndex));
                } else if (actionMasked != 3) {
                    if (actionMasked == 6) {
                        g(motionEvent);
                    }
                }
            }
            this.s = false;
            this.t = -1;
        } else {
            setTargetOffsetTopAndBottom(this.y - this.v.getTop());
            int pointerId = motionEvent.getPointerId(0);
            this.t = pointerId;
            this.s = false;
            int findPointerIndex2 = motionEvent.findPointerIndex(pointerId);
            if (findPointerIndex2 < 0) {
                return false;
            }
            this.r = motionEvent.getY(findPointerIndex2);
        }
        return this.s;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        int measuredWidth = getMeasuredWidth();
        int measuredHeight = getMeasuredHeight();
        if (getChildCount() == 0) {
            return;
        }
        if (this.f492d == null) {
            b();
        }
        View view = this.f492d;
        if (view == null) {
            return;
        }
        int paddingLeft = getPaddingLeft();
        int paddingTop = getPaddingTop();
        view.layout(paddingLeft, paddingTop, ((measuredWidth - getPaddingLeft()) - getPaddingRight()) + paddingLeft, ((measuredHeight - getPaddingTop()) - getPaddingBottom()) + paddingTop);
        int measuredWidth2 = this.v.getMeasuredWidth();
        int measuredHeight2 = this.v.getMeasuredHeight();
        int i5 = measuredWidth / 2;
        int i6 = measuredWidth2 / 2;
        int i7 = this.p;
        this.v.layout(i5 - i6, i7, i5 + i6, measuredHeight2 + i7);
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        if (this.f492d == null) {
            b();
        }
        View view = this.f492d;
        if (view == null) {
            return;
        }
        view.measure(View.MeasureSpec.makeMeasureSpec((getMeasuredWidth() - getPaddingLeft()) - getPaddingRight(), 1073741824), View.MeasureSpec.makeMeasureSpec((getMeasuredHeight() - getPaddingTop()) - getPaddingBottom(), 1073741824));
        this.v.measure(View.MeasureSpec.makeMeasureSpec(this.H, 1073741824), View.MeasureSpec.makeMeasureSpec(this.H, 1073741824));
        this.w = -1;
        for (int i3 = 0; i3 < getChildCount(); i3++) {
            if (getChildAt(i3) == this.v) {
                this.w = i3;
                return;
            }
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedFling(View view, float f2, float f3, boolean z) {
        return dispatchNestedFling(f2, f3, z);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onNestedPreFling(View view, float f2, float f3) {
        return dispatchNestedPreFling(f2, f3);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedPreScroll(View view, int i, int i2, int[] iArr) {
        if (i2 > 0) {
            float f2 = this.i;
            if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                float f3 = i2;
                if (f3 > f2) {
                    iArr[1] = i2 - ((int) f2);
                    this.i = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                } else {
                    this.i = f2 - f3;
                    iArr[1] = i2;
                }
                e(this.i);
            }
        }
        int[] iArr2 = this.l;
        if (dispatchNestedPreScroll(i - iArr[0], i2 - iArr[1], iArr2, null)) {
            iArr[0] = iArr[0] + iArr2[0];
            iArr[1] = iArr[1] + iArr2[1];
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScroll(View view, int i, int i2, int i3, int i4) {
        dispatchNestedScroll(i, i2, i3, i4, this.m);
        int i5 = i4 + this.m[1];
        if (i5 >= 0 || a()) {
            return;
        }
        float abs = this.i + Math.abs(i5);
        this.i = abs;
        e(abs);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onNestedScrollAccepted(View view, View view2, int i) {
        this.j.f2209a = i;
        startNestedScroll(i & 2);
        this.i = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        this.n = true;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public boolean onStartNestedScroll(View view, View view2, int i) {
        return (!isEnabled() || this.f494f || (i & 2) == 0) ? false : true;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void onStopNestedScroll(View view) {
        this.j.b(0);
        this.n = false;
        float f2 = this.i;
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            c(f2);
            this.i = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }
        stopNestedScroll();
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int actionMasked = motionEvent.getActionMasked();
        if (!isEnabled() || a() || this.f494f || this.n) {
            return false;
        }
        if (actionMasked == 0) {
            this.t = motionEvent.getPointerId(0);
            this.s = false;
        } else if (actionMasked == 1) {
            int findPointerIndex = motionEvent.findPointerIndex(this.t);
            if (findPointerIndex < 0) {
                Log.e(f490b, "Got ACTION_UP event but don't have an active pointer id.");
                return false;
            }
            if (this.s) {
                this.s = false;
                c((motionEvent.getY(findPointerIndex) - this.q) * 0.5f);
            }
            this.t = -1;
            return false;
        } else if (actionMasked == 2) {
            int findPointerIndex2 = motionEvent.findPointerIndex(this.t);
            if (findPointerIndex2 < 0) {
                Log.e(f490b, "Got ACTION_MOVE event but have an invalid active pointer id.");
                return false;
            }
            float y = motionEvent.getY(findPointerIndex2);
            k(y);
            if (this.s) {
                float f2 = (y - this.q) * 0.5f;
                if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                    return false;
                }
                e(f2);
            }
        } else if (actionMasked == 3) {
            return false;
        } else {
            if (actionMasked == 5) {
                int actionIndex = motionEvent.getActionIndex();
                if (actionIndex < 0) {
                    Log.e(f490b, "Got ACTION_POINTER_DOWN event but have an invalid action index.");
                    return false;
                }
                this.t = motionEvent.getPointerId(actionIndex);
            } else if (actionMasked == 6) {
                g(motionEvent);
            }
        }
        return true;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestDisallowInterceptTouchEvent(boolean z) {
        View view = this.f492d;
        if (view != null) {
            AtomicInteger atomicInteger = q.f2214a;
            if (!view.isNestedScrollingEnabled()) {
                return;
            }
        }
        super.requestDisallowInterceptTouchEvent(z);
    }

    public void setAnimationProgress(float f2) {
        this.v.setScaleX(f2);
        this.v.setScaleY(f2);
    }

    @Deprecated
    public void setColorScheme(int... iArr) {
        setColorSchemeResources(iArr);
    }

    public void setColorSchemeColors(int... iArr) {
        b();
        b.y.a.d dVar = this.B;
        d.a aVar = dVar.f2836e;
        aVar.i = iArr;
        aVar.a(0);
        dVar.f2836e.a(0);
        dVar.invalidateSelf();
    }

    public void setColorSchemeResources(int... iArr) {
        Context context = getContext();
        int[] iArr2 = new int[iArr.length];
        for (int i = 0; i < iArr.length; i++) {
            int i2 = iArr[i];
            Object obj = b.j.c.a.f2074a;
            iArr2[i] = context.getColor(i2);
        }
        setColorSchemeColors(iArr2);
    }

    public void setDistanceToTriggerSync(int i) {
        this.f496h = i;
    }

    @Override // android.view.View
    public void setEnabled(boolean z) {
        super.setEnabled(z);
        if (z) {
            return;
        }
        h();
    }

    @Override // android.view.View
    public void setNestedScrollingEnabled(boolean z) {
        b.j.j.f fVar = this.k;
        if (fVar.f2207d) {
            View view = fVar.f2206c;
            AtomicInteger atomicInteger = q.f2214a;
            view.stopNestedScroll();
        }
        fVar.f2207d = z;
    }

    public void setOnChildScrollUpCallback(g gVar) {
        this.I = gVar;
    }

    public void setOnRefreshListener(h hVar) {
        this.f493e = hVar;
    }

    @Deprecated
    public void setProgressBackgroundColor(int i) {
        setProgressBackgroundColorSchemeResource(i);
    }

    public void setProgressBackgroundColorSchemeColor(int i) {
        this.v.setBackgroundColor(i);
    }

    public void setProgressBackgroundColorSchemeResource(int i) {
        Context context = getContext();
        Object obj = b.j.c.a.f2074a;
        setProgressBackgroundColorSchemeColor(context.getColor(i));
    }

    public void setRefreshing(boolean z) {
        if (z && this.f494f != z) {
            this.f494f = z;
            setTargetOffsetTopAndBottom((this.z + this.y) - this.p);
            this.G = false;
            Animation.AnimationListener animationListener = this.J;
            this.v.setVisibility(0);
            this.B.setAlpha(255);
            b.y.a.e eVar = new b.y.a.e(this);
            this.C = eVar;
            eVar.setDuration(this.o);
            if (animationListener != null) {
                this.v.f2827b = animationListener;
            }
            this.v.clearAnimation();
            this.v.startAnimation(this.C);
            return;
        }
        i(z, false);
    }

    public void setSize(int i) {
        if (i == 0 || i == 1) {
            DisplayMetrics displayMetrics = getResources().getDisplayMetrics();
            if (i == 0) {
                this.H = (int) (displayMetrics.density * 56.0f);
            } else {
                this.H = (int) (displayMetrics.density * 40.0f);
            }
            this.v.setImageDrawable(null);
            this.B.c(i);
            this.v.setImageDrawable(this.B);
        }
    }

    public void setSlingshotDistance(int i) {
        this.A = i;
    }

    public void setTargetOffsetTopAndBottom(int i) {
        this.v.bringToFront();
        b.y.a.a aVar = this.v;
        AtomicInteger atomicInteger = q.f2214a;
        aVar.offsetTopAndBottom(i);
        this.p = this.v.getTop();
    }

    @Override // android.view.View
    public boolean startNestedScroll(int i) {
        return this.k.h(i, 0);
    }

    @Override // android.view.View
    public void stopNestedScroll() {
        this.k.i(0);
    }
}