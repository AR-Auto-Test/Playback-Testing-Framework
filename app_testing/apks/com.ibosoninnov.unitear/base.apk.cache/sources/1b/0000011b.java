package androidx.viewpager.widget;

import android.content.Context;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.database.DataSetObserver;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.FocusFinder;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.SoundEffectConstants;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.accessibility.AccessibilityEvent;
import android.view.animation.Interpolator;
import android.widget.EdgeEffect;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Scroller;
import b.j.j.q;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.lang.annotation.ElementType;
import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class ViewPager extends ViewGroup {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f506b = {16842931};

    /* renamed from: c  reason: collision with root package name */
    public static final Comparator<e> f507c = new a();

    /* renamed from: d  reason: collision with root package name */
    public static final Interpolator f508d = new b();

    /* renamed from: e  reason: collision with root package name */
    public static final l f509e = new l();
    public boolean A;
    public int B;
    public boolean C;
    public boolean D;
    public int E;
    public int F;
    public int G;
    public float H;
    public float I;
    public float J;
    public float K;
    public int L;
    public VelocityTracker M;
    public int N;
    public int O;
    public int P;
    public int Q;
    public EdgeEffect R;
    public EdgeEffect S;
    public boolean T;
    public boolean U;
    public int V;
    public List<i> W;
    public i a0;
    public List<h> b0;
    public final Runnable c0;
    public int d0;

    /* renamed from: f  reason: collision with root package name */
    public int f510f;

    /* renamed from: g  reason: collision with root package name */
    public final ArrayList<e> f511g;

    /* renamed from: h  reason: collision with root package name */
    public final e f512h;
    public final Rect i;
    public b.c0.a.a j;
    public int k;
    public int l;
    public Parcelable m;
    public ClassLoader n;
    public Scroller o;
    public boolean p;
    public j q;
    public int r;
    public Drawable s;
    public int t;
    public int u;
    public float v;
    public float w;
    public int x;
    public boolean y;
    public boolean z;

    /* loaded from: classes.dex */
    public static class a implements Comparator<e> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(e eVar, e eVar2) {
            return eVar.f515b - eVar2.f515b;
        }
    }

    /* loaded from: classes.dex */
    public static class b implements Interpolator {
        @Override // android.animation.TimeInterpolator
        public float getInterpolation(float f2) {
            float f3 = f2 - 1.0f;
            return (f3 * f3 * f3 * f3 * f3) + 1.0f;
        }
    }

    /* loaded from: classes.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            ViewPager.this.setScrollState(0);
            ViewPager viewPager = ViewPager.this;
            viewPager.q(viewPager.k);
        }
    }

    @Target({ElementType.TYPE})
    @Inherited
    @Retention(RetentionPolicy.RUNTIME)
    /* loaded from: classes.dex */
    public @interface d {
    }

    /* loaded from: classes.dex */
    public static class e {

        /* renamed from: a  reason: collision with root package name */
        public Object f514a;

        /* renamed from: b  reason: collision with root package name */
        public int f515b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f516c;

        /* renamed from: d  reason: collision with root package name */
        public float f517d;

        /* renamed from: e  reason: collision with root package name */
        public float f518e;
    }

    /* loaded from: classes.dex */
    public class g extends b.j.j.a {
        public g() {
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            b.c0.a.a aVar;
            super.onInitializeAccessibilityEvent(view, accessibilityEvent);
            accessibilityEvent.setClassName(ViewPager.class.getName());
            b.c0.a.a aVar2 = ViewPager.this.j;
            boolean z = true;
            accessibilityEvent.setScrollable((aVar2 == null || aVar2.a() <= 1) ? false : false);
            if (accessibilityEvent.getEventType() != 4096 || (aVar = ViewPager.this.j) == null) {
                return;
            }
            accessibilityEvent.setItemCount(aVar.a());
            accessibilityEvent.setFromIndex(ViewPager.this.k);
            accessibilityEvent.setToIndex(ViewPager.this.k);
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
            super.onInitializeAccessibilityNodeInfo(view, bVar);
            bVar.f2259b.setClassName(ViewPager.class.getName());
            b.c0.a.a aVar = ViewPager.this.j;
            bVar.f2259b.setScrollable(aVar != null && aVar.a() > 1);
            if (ViewPager.this.canScrollHorizontally(1)) {
                bVar.f2259b.addAction(4096);
            }
            if (ViewPager.this.canScrollHorizontally(-1)) {
                bVar.f2259b.addAction(8192);
            }
        }

        @Override // b.j.j.a
        public boolean performAccessibilityAction(View view, int i, Bundle bundle) {
            if (super.performAccessibilityAction(view, i, bundle)) {
                return true;
            }
            if (i != 4096) {
                if (i == 8192 && ViewPager.this.canScrollHorizontally(-1)) {
                    ViewPager viewPager = ViewPager.this;
                    viewPager.setCurrentItem(viewPager.k - 1);
                    return true;
                }
                return false;
            } else if (ViewPager.this.canScrollHorizontally(1)) {
                ViewPager viewPager2 = ViewPager.this;
                viewPager2.setCurrentItem(viewPager2.k + 1);
                return true;
            } else {
                return false;
            }
        }
    }

    /* loaded from: classes.dex */
    public interface h {
        void onAdapterChanged(ViewPager viewPager, b.c0.a.a aVar, b.c0.a.a aVar2);
    }

    /* loaded from: classes.dex */
    public interface i {
        void onPageScrollStateChanged(int i);

        void onPageScrolled(int i, float f2, int i2);

        void onPageSelected(int i);
    }

    /* loaded from: classes.dex */
    public class j extends DataSetObserver {
        public j() {
        }

        @Override // android.database.DataSetObserver
        public void onChanged() {
            ViewPager.this.e();
        }

        @Override // android.database.DataSetObserver
        public void onInvalidated() {
            ViewPager.this.e();
        }
    }

    /* loaded from: classes.dex */
    public static class k extends b.l.a.a {
        public static final Parcelable.Creator<k> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f527b;

        /* renamed from: c  reason: collision with root package name */
        public Parcelable f528c;

        /* renamed from: d  reason: collision with root package name */
        public ClassLoader f529d;

        /* loaded from: classes.dex */
        public static class a implements Parcelable.ClassLoaderCreator<k> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.ClassLoaderCreator
            public k createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new k(parcel, classLoader);
            }

            @Override // android.os.Parcelable.Creator
            public Object[] newArray(int i) {
                return new k[i];
            }

            @Override // android.os.Parcelable.Creator
            public Object createFromParcel(Parcel parcel) {
                return new k(parcel, null);
            }
        }

        public k(Parcelable parcelable) {
            super(parcelable);
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("FragmentPager.SavedState{");
            x.append(Integer.toHexString(System.identityHashCode(this)));
            x.append(" position=");
            return c.b.a.a.a.s(x, this.f527b, "}");
        }

        @Override // b.l.a.a, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeInt(this.f527b);
            parcel.writeParcelable(this.f528c, i);
        }

        public k(Parcel parcel, ClassLoader classLoader) {
            super(parcel, classLoader);
            classLoader = classLoader == null ? k.class.getClassLoader() : classLoader;
            this.f527b = parcel.readInt();
            this.f528c = parcel.readParcelable(classLoader);
            this.f529d = classLoader;
        }
    }

    /* loaded from: classes.dex */
    public static class l implements Comparator<View> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(View view, View view2) {
            f fVar = (f) view.getLayoutParams();
            f fVar2 = (f) view2.getLayoutParams();
            boolean z = fVar.f519a;
            if (z != fVar2.f519a) {
                return z ? 1 : -1;
            }
            return fVar.f523e - fVar2.f523e;
        }
    }

    public ViewPager(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.f511g = new ArrayList<>();
        this.f512h = new e();
        this.i = new Rect();
        this.l = -1;
        this.m = null;
        this.n = null;
        this.v = -3.4028235E38f;
        this.w = Float.MAX_VALUE;
        this.B = 1;
        this.L = -1;
        this.T = true;
        this.c0 = new c();
        this.d0 = 0;
        setWillNotDraw(false);
        setDescendantFocusability(Calib3d.CALIB_TILTED_MODEL);
        setFocusable(true);
        Context context2 = getContext();
        this.o = new Scroller(context2, f508d);
        ViewConfiguration viewConfiguration = ViewConfiguration.get(context2);
        float f2 = context2.getResources().getDisplayMetrics().density;
        this.G = viewConfiguration.getScaledPagingTouchSlop();
        this.N = (int) (400.0f * f2);
        this.O = viewConfiguration.getScaledMaximumFlingVelocity();
        this.R = new EdgeEffect(context2);
        this.S = new EdgeEffect(context2);
        this.P = (int) (25.0f * f2);
        this.Q = (int) (2.0f * f2);
        this.E = (int) (f2 * 16.0f);
        q.n(this, new g());
        if (getImportantForAccessibility() == 0) {
            setImportantForAccessibility(1);
        }
        q.b.c(this, new b.c0.a.b(this));
    }

    private int getClientWidth() {
        return (getMeasuredWidth() - getPaddingLeft()) - getPaddingRight();
    }

    private void setScrollingCacheEnabled(boolean z) {
        if (this.z != z) {
            this.z = z;
        }
    }

    public e a(int i2, int i3) {
        e eVar = new e();
        eVar.f515b = i2;
        c.e.b.ef.g gVar = (c.e.b.ef.g) this.j;
        View inflate = LayoutInflater.from(gVar.f4728b).inflate(R.layout.tutorial_item, (ViewGroup) this, false);
        ImageView imageView = (ImageView) inflate.findViewById(R.id.iv_onboard);
        int i4 = gVar.f4729c.get(i2).imageID;
        if (i4 != -1) {
            imageView.setImageResource(i4);
        } else {
            imageView.setImageBitmap(null);
        }
        addView(inflate);
        inflate.setTag("tutorialview" + i2);
        eVar.f514a = inflate;
        Objects.requireNonNull(this.j);
        eVar.f517d = 1.0f;
        if (i3 >= 0 && i3 < this.f511g.size()) {
            this.f511g.add(i3, eVar);
        } else {
            this.f511g.add(eVar);
        }
        return eVar;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void addFocusables(ArrayList<View> arrayList, int i2, int i3) {
        e h2;
        int size = arrayList.size();
        int descendantFocusability = getDescendantFocusability();
        if (descendantFocusability != 393216) {
            for (int i4 = 0; i4 < getChildCount(); i4++) {
                View childAt = getChildAt(i4);
                if (childAt.getVisibility() == 0 && (h2 = h(childAt)) != null && h2.f515b == this.k) {
                    childAt.addFocusables(arrayList, i2, i3);
                }
            }
        }
        if ((descendantFocusability != 262144 || size == arrayList.size()) && isFocusable()) {
            if ((i3 & 1) == 1 && isInTouchMode() && !isFocusableInTouchMode()) {
                return;
            }
            arrayList.add(this);
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void addTouchables(ArrayList<View> arrayList) {
        e h2;
        for (int i2 = 0; i2 < getChildCount(); i2++) {
            View childAt = getChildAt(i2);
            if (childAt.getVisibility() == 0 && (h2 = h(childAt)) != null && h2.f515b == this.k) {
                childAt.addTouchables(arrayList);
            }
        }
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i2, ViewGroup.LayoutParams layoutParams) {
        if (!checkLayoutParams(layoutParams)) {
            layoutParams = generateLayoutParams(layoutParams);
        }
        f fVar = (f) layoutParams;
        boolean z = fVar.f519a | (view.getClass().getAnnotation(d.class) != null);
        fVar.f519a = z;
        if (!this.y) {
            super.addView(view, i2, layoutParams);
        } else if (!z) {
            fVar.f522d = true;
            addViewInLayout(view, i2, layoutParams);
        } else {
            throw new IllegalStateException("Cannot add pager decor view during layout");
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:45:0x00ca  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean b(int i2) {
        View findNextFocus;
        boolean requestFocus;
        boolean z;
        View findFocus = findFocus();
        boolean z2 = false;
        if (findFocus != this) {
            if (findFocus != null) {
                ViewParent parent = findFocus.getParent();
                while (true) {
                    if (!(parent instanceof ViewGroup)) {
                        z = false;
                        break;
                    } else if (parent == this) {
                        z = true;
                        break;
                    } else {
                        parent = parent.getParent();
                    }
                }
                if (!z) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(findFocus.getClass().getSimpleName());
                    for (ViewParent parent2 = findFocus.getParent(); parent2 instanceof ViewGroup; parent2 = parent2.getParent()) {
                        sb.append(" => ");
                        sb.append(parent2.getClass().getSimpleName());
                    }
                    StringBuilder x = c.b.a.a.a.x("arrowScroll tried to find focus based on non-child current focused view ");
                    x.append(sb.toString());
                    Log.e("ViewPager", x.toString());
                }
            }
            findNextFocus = FocusFinder.getInstance().findNextFocus(this, findFocus, i2);
            if (findNextFocus == null && findNextFocus != findFocus) {
                if (i2 == 17) {
                    int i3 = g(this.i, findNextFocus).left;
                    int i4 = g(this.i, findFocus).left;
                    if (findFocus != null && i3 >= i4) {
                        requestFocus = m();
                    } else {
                        requestFocus = findNextFocus.requestFocus();
                    }
                } else if (i2 == 66) {
                    int i5 = g(this.i, findNextFocus).left;
                    int i6 = g(this.i, findFocus).left;
                    if (findFocus != null && i5 <= i6) {
                        requestFocus = n();
                    } else {
                        requestFocus = findNextFocus.requestFocus();
                    }
                }
                z2 = requestFocus;
            } else if (i2 != 17 || i2 == 1) {
                z2 = m();
            } else if (i2 == 66 || i2 == 2) {
                z2 = n();
            }
            if (z2) {
                playSoundEffect(SoundEffectConstants.getContantForFocusDirection(i2));
            }
            return z2;
        }
        findFocus = null;
        findNextFocus = FocusFinder.getInstance().findNextFocus(this, findFocus, i2);
        if (findNextFocus == null) {
        }
        if (i2 != 17) {
        }
        z2 = m();
        if (z2) {
        }
        return z2;
    }

    public boolean c(View view, boolean z, int i2, int i3, int i4) {
        int i5;
        if (view instanceof ViewGroup) {
            ViewGroup viewGroup = (ViewGroup) view;
            int scrollX = view.getScrollX();
            int scrollY = view.getScrollY();
            for (int childCount = viewGroup.getChildCount() - 1; childCount >= 0; childCount--) {
                View childAt = viewGroup.getChildAt(childCount);
                int i6 = i3 + scrollX;
                if (i6 >= childAt.getLeft() && i6 < childAt.getRight() && (i5 = i4 + scrollY) >= childAt.getTop() && i5 < childAt.getBottom() && c(childAt, true, i2, i6 - childAt.getLeft(), i5 - childAt.getTop())) {
                    return true;
                }
            }
        }
        return z && view.canScrollHorizontally(-i2);
    }

    @Override // android.view.View
    public boolean canScrollHorizontally(int i2) {
        if (this.j == null) {
            return false;
        }
        int clientWidth = getClientWidth();
        int scrollX = getScrollX();
        return i2 < 0 ? scrollX > ((int) (((float) clientWidth) * this.v)) : i2 > 0 && scrollX < ((int) (((float) clientWidth) * this.w));
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return (layoutParams instanceof f) && super.checkLayoutParams(layoutParams);
    }

    @Override // android.view.View
    public void computeScroll() {
        this.p = true;
        if (!this.o.isFinished() && this.o.computeScrollOffset()) {
            int scrollX = getScrollX();
            int scrollY = getScrollY();
            int currX = this.o.getCurrX();
            int currY = this.o.getCurrY();
            if (scrollX != currX || scrollY != currY) {
                scrollTo(currX, currY);
                if (!o(currX)) {
                    this.o.abortAnimation();
                    scrollTo(0, currY);
                }
            }
            AtomicInteger atomicInteger = q.f2214a;
            postInvalidateOnAnimation();
            return;
        }
        d(true);
    }

    public final void d(boolean z) {
        boolean z2 = this.d0 == 2;
        if (z2) {
            setScrollingCacheEnabled(false);
            if (!this.o.isFinished()) {
                this.o.abortAnimation();
                int scrollX = getScrollX();
                int scrollY = getScrollY();
                int currX = this.o.getCurrX();
                int currY = this.o.getCurrY();
                if (scrollX != currX || scrollY != currY) {
                    scrollTo(currX, currY);
                    if (currX != scrollX) {
                        o(currX);
                    }
                }
            }
        }
        this.A = false;
        for (int i2 = 0; i2 < this.f511g.size(); i2++) {
            e eVar = this.f511g.get(i2);
            if (eVar.f516c) {
                eVar.f516c = false;
                z2 = true;
            }
        }
        if (z2) {
            if (z) {
                Runnable runnable = this.c0;
                AtomicInteger atomicInteger = q.f2214a;
                postOnAnimation(runnable);
                return;
            }
            this.c0.run();
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:31:? A[RETURN, SYNTHETIC] */
    @Override // android.view.ViewGroup, android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean dispatchKeyEvent(KeyEvent keyEvent) {
        boolean z;
        if (!super.dispatchKeyEvent(keyEvent)) {
            if (keyEvent.getAction() == 0) {
                int keyCode = keyEvent.getKeyCode();
                if (keyCode != 21) {
                    if (keyCode != 22) {
                        if (keyCode == 61) {
                            if (keyEvent.hasNoModifiers()) {
                                z = b(2);
                            } else if (keyEvent.hasModifiers(1)) {
                                z = b(1);
                            }
                        }
                    } else if (keyEvent.hasModifiers(2)) {
                        z = n();
                    } else {
                        z = b(66);
                    }
                } else if (keyEvent.hasModifiers(2)) {
                    z = m();
                } else {
                    z = b(17);
                }
                if (!z) {
                    return false;
                }
            }
            z = false;
            if (!z) {
            }
        }
        return true;
    }

    @Override // android.view.View
    public boolean dispatchPopulateAccessibilityEvent(AccessibilityEvent accessibilityEvent) {
        e h2;
        if (accessibilityEvent.getEventType() == 4096) {
            return super.dispatchPopulateAccessibilityEvent(accessibilityEvent);
        }
        int childCount = getChildCount();
        for (int i2 = 0; i2 < childCount; i2++) {
            View childAt = getChildAt(i2);
            if (childAt.getVisibility() == 0 && (h2 = h(childAt)) != null && h2.f515b == this.k && childAt.dispatchPopulateAccessibilityEvent(accessibilityEvent)) {
                return true;
            }
        }
        return false;
    }

    @Override // android.view.View
    public void draw(Canvas canvas) {
        b.c0.a.a aVar;
        super.draw(canvas);
        int overScrollMode = getOverScrollMode();
        boolean z = false;
        if (overScrollMode != 0 && (overScrollMode != 1 || (aVar = this.j) == null || aVar.a() <= 1)) {
            this.R.finish();
            this.S.finish();
        } else {
            if (!this.R.isFinished()) {
                int save = canvas.save();
                int height = (getHeight() - getPaddingTop()) - getPaddingBottom();
                int width = getWidth();
                canvas.rotate(270.0f);
                canvas.translate(getPaddingTop() + (-height), this.v * width);
                this.R.setSize(height, width);
                z = false | this.R.draw(canvas);
                canvas.restoreToCount(save);
            }
            if (!this.S.isFinished()) {
                int save2 = canvas.save();
                int width2 = getWidth();
                int height2 = (getHeight() - getPaddingTop()) - getPaddingBottom();
                canvas.rotate(90.0f);
                canvas.translate(-getPaddingTop(), (-(this.w + 1.0f)) * width2);
                this.S.setSize(height2, width2);
                z |= this.S.draw(canvas);
                canvas.restoreToCount(save2);
            }
        }
        if (z) {
            AtomicInteger atomicInteger = q.f2214a;
            postInvalidateOnAnimation();
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        Drawable drawable = this.s;
        if (drawable == null || !drawable.isStateful()) {
            return;
        }
        drawable.setState(getDrawableState());
    }

    public void e() {
        int a2 = this.j.a();
        this.f510f = a2;
        boolean z = this.f511g.size() < (this.B * 2) + 1 && this.f511g.size() < a2;
        int i2 = this.k;
        for (int i3 = 0; i3 < this.f511g.size(); i3++) {
            b.c0.a.a aVar = this.j;
            Object obj = this.f511g.get(i3).f514a;
            Objects.requireNonNull(aVar);
        }
        Collections.sort(this.f511g, f507c);
        if (z) {
            int childCount = getChildCount();
            for (int i4 = 0; i4 < childCount; i4++) {
                f fVar = (f) getChildAt(i4).getLayoutParams();
                if (!fVar.f519a) {
                    fVar.f521c = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                }
            }
            v(i2, false, true, 0);
            requestLayout();
        }
    }

    public final void f(int i2) {
        i iVar = this.a0;
        if (iVar != null) {
            iVar.onPageSelected(i2);
        }
        List<i> list = this.W;
        if (list != null) {
            int size = list.size();
            for (int i3 = 0; i3 < size; i3++) {
                i iVar2 = this.W.get(i3);
                if (iVar2 != null) {
                    iVar2.onPageSelected(i2);
                }
            }
        }
    }

    public final Rect g(Rect rect, View view) {
        if (rect == null) {
            rect = new Rect();
        }
        if (view == null) {
            rect.set(0, 0, 0, 0);
            return rect;
        }
        rect.left = view.getLeft();
        rect.right = view.getRight();
        rect.top = view.getTop();
        rect.bottom = view.getBottom();
        ViewParent parent = view.getParent();
        while ((parent instanceof ViewGroup) && parent != this) {
            ViewGroup viewGroup = (ViewGroup) parent;
            rect.left = viewGroup.getLeft() + rect.left;
            rect.right = viewGroup.getRight() + rect.right;
            rect.top = viewGroup.getTop() + rect.top;
            rect.bottom = viewGroup.getBottom() + rect.bottom;
            parent = viewGroup.getParent();
        }
        return rect;
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateDefaultLayoutParams() {
        return new f();
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return generateDefaultLayoutParams();
    }

    public b.c0.a.a getAdapter() {
        return this.j;
    }

    @Override // android.view.ViewGroup
    public int getChildDrawingOrder(int i2, int i3) {
        throw null;
    }

    public int getCurrentItem() {
        return this.k;
    }

    public int getOffscreenPageLimit() {
        return this.B;
    }

    public int getPageMargin() {
        return this.r;
    }

    public e h(View view) {
        for (int i2 = 0; i2 < this.f511g.size(); i2++) {
            e eVar = this.f511g.get(i2);
            b.c0.a.a aVar = this.j;
            Object obj = eVar.f514a;
            Objects.requireNonNull((c.e.b.ef.g) aVar);
            if (view == obj) {
                return eVar;
            }
        }
        return null;
    }

    public final e i() {
        int i2;
        int clientWidth = getClientWidth();
        float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        float scrollX = clientWidth > 0 ? getScrollX() / clientWidth : 0.0f;
        float f3 = clientWidth > 0 ? this.r / clientWidth : 0.0f;
        e eVar = null;
        int i3 = 0;
        int i4 = -1;
        boolean z = true;
        float f4 = 0.0f;
        while (i3 < this.f511g.size()) {
            e eVar2 = this.f511g.get(i3);
            if (!z && eVar2.f515b != (i2 = i4 + 1)) {
                eVar2 = this.f512h;
                eVar2.f518e = f2 + f4 + f3;
                eVar2.f515b = i2;
                Objects.requireNonNull(this.j);
                eVar2.f517d = 1.0f;
                i3--;
            }
            f2 = eVar2.f518e;
            float f5 = eVar2.f517d + f2 + f3;
            if (!z && scrollX < f2) {
                return eVar;
            }
            if (scrollX < f5 || i3 == this.f511g.size() - 1) {
                return eVar2;
            }
            i4 = eVar2.f515b;
            f4 = eVar2.f517d;
            i3++;
            z = false;
            eVar = eVar2;
        }
        return eVar;
    }

    public e j(int i2) {
        for (int i3 = 0; i3 < this.f511g.size(); i3++) {
            e eVar = this.f511g.get(i3);
            if (eVar.f515b == i2) {
                return eVar;
            }
        }
        return null;
    }

    /* JADX WARN: Removed duplicated region for block: B:22:0x0064  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void k(int i2, float f2, int i3) {
        int max;
        int i4;
        int left;
        if (this.V > 0) {
            int scrollX = getScrollX();
            int paddingLeft = getPaddingLeft();
            int paddingRight = getPaddingRight();
            int width = getWidth();
            int childCount = getChildCount();
            for (int i5 = 0; i5 < childCount; i5++) {
                View childAt = getChildAt(i5);
                f fVar = (f) childAt.getLayoutParams();
                if (fVar.f519a) {
                    int i6 = fVar.f520b & 7;
                    if (i6 == 1) {
                        max = Math.max((width - childAt.getMeasuredWidth()) / 2, paddingLeft);
                    } else {
                        if (i6 == 3) {
                            i4 = childAt.getWidth() + paddingLeft;
                        } else if (i6 != 5) {
                            i4 = paddingLeft;
                        } else {
                            max = (width - paddingRight) - childAt.getMeasuredWidth();
                            paddingRight += childAt.getMeasuredWidth();
                        }
                        left = (paddingLeft + scrollX) - childAt.getLeft();
                        if (left != 0) {
                            childAt.offsetLeftAndRight(left);
                        }
                        paddingLeft = i4;
                    }
                    int i7 = max;
                    i4 = paddingLeft;
                    paddingLeft = i7;
                    left = (paddingLeft + scrollX) - childAt.getLeft();
                    if (left != 0) {
                    }
                    paddingLeft = i4;
                }
            }
        }
        i iVar = this.a0;
        if (iVar != null) {
            iVar.onPageScrolled(i2, f2, i3);
        }
        List<i> list = this.W;
        if (list != null) {
            int size = list.size();
            for (int i8 = 0; i8 < size; i8++) {
                i iVar2 = this.W.get(i8);
                if (iVar2 != null) {
                    iVar2.onPageScrolled(i2, f2, i3);
                }
            }
        }
        this.U = true;
    }

    public final void l(MotionEvent motionEvent) {
        int actionIndex = motionEvent.getActionIndex();
        if (motionEvent.getPointerId(actionIndex) == this.L) {
            int i2 = actionIndex == 0 ? 1 : 0;
            this.H = motionEvent.getX(i2);
            this.L = motionEvent.getPointerId(i2);
            VelocityTracker velocityTracker = this.M;
            if (velocityTracker != null) {
                velocityTracker.clear();
            }
        }
    }

    public boolean m() {
        int i2 = this.k;
        if (i2 > 0) {
            this.A = false;
            v(i2 - 1, true, false, 0);
            return true;
        }
        return false;
    }

    public boolean n() {
        b.c0.a.a aVar = this.j;
        if (aVar == null || this.k >= aVar.a() - 1) {
            return false;
        }
        this.A = false;
        v(this.k + 1, true, false, 0);
        return true;
    }

    public final boolean o(int i2) {
        if (this.f511g.size() == 0) {
            if (this.T) {
                return false;
            }
            this.U = false;
            k(0, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0);
            if (this.U) {
                return false;
            }
            throw new IllegalStateException("onPageScrolled did not call superclass implementation");
        }
        e i3 = i();
        int clientWidth = getClientWidth();
        int i4 = this.r;
        int i5 = clientWidth + i4;
        float f2 = clientWidth;
        int i6 = i3.f515b;
        float f3 = ((i2 / f2) - i3.f518e) / (i3.f517d + (i4 / f2));
        this.U = false;
        k(i6, f3, (int) (i5 * f3));
        if (this.U) {
            return true;
        }
        throw new IllegalStateException("onPageScrolled did not call superclass implementation");
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        this.T = true;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        removeCallbacks(this.c0);
        Scroller scroller = this.o;
        if (scroller != null && !scroller.isFinished()) {
            this.o.abortAnimation();
        }
        super.onDetachedFromWindow();
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
        int width;
        int i2;
        float f2;
        float f3;
        super.onDraw(canvas);
        if (this.r <= 0 || this.s == null || this.f511g.size() <= 0 || this.j == null) {
            return;
        }
        int scrollX = getScrollX();
        float width2 = getWidth();
        float f4 = this.r / width2;
        int i3 = 0;
        e eVar = this.f511g.get(0);
        float f5 = eVar.f518e;
        int size = this.f511g.size();
        int i4 = eVar.f515b;
        int i5 = this.f511g.get(size - 1).f515b;
        while (i4 < i5) {
            while (true) {
                i2 = eVar.f515b;
                if (i4 <= i2 || i3 >= size) {
                    break;
                }
                i3++;
                eVar = this.f511g.get(i3);
            }
            if (i4 == i2) {
                float f6 = eVar.f518e;
                float f7 = eVar.f517d;
                f2 = (f6 + f7) * width2;
                f5 = f6 + f7 + f4;
            } else {
                Objects.requireNonNull(this.j);
                f2 = (f5 + 1.0f) * width2;
                f5 = 1.0f + f4 + f5;
            }
            if (this.r + f2 > scrollX) {
                f3 = f4;
                this.s.setBounds(Math.round(f2), this.t, Math.round(this.r + f2), this.u);
                this.s.draw(canvas);
            } else {
                f3 = f4;
            }
            if (f2 > scrollX + width) {
                return;
            }
            i4++;
            f4 = f3;
        }
    }

    @Override // android.view.ViewGroup
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & 255;
        if (action != 3 && action != 1) {
            if (action != 0) {
                if (this.C) {
                    return true;
                }
                if (this.D) {
                    return false;
                }
            }
            if (action == 0) {
                float x = motionEvent.getX();
                this.J = x;
                this.H = x;
                float y = motionEvent.getY();
                this.K = y;
                this.I = y;
                this.L = motionEvent.getPointerId(0);
                this.D = false;
                this.p = true;
                this.o.computeScrollOffset();
                if (this.d0 == 2 && Math.abs(this.o.getFinalX() - this.o.getCurrX()) > this.Q) {
                    this.o.abortAnimation();
                    this.A = false;
                    q(this.k);
                    this.C = true;
                    s(true);
                    setScrollState(1);
                } else {
                    d(false);
                    this.C = false;
                }
            } else if (action == 2) {
                int i2 = this.L;
                if (i2 != -1) {
                    int findPointerIndex = motionEvent.findPointerIndex(i2);
                    float x2 = motionEvent.getX(findPointerIndex);
                    float f2 = x2 - this.H;
                    float abs = Math.abs(f2);
                    float y2 = motionEvent.getY(findPointerIndex);
                    float abs2 = Math.abs(y2 - this.K);
                    int i3 = (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
                    if (i3 != 0) {
                        float f3 = this.H;
                        if (!((f3 < ((float) this.F) && i3 > 0) || (f3 > ((float) (getWidth() - this.F)) && f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD)) && c(this, false, (int) f2, (int) x2, (int) y2)) {
                            this.H = x2;
                            this.I = y2;
                            this.D = true;
                            return false;
                        }
                    }
                    int i4 = this.G;
                    if (abs > i4 && abs * 0.5f > abs2) {
                        this.C = true;
                        s(true);
                        setScrollState(1);
                        float f4 = this.J;
                        float f5 = this.G;
                        this.H = i3 > 0 ? f4 + f5 : f4 - f5;
                        this.I = y2;
                        setScrollingCacheEnabled(true);
                    } else if (abs2 > i4) {
                        this.D = true;
                    }
                    if (this.C && p(x2)) {
                        AtomicInteger atomicInteger = q.f2214a;
                        postInvalidateOnAnimation();
                    }
                }
            } else if (action == 6) {
                l(motionEvent);
            }
            if (this.M == null) {
                this.M = VelocityTracker.obtain();
            }
            this.M.addMovement(motionEvent);
            return this.C;
        }
        t();
        return false;
    }

    /* JADX WARN: Removed duplicated region for block: B:22:0x0071  */
    /* JADX WARN: Removed duplicated region for block: B:29:0x008e  */
    @Override // android.view.ViewGroup, android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void onLayout(boolean z, int i2, int i3, int i4, int i5) {
        boolean z2;
        e h2;
        int max;
        int i6;
        int max2;
        int i7;
        int childCount = getChildCount();
        int i8 = i4 - i2;
        int i9 = i5 - i3;
        int paddingLeft = getPaddingLeft();
        int paddingTop = getPaddingTop();
        int paddingRight = getPaddingRight();
        int paddingBottom = getPaddingBottom();
        int scrollX = getScrollX();
        int i10 = 0;
        for (int i11 = 0; i11 < childCount; i11++) {
            View childAt = getChildAt(i11);
            if (childAt.getVisibility() != 8) {
                f fVar = (f) childAt.getLayoutParams();
                if (fVar.f519a) {
                    int i12 = fVar.f520b;
                    int i13 = i12 & 7;
                    int i14 = i12 & 112;
                    if (i13 == 1) {
                        max = Math.max((i8 - childAt.getMeasuredWidth()) / 2, paddingLeft);
                    } else {
                        if (i13 == 3) {
                            i6 = childAt.getMeasuredWidth() + paddingLeft;
                        } else if (i13 != 5) {
                            i6 = paddingLeft;
                        } else {
                            max = (i8 - paddingRight) - childAt.getMeasuredWidth();
                            paddingRight += childAt.getMeasuredWidth();
                        }
                        if (i14 != 16) {
                            max2 = Math.max((i9 - childAt.getMeasuredHeight()) / 2, paddingTop);
                        } else {
                            if (i14 == 48) {
                                i7 = childAt.getMeasuredHeight() + paddingTop;
                            } else if (i14 != 80) {
                                i7 = paddingTop;
                            } else {
                                max2 = (i9 - paddingBottom) - childAt.getMeasuredHeight();
                                paddingBottom += childAt.getMeasuredHeight();
                            }
                            int i15 = paddingLeft + scrollX;
                            childAt.layout(i15, paddingTop, childAt.getMeasuredWidth() + i15, childAt.getMeasuredHeight() + paddingTop);
                            i10++;
                            paddingTop = i7;
                            paddingLeft = i6;
                        }
                        int i16 = max2;
                        i7 = paddingTop;
                        paddingTop = i16;
                        int i152 = paddingLeft + scrollX;
                        childAt.layout(i152, paddingTop, childAt.getMeasuredWidth() + i152, childAt.getMeasuredHeight() + paddingTop);
                        i10++;
                        paddingTop = i7;
                        paddingLeft = i6;
                    }
                    int i17 = max;
                    i6 = paddingLeft;
                    paddingLeft = i17;
                    if (i14 != 16) {
                    }
                    int i162 = max2;
                    i7 = paddingTop;
                    paddingTop = i162;
                    int i1522 = paddingLeft + scrollX;
                    childAt.layout(i1522, paddingTop, childAt.getMeasuredWidth() + i1522, childAt.getMeasuredHeight() + paddingTop);
                    i10++;
                    paddingTop = i7;
                    paddingLeft = i6;
                }
            }
        }
        int i18 = (i8 - paddingLeft) - paddingRight;
        for (int i19 = 0; i19 < childCount; i19++) {
            View childAt2 = getChildAt(i19);
            if (childAt2.getVisibility() != 8) {
                f fVar2 = (f) childAt2.getLayoutParams();
                if (!fVar2.f519a && (h2 = h(childAt2)) != null) {
                    float f2 = i18;
                    int i20 = ((int) (h2.f518e * f2)) + paddingLeft;
                    if (fVar2.f522d) {
                        fVar2.f522d = false;
                        childAt2.measure(View.MeasureSpec.makeMeasureSpec((int) (f2 * fVar2.f521c), 1073741824), View.MeasureSpec.makeMeasureSpec((i9 - paddingTop) - paddingBottom, 1073741824));
                    }
                    childAt2.layout(i20, paddingTop, childAt2.getMeasuredWidth() + i20, childAt2.getMeasuredHeight() + paddingTop);
                }
            }
        }
        this.t = paddingTop;
        this.u = i9 - paddingBottom;
        this.V = i10;
        if (this.T) {
            z2 = false;
            u(this.k, false, 0, false);
        } else {
            z2 = false;
        }
        this.T = z2;
    }

    @Override // android.view.View
    public void onMeasure(int i2, int i3) {
        f fVar;
        f fVar2;
        int i4;
        setMeasuredDimension(ViewGroup.getDefaultSize(0, i2), ViewGroup.getDefaultSize(0, i3));
        int measuredWidth = getMeasuredWidth();
        this.F = Math.min(measuredWidth / 10, this.E);
        int paddingLeft = (measuredWidth - getPaddingLeft()) - getPaddingRight();
        int measuredHeight = (getMeasuredHeight() - getPaddingTop()) - getPaddingBottom();
        int childCount = getChildCount();
        int i5 = 0;
        while (true) {
            boolean z = true;
            int i6 = 1073741824;
            if (i5 >= childCount) {
                break;
            }
            View childAt = getChildAt(i5);
            if (childAt.getVisibility() != 8 && (fVar2 = (f) childAt.getLayoutParams()) != null && fVar2.f519a) {
                int i7 = fVar2.f520b;
                int i8 = i7 & 7;
                int i9 = i7 & 112;
                boolean z2 = i9 == 48 || i9 == 80;
                if (i8 != 3 && i8 != 5) {
                    z = false;
                }
                int i10 = Integer.MIN_VALUE;
                if (z2) {
                    i4 = Integer.MIN_VALUE;
                    i10 = 1073741824;
                } else {
                    i4 = z ? 1073741824 : Integer.MIN_VALUE;
                }
                int i11 = ((ViewGroup.LayoutParams) fVar2).width;
                if (i11 != -2) {
                    if (i11 == -1) {
                        i11 = paddingLeft;
                    }
                    i10 = 1073741824;
                } else {
                    i11 = paddingLeft;
                }
                int i12 = ((ViewGroup.LayoutParams) fVar2).height;
                if (i12 == -2) {
                    i12 = measuredHeight;
                    i6 = i4;
                } else if (i12 == -1) {
                    i12 = measuredHeight;
                }
                childAt.measure(View.MeasureSpec.makeMeasureSpec(i11, i10), View.MeasureSpec.makeMeasureSpec(i12, i6));
                if (z2) {
                    measuredHeight -= childAt.getMeasuredHeight();
                } else if (z) {
                    paddingLeft -= childAt.getMeasuredWidth();
                }
            }
            i5++;
        }
        View.MeasureSpec.makeMeasureSpec(paddingLeft, 1073741824);
        this.x = View.MeasureSpec.makeMeasureSpec(measuredHeight, 1073741824);
        this.y = true;
        q(this.k);
        this.y = false;
        int childCount2 = getChildCount();
        for (int i13 = 0; i13 < childCount2; i13++) {
            View childAt2 = getChildAt(i13);
            if (childAt2.getVisibility() != 8 && ((fVar = (f) childAt2.getLayoutParams()) == null || !fVar.f519a)) {
                childAt2.measure(View.MeasureSpec.makeMeasureSpec((int) (paddingLeft * fVar.f521c), 1073741824), this.x);
            }
        }
    }

    @Override // android.view.ViewGroup
    public boolean onRequestFocusInDescendants(int i2, Rect rect) {
        int i3;
        int i4;
        e h2;
        int childCount = getChildCount();
        int i5 = -1;
        if ((i2 & 2) != 0) {
            i5 = childCount;
            i3 = 0;
            i4 = 1;
        } else {
            i3 = childCount - 1;
            i4 = -1;
        }
        while (i3 != i5) {
            View childAt = getChildAt(i3);
            if (childAt.getVisibility() == 0 && (h2 = h(childAt)) != null && h2.f515b == this.k && childAt.requestFocus(i2, rect)) {
                return true;
            }
            i3 += i4;
        }
        return false;
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof k)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        k kVar = (k) parcelable;
        super.onRestoreInstanceState(kVar.getSuperState());
        if (this.j != null) {
            v(kVar.f527b, false, true, 0);
            return;
        }
        this.l = kVar.f527b;
        this.m = kVar.f528c;
        this.n = kVar.f529d;
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        k kVar = new k(super.onSaveInstanceState());
        kVar.f527b = this.k;
        b.c0.a.a aVar = this.j;
        if (aVar != null) {
            Objects.requireNonNull(aVar);
            kVar.f528c = null;
        }
        return kVar;
    }

    @Override // android.view.View
    public void onSizeChanged(int i2, int i3, int i4, int i5) {
        super.onSizeChanged(i2, i3, i4, i5);
        if (i2 != i4) {
            int i6 = this.r;
            r(i2, i4, i6, i6);
        }
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        b.c0.a.a aVar;
        ArrayList<e> arrayList;
        boolean z = false;
        if ((motionEvent.getAction() == 0 && motionEvent.getEdgeFlags() != 0) || (aVar = this.j) == null || aVar.a() == 0) {
            return false;
        }
        if (this.M == null) {
            this.M = VelocityTracker.obtain();
        }
        this.M.addMovement(motionEvent);
        int action = motionEvent.getAction() & 255;
        if (action == 0) {
            this.o.abortAnimation();
            this.A = false;
            q(this.k);
            float x = motionEvent.getX();
            this.J = x;
            this.H = x;
            float y = motionEvent.getY();
            this.K = y;
            this.I = y;
            this.L = motionEvent.getPointerId(0);
        } else if (action != 1) {
            if (action == 2) {
                if (!this.C) {
                    int findPointerIndex = motionEvent.findPointerIndex(this.L);
                    if (findPointerIndex == -1) {
                        z = t();
                    } else {
                        float x2 = motionEvent.getX(findPointerIndex);
                        float abs = Math.abs(x2 - this.H);
                        float y2 = motionEvent.getY(findPointerIndex);
                        float abs2 = Math.abs(y2 - this.I);
                        if (abs > this.G && abs > abs2) {
                            this.C = true;
                            s(true);
                            float f2 = this.J;
                            this.H = x2 - f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? f2 + this.G : f2 - this.G;
                            this.I = y2;
                            setScrollState(1);
                            setScrollingCacheEnabled(true);
                            ViewParent parent = getParent();
                            if (parent != null) {
                                parent.requestDisallowInterceptTouchEvent(true);
                            }
                        }
                    }
                }
                if (this.C) {
                    z = false | p(motionEvent.getX(motionEvent.findPointerIndex(this.L)));
                }
            } else if (action != 3) {
                if (action == 5) {
                    int actionIndex = motionEvent.getActionIndex();
                    this.H = motionEvent.getX(actionIndex);
                    this.L = motionEvent.getPointerId(actionIndex);
                } else if (action == 6) {
                    l(motionEvent);
                    this.H = motionEvent.getX(motionEvent.findPointerIndex(this.L));
                }
            } else if (this.C) {
                u(this.k, true, 0, false);
                z = t();
            }
        } else if (this.C) {
            VelocityTracker velocityTracker = this.M;
            velocityTracker.computeCurrentVelocity(1000, this.O);
            int xVelocity = (int) velocityTracker.getXVelocity(this.L);
            this.A = true;
            int clientWidth = getClientWidth();
            int scrollX = getScrollX();
            e i2 = i();
            float f3 = clientWidth;
            int i3 = i2.f515b;
            float f4 = ((scrollX / f3) - i2.f518e) / (i2.f517d + (this.r / f3));
            if (Math.abs((int) (motionEvent.getX(motionEvent.findPointerIndex(this.L)) - this.J)) <= this.P || Math.abs(xVelocity) <= this.N) {
                i3 += (int) (f4 + (i3 >= this.k ? 0.4f : 0.6f));
            } else if (xVelocity <= 0) {
                i3++;
            }
            if (this.f511g.size() > 0) {
                i3 = Math.max(this.f511g.get(0).f515b, Math.min(i3, this.f511g.get(arrayList.size() - 1).f515b));
            }
            v(i3, true, true, xVelocity);
            z = t();
        }
        if (z) {
            AtomicInteger atomicInteger = q.f2214a;
            postInvalidateOnAnimation();
        }
        return true;
    }

    public final boolean p(float f2) {
        boolean z;
        boolean z2;
        float f3 = this.H - f2;
        this.H = f2;
        float scrollX = getScrollX() + f3;
        float clientWidth = getClientWidth();
        float f4 = this.v * clientWidth;
        float f5 = this.w * clientWidth;
        boolean z3 = false;
        e eVar = this.f511g.get(0);
        ArrayList<e> arrayList = this.f511g;
        e eVar2 = arrayList.get(arrayList.size() - 1);
        if (eVar.f515b != 0) {
            f4 = eVar.f518e * clientWidth;
            z = false;
        } else {
            z = true;
        }
        if (eVar2.f515b != this.j.a() - 1) {
            f5 = eVar2.f518e * clientWidth;
            z2 = false;
        } else {
            z2 = true;
        }
        if (scrollX < f4) {
            if (z) {
                this.R.onPull(Math.abs(f4 - scrollX) / clientWidth);
                z3 = true;
            }
            scrollX = f4;
        } else if (scrollX > f5) {
            if (z2) {
                this.S.onPull(Math.abs(scrollX - f5) / clientWidth);
                z3 = true;
            }
            scrollX = f5;
        }
        int i2 = (int) scrollX;
        this.H = (scrollX - i2) + this.H;
        scrollTo(i2, getScrollY());
        o(i2);
        return z3;
    }

    /* JADX WARN: Code restructure failed: missing block: B:21:0x0055, code lost:
        if (r5 == r6) goto L24;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void q(int i2) {
        e eVar;
        String hexString;
        e eVar2;
        e eVar3;
        e h2;
        int i3;
        int i4;
        e eVar4;
        e eVar5;
        int i5 = this.k;
        if (i5 != i2) {
            eVar = j(i5);
            this.k = i2;
        } else {
            eVar = null;
        }
        if (this.j == null || this.A || getWindowToken() == null) {
            return;
        }
        Objects.requireNonNull(this.j);
        int i6 = this.B;
        int i7 = 0;
        int max = Math.max(0, this.k - i6);
        int a2 = this.j.a();
        int min = Math.min(a2 - 1, this.k + i6);
        if (a2 == this.f510f) {
            while (true) {
                if (i7 >= this.f511g.size()) {
                    break;
                }
                eVar2 = this.f511g.get(i7);
                int i8 = eVar2.f515b;
                int i9 = this.k;
                if (i8 < i9) {
                    i7++;
                }
            }
            eVar2 = null;
            if (eVar2 == null && a2 > 0) {
                eVar2 = a(this.k, i7);
            }
            if (eVar2 != null) {
                int i10 = i7 - 1;
                e eVar6 = i10 >= 0 ? this.f511g.get(i10) : null;
                int clientWidth = getClientWidth();
                float paddingLeft = clientWidth <= 0 ? 0.0f : (getPaddingLeft() / clientWidth) + (2.0f - eVar2.f517d);
                float f2 = 0.0f;
                for (int i11 = this.k - 1; i11 >= 0; i11--) {
                    if (f2 < paddingLeft || i11 >= max) {
                        if (eVar6 != null && i11 == eVar6.f515b) {
                            f2 += eVar6.f517d;
                            i10--;
                            if (i10 >= 0) {
                                eVar6 = this.f511g.get(i10);
                            }
                            eVar6 = null;
                        } else {
                            f2 += a(i11, i10 + 1).f517d;
                            i7++;
                            if (i10 >= 0) {
                                eVar6 = this.f511g.get(i10);
                            }
                            eVar6 = null;
                        }
                    } else if (eVar6 == null) {
                        break;
                    } else if (i11 == eVar6.f515b && !eVar6.f516c) {
                        this.f511g.remove(i10);
                        b.c0.a.a aVar = this.j;
                        Object obj = eVar6.f514a;
                        Objects.requireNonNull((c.e.b.ef.g) aVar);
                        removeView((RelativeLayout) obj);
                        i10--;
                        i7--;
                        if (i10 >= 0) {
                            eVar6 = this.f511g.get(i10);
                        }
                        eVar6 = null;
                    }
                }
                float f3 = eVar2.f517d;
                int i12 = i7 + 1;
                if (f3 < 2.0f) {
                    e eVar7 = i12 < this.f511g.size() ? this.f511g.get(i12) : null;
                    float paddingRight = clientWidth <= 0 ? 0.0f : (getPaddingRight() / clientWidth) + 2.0f;
                    int i13 = i12;
                    for (int i14 = this.k + 1; i14 < a2; i14++) {
                        if (f3 < paddingRight || i14 <= min) {
                            if (eVar7 != null && i14 == eVar7.f515b) {
                                f3 += eVar7.f517d;
                                i13++;
                                if (i13 < this.f511g.size()) {
                                    eVar7 = this.f511g.get(i13);
                                }
                                eVar7 = null;
                            } else {
                                e a3 = a(i14, i13);
                                i13++;
                                f3 += a3.f517d;
                                if (i13 < this.f511g.size()) {
                                    eVar7 = this.f511g.get(i13);
                                }
                                eVar7 = null;
                            }
                        } else if (eVar7 == null) {
                            break;
                        } else if (i14 == eVar7.f515b && !eVar7.f516c) {
                            this.f511g.remove(i13);
                            b.c0.a.a aVar2 = this.j;
                            Object obj2 = eVar7.f514a;
                            Objects.requireNonNull((c.e.b.ef.g) aVar2);
                            removeView((RelativeLayout) obj2);
                            if (i13 < this.f511g.size()) {
                                eVar7 = this.f511g.get(i13);
                            }
                            eVar7 = null;
                        }
                    }
                }
                int a4 = this.j.a();
                int clientWidth2 = getClientWidth();
                float f4 = clientWidth2 > 0 ? this.r / clientWidth2 : 0.0f;
                if (eVar != null) {
                    int i15 = eVar.f515b;
                    int i16 = eVar2.f515b;
                    if (i15 < i16) {
                        float f5 = eVar.f518e + eVar.f517d + f4;
                        int i17 = 0;
                        while (true) {
                            i15++;
                            if (i15 > eVar2.f515b || i17 >= this.f511g.size()) {
                                break;
                            }
                            e eVar8 = this.f511g.get(i17);
                            while (true) {
                                eVar5 = eVar8;
                                if (i15 <= eVar5.f515b || i17 >= this.f511g.size() - 1) {
                                    break;
                                }
                                i17++;
                                eVar8 = this.f511g.get(i17);
                            }
                            while (i15 < eVar5.f515b) {
                                Objects.requireNonNull(this.j);
                                f5 += 1.0f + f4;
                                i15++;
                            }
                            eVar5.f518e = f5;
                            f5 += eVar5.f517d + f4;
                        }
                    } else if (i15 > i16) {
                        int size = this.f511g.size() - 1;
                        float f6 = eVar.f518e;
                        while (true) {
                            i15--;
                            if (i15 < eVar2.f515b || size < 0) {
                                break;
                            }
                            e eVar9 = this.f511g.get(size);
                            while (true) {
                                eVar4 = eVar9;
                                if (i15 >= eVar4.f515b || size <= 0) {
                                    break;
                                }
                                size--;
                                eVar9 = this.f511g.get(size);
                            }
                            while (i15 > eVar4.f515b) {
                                Objects.requireNonNull(this.j);
                                f6 -= 1.0f + f4;
                                i15--;
                            }
                            f6 -= eVar4.f517d + f4;
                            eVar4.f518e = f6;
                        }
                    }
                }
                int size2 = this.f511g.size();
                float f7 = eVar2.f518e;
                int i18 = eVar2.f515b;
                int i19 = i18 - 1;
                this.v = i18 == 0 ? f7 : -3.4028235E38f;
                int i20 = a4 - 1;
                this.w = i18 == i20 ? (eVar2.f517d + f7) - 1.0f : Float.MAX_VALUE;
                int i21 = i7 - 1;
                while (i21 >= 0) {
                    e eVar10 = this.f511g.get(i21);
                    while (true) {
                        i4 = eVar10.f515b;
                        if (i19 <= i4) {
                            break;
                        }
                        i19--;
                        Objects.requireNonNull(this.j);
                        f7 -= 1.0f + f4;
                    }
                    f7 -= eVar10.f517d + f4;
                    eVar10.f518e = f7;
                    if (i4 == 0) {
                        this.v = f7;
                    }
                    i21--;
                    i19--;
                }
                float f8 = eVar2.f518e + eVar2.f517d + f4;
                int i22 = eVar2.f515b;
                while (true) {
                    i22++;
                    if (i12 >= size2) {
                        break;
                    }
                    e eVar11 = this.f511g.get(i12);
                    while (true) {
                        i3 = eVar11.f515b;
                        if (i22 >= i3) {
                            break;
                        }
                        i22++;
                        Objects.requireNonNull(this.j);
                        f8 += 1.0f + f4;
                    }
                    if (i3 == i20) {
                        this.w = (eVar11.f517d + f8) - 1.0f;
                    }
                    eVar11.f518e = f8;
                    f8 += eVar11.f517d + f4;
                    i12++;
                }
                Objects.requireNonNull(this.j);
            }
            Objects.requireNonNull(this.j);
            int childCount = getChildCount();
            for (int i23 = 0; i23 < childCount; i23++) {
                View childAt = getChildAt(i23);
                f fVar = (f) childAt.getLayoutParams();
                fVar.f524f = i23;
                if (!fVar.f519a && fVar.f521c == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && (h2 = h(childAt)) != null) {
                    fVar.f521c = h2.f517d;
                    fVar.f523e = h2.f515b;
                }
            }
            if (hasFocus()) {
                View findFocus = findFocus();
                if (findFocus != null) {
                    while (true) {
                        ViewParent parent = findFocus.getParent();
                        if (parent != this) {
                            if (parent == null || !(parent instanceof View)) {
                                break;
                            }
                            findFocus = (View) parent;
                        } else {
                            eVar3 = h(findFocus);
                            break;
                        }
                    }
                }
                eVar3 = null;
                if (eVar3 == null || eVar3.f515b != this.k) {
                    for (int i24 = 0; i24 < getChildCount(); i24++) {
                        View childAt2 = getChildAt(i24);
                        e h3 = h(childAt2);
                        if (h3 != null && h3.f515b == this.k && childAt2.requestFocus(2)) {
                            return;
                        }
                    }
                    return;
                }
                return;
            }
            return;
        }
        try {
            hexString = getResources().getResourceName(getId());
        } catch (Resources.NotFoundException unused) {
            hexString = Integer.toHexString(getId());
        }
        StringBuilder x = c.b.a.a.a.x("The application's PagerAdapter changed the adapter's contents without calling PagerAdapter#notifyDataSetChanged! Expected adapter item count: ");
        x.append(this.f510f);
        x.append(", found: ");
        x.append(a2);
        x.append(" Pager id: ");
        x.append(hexString);
        x.append(" Pager class: ");
        x.append(getClass());
        x.append(" Problematic adapter: ");
        x.append(this.j.getClass());
        throw new IllegalStateException(x.toString());
    }

    public final void r(int i2, int i3, int i4, int i5) {
        if (i3 > 0 && !this.f511g.isEmpty()) {
            if (!this.o.isFinished()) {
                this.o.setFinalX(getCurrentItem() * getClientWidth());
                return;
            } else {
                scrollTo((int) ((getScrollX() / (((i3 - getPaddingLeft()) - getPaddingRight()) + i5)) * (((i2 - getPaddingLeft()) - getPaddingRight()) + i4)), getScrollY());
                return;
            }
        }
        e j2 = j(this.k);
        int min = (int) ((j2 != null ? Math.min(j2.f518e, this.w) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) * ((i2 - getPaddingLeft()) - getPaddingRight()));
        if (min != getScrollX()) {
            d(false);
            scrollTo(min, getScrollY());
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewManager
    public void removeView(View view) {
        if (this.y) {
            removeViewInLayout(view);
        } else {
            super.removeView(view);
        }
    }

    public final void s(boolean z) {
        ViewParent parent = getParent();
        if (parent != null) {
            parent.requestDisallowInterceptTouchEvent(z);
        }
    }

    public void setAdapter(b.c0.a.a aVar) {
        b.c0.a.a aVar2 = this.j;
        if (aVar2 != null) {
            synchronized (aVar2) {
            }
            Objects.requireNonNull(this.j);
            for (int i2 = 0; i2 < this.f511g.size(); i2++) {
                e eVar = this.f511g.get(i2);
                b.c0.a.a aVar3 = this.j;
                int i3 = eVar.f515b;
                Object obj = eVar.f514a;
                Objects.requireNonNull((c.e.b.ef.g) aVar3);
                removeView((RelativeLayout) obj);
            }
            Objects.requireNonNull(this.j);
            this.f511g.clear();
            int i4 = 0;
            while (i4 < getChildCount()) {
                if (!((f) getChildAt(i4).getLayoutParams()).f519a) {
                    removeViewAt(i4);
                    i4--;
                }
                i4++;
            }
            this.k = 0;
            scrollTo(0, 0);
        }
        b.c0.a.a aVar4 = this.j;
        this.j = aVar;
        this.f510f = 0;
        if (aVar != null) {
            if (this.q == null) {
                this.q = new j();
            }
            synchronized (this.j) {
            }
            this.A = false;
            boolean z = this.T;
            this.T = true;
            this.f510f = this.j.a();
            if (this.l >= 0) {
                Objects.requireNonNull(this.j);
                v(this.l, false, true, 0);
                this.l = -1;
                this.m = null;
                this.n = null;
            } else if (!z) {
                q(this.k);
            } else {
                requestLayout();
            }
        }
        List<h> list = this.b0;
        if (list == null || list.isEmpty()) {
            return;
        }
        int size = this.b0.size();
        for (int i5 = 0; i5 < size; i5++) {
            this.b0.get(i5).onAdapterChanged(this, aVar4, aVar);
        }
    }

    public void setCurrentItem(int i2) {
        this.A = false;
        v(i2, !this.T, false, 0);
    }

    public void setOffscreenPageLimit(int i2) {
        if (i2 < 1) {
            Log.w("ViewPager", "Requested offscreen page limit " + i2 + " too small; defaulting to 1");
            i2 = 1;
        }
        if (i2 != this.B) {
            this.B = i2;
            q(this.k);
        }
    }

    @Deprecated
    public void setOnPageChangeListener(i iVar) {
        this.a0 = iVar;
    }

    public void setPageMargin(int i2) {
        int i3 = this.r;
        this.r = i2;
        int width = getWidth();
        r(width, width, i2, i3);
        requestLayout();
    }

    public void setPageMarginDrawable(Drawable drawable) {
        this.s = drawable;
        if (drawable != null) {
            refreshDrawableState();
        }
        setWillNotDraw(drawable == null);
        invalidate();
    }

    public void setScrollState(int i2) {
        if (this.d0 == i2) {
            return;
        }
        this.d0 = i2;
        i iVar = this.a0;
        if (iVar != null) {
            iVar.onPageScrollStateChanged(i2);
        }
        List<i> list = this.W;
        if (list != null) {
            int size = list.size();
            for (int i3 = 0; i3 < size; i3++) {
                i iVar2 = this.W.get(i3);
                if (iVar2 != null) {
                    iVar2.onPageScrollStateChanged(i2);
                }
            }
        }
    }

    public final boolean t() {
        this.L = -1;
        this.C = false;
        this.D = false;
        VelocityTracker velocityTracker = this.M;
        if (velocityTracker != null) {
            velocityTracker.recycle();
            this.M = null;
        }
        this.R.onRelease();
        this.S.onRelease();
        return this.R.isFinished() || this.S.isFinished();
    }

    public final void u(int i2, boolean z, int i3, boolean z2) {
        int scrollX;
        int abs;
        e j2 = j(i2);
        int max = j2 != null ? (int) (Math.max(this.v, Math.min(j2.f518e, this.w)) * getClientWidth()) : 0;
        if (z) {
            if (getChildCount() == 0) {
                setScrollingCacheEnabled(false);
            } else {
                Scroller scroller = this.o;
                if ((scroller == null || scroller.isFinished()) ? false : true) {
                    scrollX = this.p ? this.o.getCurrX() : this.o.getStartX();
                    this.o.abortAnimation();
                    setScrollingCacheEnabled(false);
                } else {
                    scrollX = getScrollX();
                }
                int i4 = scrollX;
                int scrollY = getScrollY();
                int i5 = max - i4;
                int i6 = 0 - scrollY;
                if (i5 == 0 && i6 == 0) {
                    d(false);
                    q(this.k);
                    setScrollState(0);
                } else {
                    setScrollingCacheEnabled(true);
                    setScrollState(2);
                    int clientWidth = getClientWidth();
                    int i7 = clientWidth / 2;
                    float f2 = clientWidth;
                    float f3 = i7;
                    float sin = (((float) Math.sin((Math.min(1.0f, (Math.abs(i5) * 1.0f) / f2) - 0.5f) * 0.47123894f)) * f3) + f3;
                    int abs2 = Math.abs(i3);
                    if (abs2 > 0) {
                        abs = Math.round(Math.abs(sin / abs2) * 1000.0f) * 4;
                    } else {
                        Objects.requireNonNull(this.j);
                        abs = (int) (((Math.abs(i5) / ((f2 * 1.0f) + this.r)) + 1.0f) * 100.0f);
                    }
                    int min = Math.min(abs, 600);
                    this.p = false;
                    this.o.startScroll(i4, scrollY, i5, i6, min);
                    AtomicInteger atomicInteger = q.f2214a;
                    postInvalidateOnAnimation();
                }
            }
            if (z2) {
                f(i2);
                return;
            }
            return;
        }
        if (z2) {
            f(i2);
        }
        d(false);
        scrollTo(max, 0);
        o(max);
    }

    public void v(int i2, boolean z, boolean z2, int i3) {
        b.c0.a.a aVar = this.j;
        if (aVar != null && aVar.a() > 0) {
            if (!z2 && this.k == i2 && this.f511g.size() != 0) {
                setScrollingCacheEnabled(false);
                return;
            }
            if (i2 < 0) {
                i2 = 0;
            } else if (i2 >= this.j.a()) {
                i2 = this.j.a() - 1;
            }
            int i4 = this.B;
            int i5 = this.k;
            if (i2 > i5 + i4 || i2 < i5 - i4) {
                for (int i6 = 0; i6 < this.f511g.size(); i6++) {
                    this.f511g.get(i6).f516c = true;
                }
            }
            boolean z3 = this.k != i2;
            if (this.T) {
                this.k = i2;
                if (z3) {
                    f(i2);
                }
                requestLayout();
                return;
            }
            q(i2);
            u(i2, z, i3, z3);
            return;
        }
        setScrollingCacheEnabled(false);
    }

    @Override // android.view.View
    public boolean verifyDrawable(Drawable drawable) {
        return super.verifyDrawable(drawable) || drawable == this.s;
    }

    /* loaded from: classes.dex */
    public static class f extends ViewGroup.LayoutParams {

        /* renamed from: a  reason: collision with root package name */
        public boolean f519a;

        /* renamed from: b  reason: collision with root package name */
        public int f520b;

        /* renamed from: c  reason: collision with root package name */
        public float f521c;

        /* renamed from: d  reason: collision with root package name */
        public boolean f522d;

        /* renamed from: e  reason: collision with root package name */
        public int f523e;

        /* renamed from: f  reason: collision with root package name */
        public int f524f;

        public f() {
            super(-1, -1);
            this.f521c = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public f(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f521c = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, ViewPager.f506b);
            this.f520b = obtainStyledAttributes.getInteger(0, 48);
            obtainStyledAttributes.recycle();
        }
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(AttributeSet attributeSet) {
        return new f(getContext(), attributeSet);
    }

    public void setPageMarginDrawable(int i2) {
        Context context = getContext();
        Object obj = b.j.c.a.f2074a;
        setPageMarginDrawable(context.getDrawable(i2));
    }
}