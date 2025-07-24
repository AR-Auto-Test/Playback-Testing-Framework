package androidx.drawerlayout.widget;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Parcel;
import android.os.Parcelable;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.WindowInsets;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import b.j.j.q;
import b.j.j.w;
import b.j.j.x.b;
import b.j.j.x.d;
import b.l.b.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class DrawerLayout extends ViewGroup {

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f271b = {16843828};

    /* renamed from: c  reason: collision with root package name */
    public static final int[] f272c = {16842931};

    /* renamed from: d  reason: collision with root package name */
    public static final boolean f273d;

    /* renamed from: e  reason: collision with root package name */
    public static final boolean f274e;

    /* renamed from: f  reason: collision with root package name */
    public static boolean f275f;
    public float A;
    public float B;
    public Drawable C;
    public Object D;
    public boolean E;
    public final ArrayList<View> F;
    public Rect G;
    public Matrix H;
    public final b.j.j.x.d I;

    /* renamed from: g  reason: collision with root package name */
    public final c f276g;

    /* renamed from: h  reason: collision with root package name */
    public float f277h;
    public int i;
    public int j;
    public float k;
    public Paint l;
    public final b.l.b.e m;
    public final b.l.b.e n;
    public final g o;
    public final g p;
    public int q;
    public boolean r;
    public boolean s;
    public int t;
    public int u;
    public int v;
    public int w;
    public boolean x;
    public d y;
    public List<d> z;

    /* loaded from: classes.dex */
    public class a implements b.j.j.x.d {
        public a() {
        }

        @Override // b.j.j.x.d
        public boolean perform(View view, d.a aVar) {
            if (!DrawerLayout.this.l(view) || DrawerLayout.this.g(view) == 2) {
                return false;
            }
            DrawerLayout.this.b(view);
            return true;
        }
    }

    /* loaded from: classes.dex */
    public class b extends b.j.j.a {

        /* renamed from: a  reason: collision with root package name */
        public final Rect f279a = new Rect();

        public b() {
        }

        @Override // b.j.j.a
        public boolean dispatchPopulateAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            if (accessibilityEvent.getEventType() == 32) {
                accessibilityEvent.getText();
                View f2 = DrawerLayout.this.f();
                if (f2 != null) {
                    int h2 = DrawerLayout.this.h(f2);
                    DrawerLayout drawerLayout = DrawerLayout.this;
                    Objects.requireNonNull(drawerLayout);
                    AtomicInteger atomicInteger = q.f2214a;
                    Gravity.getAbsoluteGravity(h2, drawerLayout.getLayoutDirection());
                    return true;
                }
                return true;
            }
            return super.dispatchPopulateAccessibilityEvent(view, accessibilityEvent);
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            super.onInitializeAccessibilityEvent(view, accessibilityEvent);
            accessibilityEvent.setClassName("androidx.drawerlayout.widget.DrawerLayout");
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
            if (DrawerLayout.f273d) {
                super.onInitializeAccessibilityNodeInfo(view, bVar);
            } else {
                AccessibilityNodeInfo obtain = AccessibilityNodeInfo.obtain(bVar.f2259b);
                b.j.j.x.b bVar2 = new b.j.j.x.b(obtain);
                super.onInitializeAccessibilityNodeInfo(view, bVar2);
                bVar.f2261d = -1;
                bVar.f2259b.setSource(view);
                AtomicInteger atomicInteger = q.f2214a;
                ViewParent parentForAccessibility = view.getParentForAccessibility();
                if (parentForAccessibility instanceof View) {
                    bVar.p((View) parentForAccessibility);
                }
                Rect rect = this.f279a;
                obtain.getBoundsInScreen(rect);
                bVar.f2259b.setBoundsInScreen(rect);
                bVar.f2259b.setVisibleToUser(obtain.isVisibleToUser());
                bVar.f2259b.setPackageName(obtain.getPackageName());
                bVar.f2259b.setClassName(bVar2.e());
                bVar.f2259b.setContentDescription(bVar2.g());
                bVar.f2259b.setEnabled(bVar2.j());
                bVar.f2259b.setFocused(obtain.isFocused());
                bVar.f2259b.setAccessibilityFocused(obtain.isAccessibilityFocused());
                bVar.f2259b.setSelected(obtain.isSelected());
                bVar.f2259b.addAction(bVar2.d());
                obtain.recycle();
                ViewGroup viewGroup = (ViewGroup) view;
                int childCount = viewGroup.getChildCount();
                for (int i = 0; i < childCount; i++) {
                    View childAt = viewGroup.getChildAt(i);
                    if (DrawerLayout.j(childAt)) {
                        bVar.f2259b.addChild(childAt);
                    }
                }
            }
            bVar.f2259b.setClassName("androidx.drawerlayout.widget.DrawerLayout");
            bVar.f2259b.setFocusable(false);
            bVar.f2259b.setFocused(false);
            bVar.k(b.a.f2262a);
            bVar.k(b.a.f2263b);
        }

        @Override // b.j.j.a
        public boolean onRequestSendAccessibilityEvent(ViewGroup viewGroup, View view, AccessibilityEvent accessibilityEvent) {
            if (DrawerLayout.f273d || DrawerLayout.j(view)) {
                return super.onRequestSendAccessibilityEvent(viewGroup, view, accessibilityEvent);
            }
            return false;
        }
    }

    /* loaded from: classes.dex */
    public static final class c extends b.j.j.a {
        @Override // b.j.j.a
        public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
            super.onInitializeAccessibilityNodeInfo(view, bVar);
            if (DrawerLayout.j(view)) {
                return;
            }
            bVar.p(null);
        }
    }

    /* loaded from: classes.dex */
    public interface d {
        void a(View view);

        void b(View view);

        void c(int i);

        void d(View view, float f2);
    }

    /* loaded from: classes.dex */
    public class g extends e.c {

        /* renamed from: a  reason: collision with root package name */
        public final int f290a;

        /* renamed from: b  reason: collision with root package name */
        public b.l.b.e f291b;

        /* renamed from: c  reason: collision with root package name */
        public final Runnable f292c = new a();

        /* loaded from: classes.dex */
        public class a implements Runnable {
            public a() {
            }

            @Override // java.lang.Runnable
            public void run() {
                View d2;
                int width;
                g gVar = g.this;
                int i = gVar.f291b.p;
                boolean z = gVar.f290a == 3;
                if (z) {
                    d2 = DrawerLayout.this.d(3);
                    width = (d2 != null ? -d2.getWidth() : 0) + i;
                } else {
                    d2 = DrawerLayout.this.d(5);
                    width = DrawerLayout.this.getWidth() - i;
                }
                if (d2 != null) {
                    if (((!z || d2.getLeft() >= width) && (z || d2.getLeft() <= width)) || DrawerLayout.this.g(d2) != 0) {
                        return;
                    }
                    gVar.f291b.x(d2, width, d2.getTop());
                    ((e) d2.getLayoutParams()).f283c = true;
                    DrawerLayout.this.invalidate();
                    gVar.a();
                    DrawerLayout drawerLayout = DrawerLayout.this;
                    if (drawerLayout.x) {
                        return;
                    }
                    long uptimeMillis = SystemClock.uptimeMillis();
                    MotionEvent obtain = MotionEvent.obtain(uptimeMillis, uptimeMillis, 3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0);
                    int childCount = drawerLayout.getChildCount();
                    for (int i2 = 0; i2 < childCount; i2++) {
                        drawerLayout.getChildAt(i2).dispatchTouchEvent(obtain);
                    }
                    obtain.recycle();
                    drawerLayout.x = true;
                }
            }
        }

        public g(int i) {
            this.f290a = i;
        }

        public final void a() {
            View d2 = DrawerLayout.this.d(this.f290a == 3 ? 5 : 3);
            if (d2 != null) {
                DrawerLayout.this.b(d2);
            }
        }

        public void b() {
            DrawerLayout.this.removeCallbacks(this.f292c);
        }

        @Override // b.l.b.e.c
        public int clampViewPositionHorizontal(View view, int i, int i2) {
            if (DrawerLayout.this.a(view, 3)) {
                return Math.max(-view.getWidth(), Math.min(i, 0));
            }
            int width = DrawerLayout.this.getWidth();
            return Math.max(width - view.getWidth(), Math.min(i, width));
        }

        @Override // b.l.b.e.c
        public int clampViewPositionVertical(View view, int i, int i2) {
            return view.getTop();
        }

        @Override // b.l.b.e.c
        public int getViewHorizontalDragRange(View view) {
            if (DrawerLayout.this.m(view)) {
                return view.getWidth();
            }
            return 0;
        }

        @Override // b.l.b.e.c
        public void onEdgeDragStarted(int i, int i2) {
            View d2;
            if ((i & 1) == 1) {
                d2 = DrawerLayout.this.d(3);
            } else {
                d2 = DrawerLayout.this.d(5);
            }
            if (d2 == null || DrawerLayout.this.g(d2) != 0) {
                return;
            }
            this.f291b.b(d2, i2);
        }

        @Override // b.l.b.e.c
        public boolean onEdgeLock(int i) {
            return false;
        }

        @Override // b.l.b.e.c
        public void onEdgeTouched(int i, int i2) {
            DrawerLayout.this.postDelayed(this.f292c, 160L);
        }

        @Override // b.l.b.e.c
        public void onViewCaptured(View view, int i) {
            ((e) view.getLayoutParams()).f283c = false;
            a();
        }

        @Override // b.l.b.e.c
        public void onViewDragStateChanged(int i) {
            DrawerLayout.this.t(i, this.f291b.u);
        }

        @Override // b.l.b.e.c
        public void onViewPositionChanged(View view, int i, int i2, int i3, int i4) {
            int width = view.getWidth();
            float width2 = (DrawerLayout.this.a(view, 3) ? i + width : DrawerLayout.this.getWidth() - i) / width;
            DrawerLayout.this.q(view, width2);
            view.setVisibility(width2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 4 : 0);
            DrawerLayout.this.invalidate();
        }

        @Override // b.l.b.e.c
        public void onViewReleased(View view, float f2, float f3) {
            int i;
            Objects.requireNonNull(DrawerLayout.this);
            float f4 = ((e) view.getLayoutParams()).f282b;
            int width = view.getWidth();
            if (DrawerLayout.this.a(view, 3)) {
                int i2 = (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 1 : (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : -1));
                i = (i2 > 0 || (i2 == 0 && f4 > 0.5f)) ? 0 : -width;
            } else {
                int width2 = DrawerLayout.this.getWidth();
                if (f2 < StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD || (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && f4 > 0.5f)) {
                    width2 -= width;
                }
                i = width2;
            }
            this.f291b.v(i, view.getTop());
            DrawerLayout.this.invalidate();
        }

        @Override // b.l.b.e.c
        public boolean tryCaptureView(View view, int i) {
            return DrawerLayout.this.m(view) && DrawerLayout.this.a(view, this.f290a) && DrawerLayout.this.g(view) == 0;
        }
    }

    static {
        int i = Build.VERSION.SDK_INT;
        f273d = true;
        f274e = true;
        f275f = i >= 29;
    }

    public DrawerLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, R.attr.drawerLayoutStyle);
        this.f276g = new c();
        this.j = -1728053248;
        this.l = new Paint();
        this.s = true;
        this.t = 3;
        this.u = 3;
        this.v = 3;
        this.w = 3;
        this.I = new a();
        setDescendantFocusability(Calib3d.CALIB_TILTED_MODEL);
        float f2 = getResources().getDisplayMetrics().density;
        this.i = (int) ((64.0f * f2) + 0.5f);
        float f3 = f2 * 400.0f;
        g gVar = new g(3);
        this.o = gVar;
        g gVar2 = new g(5);
        this.p = gVar2;
        b.l.b.e j = b.l.b.e.j(this, 1.0f, gVar);
        this.m = j;
        j.r = 1;
        j.o = f3;
        gVar.f291b = j;
        b.l.b.e j2 = b.l.b.e.j(this, 1.0f, gVar2);
        this.n = j2;
        j2.r = 2;
        j2.o = f3;
        gVar2.f291b = j2;
        setFocusableInTouchMode(true);
        AtomicInteger atomicInteger = q.f2214a;
        setImportantForAccessibility(1);
        q.n(this, new b());
        setMotionEventSplittingEnabled(false);
        if (getFitsSystemWindows()) {
            setOnApplyWindowInsetsListener(new b.n.b.a(this));
            setSystemUiVisibility(1280);
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(f271b);
            try {
                this.C = obtainStyledAttributes.getDrawable(0);
            } finally {
                obtainStyledAttributes.recycle();
            }
        }
        TypedArray obtainStyledAttributes2 = context.obtainStyledAttributes(attributeSet, b.n.a.f2338a, R.attr.drawerLayoutStyle, 0);
        try {
            if (obtainStyledAttributes2.hasValue(0)) {
                this.f277h = obtainStyledAttributes2.getDimension(0, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            } else {
                this.f277h = getResources().getDimension(R.dimen.def_drawer_elevation);
            }
            obtainStyledAttributes2.recycle();
            this.F = new ArrayList<>();
        } catch (Throwable th) {
            obtainStyledAttributes2.recycle();
            throw th;
        }
    }

    public static String i(int i) {
        return (i & 3) == 3 ? "LEFT" : (i & 5) == 5 ? "RIGHT" : Integer.toHexString(i);
    }

    public static boolean j(View view) {
        AtomicInteger atomicInteger = q.f2214a;
        return (view.getImportantForAccessibility() == 4 || view.getImportantForAccessibility() == 2) ? false : true;
    }

    public boolean a(View view, int i) {
        return (h(view) & i) == i;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void addFocusables(ArrayList<View> arrayList, int i, int i2) {
        if (getDescendantFocusability() == 393216) {
            return;
        }
        int childCount = getChildCount();
        boolean z = false;
        for (int i3 = 0; i3 < childCount; i3++) {
            View childAt = getChildAt(i3);
            if (m(childAt)) {
                if (l(childAt)) {
                    childAt.addFocusables(arrayList, i, i2);
                    z = true;
                }
            } else {
                this.F.add(childAt);
            }
        }
        if (!z) {
            int size = this.F.size();
            for (int i4 = 0; i4 < size; i4++) {
                View view = this.F.get(i4);
                if (view.getVisibility() == 0) {
                    view.addFocusables(arrayList, i, i2);
                }
            }
        }
        this.F.clear();
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i, ViewGroup.LayoutParams layoutParams) {
        super.addView(view, i, layoutParams);
        if (e() == null && !m(view)) {
            AtomicInteger atomicInteger = q.f2214a;
            view.setImportantForAccessibility(1);
        } else {
            AtomicInteger atomicInteger2 = q.f2214a;
            view.setImportantForAccessibility(4);
        }
        if (f273d) {
            return;
        }
        q.n(view, this.f276g);
    }

    public void b(View view) {
        if (m(view)) {
            e eVar = (e) view.getLayoutParams();
            if (this.s) {
                eVar.f282b = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                eVar.f284d = 0;
            } else {
                eVar.f284d |= 4;
                if (a(view, 3)) {
                    this.m.x(view, -view.getWidth(), view.getTop());
                } else {
                    this.n.x(view, getWidth(), view.getTop());
                }
            }
            invalidate();
            return;
        }
        throw new IllegalArgumentException("View " + view + " is not a sliding drawer");
    }

    public void c(boolean z) {
        boolean x;
        int childCount = getChildCount();
        boolean z2 = false;
        for (int i = 0; i < childCount; i++) {
            View childAt = getChildAt(i);
            e eVar = (e) childAt.getLayoutParams();
            if (m(childAt) && (!z || eVar.f283c)) {
                int width = childAt.getWidth();
                if (a(childAt, 3)) {
                    x = this.m.x(childAt, -width, childAt.getTop());
                } else {
                    x = this.n.x(childAt, getWidth(), childAt.getTop());
                }
                z2 |= x;
                eVar.f283c = false;
            }
        }
        this.o.b();
        this.p.b();
        if (z2) {
            invalidate();
        }
    }

    @Override // android.view.ViewGroup
    public boolean checkLayoutParams(ViewGroup.LayoutParams layoutParams) {
        return (layoutParams instanceof e) && super.checkLayoutParams(layoutParams);
    }

    @Override // android.view.View
    public void computeScroll() {
        int childCount = getChildCount();
        float f2 = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        for (int i = 0; i < childCount; i++) {
            f2 = Math.max(f2, ((e) getChildAt(i).getLayoutParams()).f282b);
        }
        this.k = f2;
        boolean i2 = this.m.i(true);
        boolean i3 = this.n.i(true);
        if (i2 || i3) {
            AtomicInteger atomicInteger = q.f2214a;
            postInvalidateOnAnimation();
        }
    }

    public View d(int i) {
        AtomicInteger atomicInteger = q.f2214a;
        int absoluteGravity = Gravity.getAbsoluteGravity(i, getLayoutDirection()) & 7;
        int childCount = getChildCount();
        for (int i2 = 0; i2 < childCount; i2++) {
            View childAt = getChildAt(i2);
            if ((h(childAt) & 7) == absoluteGravity) {
                return childAt;
            }
        }
        return null;
    }

    @Override // android.view.View
    public boolean dispatchGenericMotionEvent(MotionEvent motionEvent) {
        boolean dispatchGenericMotionEvent;
        if ((motionEvent.getSource() & 2) != 0 && motionEvent.getAction() != 10 && this.k > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
            int childCount = getChildCount();
            if (childCount != 0) {
                float x = motionEvent.getX();
                float y = motionEvent.getY();
                for (int i = childCount - 1; i >= 0; i--) {
                    View childAt = getChildAt(i);
                    if (this.G == null) {
                        this.G = new Rect();
                    }
                    childAt.getHitRect(this.G);
                    if (this.G.contains((int) x, (int) y) && !k(childAt)) {
                        if (!childAt.getMatrix().isIdentity()) {
                            MotionEvent obtain = MotionEvent.obtain(motionEvent);
                            obtain.offsetLocation(getScrollX() - childAt.getLeft(), getScrollY() - childAt.getTop());
                            Matrix matrix = childAt.getMatrix();
                            if (!matrix.isIdentity()) {
                                if (this.H == null) {
                                    this.H = new Matrix();
                                }
                                matrix.invert(this.H);
                                obtain.transform(this.H);
                            }
                            dispatchGenericMotionEvent = childAt.dispatchGenericMotionEvent(obtain);
                            obtain.recycle();
                        } else {
                            float scrollX = getScrollX() - childAt.getLeft();
                            float scrollY = getScrollY() - childAt.getTop();
                            motionEvent.offsetLocation(scrollX, scrollY);
                            dispatchGenericMotionEvent = childAt.dispatchGenericMotionEvent(motionEvent);
                            motionEvent.offsetLocation(-scrollX, -scrollY);
                        }
                        if (dispatchGenericMotionEvent) {
                            return true;
                        }
                    }
                }
                return false;
            }
            return false;
        }
        return super.dispatchGenericMotionEvent(motionEvent);
    }

    @Override // android.view.ViewGroup
    public boolean drawChild(Canvas canvas, View view, long j) {
        int i;
        int height = getHeight();
        boolean k = k(view);
        int width = getWidth();
        int save = canvas.save();
        int i2 = 0;
        if (k) {
            int childCount = getChildCount();
            int i3 = 0;
            for (int i4 = 0; i4 < childCount; i4++) {
                View childAt = getChildAt(i4);
                if (childAt != view && childAt.getVisibility() == 0) {
                    Drawable background = childAt.getBackground();
                    if ((background != null && background.getOpacity() == -1) && m(childAt) && childAt.getHeight() >= height) {
                        if (a(childAt, 3)) {
                            int right = childAt.getRight();
                            if (right > i3) {
                                i3 = right;
                            }
                        } else {
                            int left = childAt.getLeft();
                            if (left < width) {
                                width = left;
                            }
                        }
                    }
                }
            }
            canvas.clipRect(i3, 0, width, getHeight());
            i2 = i3;
        }
        boolean drawChild = super.drawChild(canvas, view, j);
        canvas.restoreToCount(save);
        float f2 = this.k;
        if (f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && k) {
            this.l.setColor((((int) ((((-16777216) & i) >>> 24) * f2)) << 24) | (this.j & 16777215));
            canvas.drawRect(i2, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, width, getHeight(), this.l);
        }
        return drawChild;
    }

    public View e() {
        int childCount = getChildCount();
        for (int i = 0; i < childCount; i++) {
            View childAt = getChildAt(i);
            if ((((e) childAt.getLayoutParams()).f284d & 1) == 1) {
                return childAt;
            }
        }
        return null;
    }

    public View f() {
        int childCount = getChildCount();
        for (int i = 0; i < childCount; i++) {
            View childAt = getChildAt(i);
            if (m(childAt)) {
                if (m(childAt)) {
                    if (((e) childAt.getLayoutParams()).f282b > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                        return childAt;
                    }
                } else {
                    throw new IllegalArgumentException("View " + childAt + " is not a drawer");
                }
            }
        }
        return null;
    }

    public int g(View view) {
        if (m(view)) {
            int i = ((e) view.getLayoutParams()).f281a;
            AtomicInteger atomicInteger = q.f2214a;
            int layoutDirection = getLayoutDirection();
            if (i == 3) {
                int i2 = this.t;
                if (i2 != 3) {
                    return i2;
                }
                int i3 = layoutDirection == 0 ? this.v : this.w;
                if (i3 != 3) {
                    return i3;
                }
            } else if (i == 5) {
                int i4 = this.u;
                if (i4 != 3) {
                    return i4;
                }
                int i5 = layoutDirection == 0 ? this.w : this.v;
                if (i5 != 3) {
                    return i5;
                }
            } else if (i == 8388611) {
                int i6 = this.v;
                if (i6 != 3) {
                    return i6;
                }
                int i7 = layoutDirection == 0 ? this.t : this.u;
                if (i7 != 3) {
                    return i7;
                }
            } else if (i == 8388613) {
                int i8 = this.w;
                if (i8 != 3) {
                    return i8;
                }
                int i9 = layoutDirection == 0 ? this.u : this.t;
                if (i9 != 3) {
                    return i9;
                }
            }
            return 0;
        }
        throw new IllegalArgumentException("View " + view + " is not a drawer");
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateDefaultLayoutParams() {
        return new e(-1, -1);
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(ViewGroup.LayoutParams layoutParams) {
        if (layoutParams instanceof e) {
            return new e((e) layoutParams);
        }
        if (layoutParams instanceof ViewGroup.MarginLayoutParams) {
            return new e((ViewGroup.MarginLayoutParams) layoutParams);
        }
        return new e(layoutParams);
    }

    public float getDrawerElevation() {
        return f274e ? this.f277h : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public Drawable getStatusBarBackgroundDrawable() {
        return this.C;
    }

    public int h(View view) {
        int i = ((e) view.getLayoutParams()).f281a;
        AtomicInteger atomicInteger = q.f2214a;
        return Gravity.getAbsoluteGravity(i, getLayoutDirection());
    }

    public boolean k(View view) {
        return ((e) view.getLayoutParams()).f281a == 0;
    }

    public boolean l(View view) {
        if (m(view)) {
            return (((e) view.getLayoutParams()).f284d & 1) == 1;
        }
        throw new IllegalArgumentException("View " + view + " is not a drawer");
    }

    public boolean m(View view) {
        int i = ((e) view.getLayoutParams()).f281a;
        AtomicInteger atomicInteger = q.f2214a;
        int absoluteGravity = Gravity.getAbsoluteGravity(i, view.getLayoutDirection());
        return ((absoluteGravity & 3) == 0 && (absoluteGravity & 5) == 0) ? false : true;
    }

    public void n(View view, float f2) {
        float f3 = ((e) view.getLayoutParams()).f282b;
        float width = view.getWidth();
        int i = ((int) (width * f2)) - ((int) (f3 * width));
        if (!a(view, 3)) {
            i = -i;
        }
        view.offsetLeftAndRight(i);
        q(view, f2);
    }

    public void o(View view, boolean z) {
        if (m(view)) {
            e eVar = (e) view.getLayoutParams();
            if (this.s) {
                eVar.f282b = 1.0f;
                eVar.f284d = 1;
                s(view, true);
                r(view);
            } else if (z) {
                eVar.f284d |= 2;
                if (a(view, 3)) {
                    this.m.x(view, 0, view.getTop());
                } else {
                    this.n.x(view, getWidth() - view.getWidth(), view.getTop());
                }
            } else {
                n(view, 1.0f);
                t(0, view);
                view.setVisibility(0);
            }
            invalidate();
            return;
        }
        throw new IllegalArgumentException("View " + view + " is not a sliding drawer");
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        this.s = true;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        this.s = true;
    }

    @Override // android.view.View
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (!this.E || this.C == null) {
            return;
        }
        Object obj = this.D;
        int systemWindowInsetTop = obj != null ? ((WindowInsets) obj).getSystemWindowInsetTop() : 0;
        if (systemWindowInsetTop > 0) {
            this.C.setBounds(0, 0, getWidth(), systemWindowInsetTop);
            this.C.draw(canvas);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:8:0x001b, code lost:
        if (r0 != 3) goto L8;
     */
    /* JADX WARN: Removed duplicated region for block: B:21:0x0051 A[LOOP:1: B:11:0x0024->B:21:0x0051, LOOP_END] */
    /* JADX WARN: Removed duplicated region for block: B:53:0x004f A[SYNTHETIC] */
    @Override // android.view.ViewGroup
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        boolean z;
        View l;
        boolean z2;
        boolean z3;
        boolean z4;
        int actionMasked = motionEvent.getActionMasked();
        boolean w = this.m.w(motionEvent) | this.n.w(motionEvent);
        if (actionMasked != 0) {
            if (actionMasked != 1) {
                if (actionMasked == 2) {
                    b.l.b.e eVar = this.m;
                    int length = eVar.f2323e.length;
                    int i = 0;
                    while (true) {
                        if (i >= length) {
                            z3 = false;
                            break;
                        }
                        if (eVar.n(i)) {
                            float f2 = eVar.f2325g[i] - eVar.f2323e[i];
                            float f3 = eVar.f2326h[i] - eVar.f2324f[i];
                            float f4 = (f3 * f3) + (f2 * f2);
                            int i2 = eVar.f2321c;
                            if (f4 > i2 * i2) {
                                z4 = true;
                                if (!z4) {
                                    z3 = true;
                                    break;
                                }
                                i++;
                            }
                        }
                        z4 = false;
                        if (!z4) {
                        }
                    }
                    if (z3) {
                        this.o.b();
                        this.p.b();
                    }
                }
                z = false;
            }
            c(true);
            this.x = false;
            z = false;
        } else {
            float x = motionEvent.getX();
            float y = motionEvent.getY();
            this.A = x;
            this.B = y;
            z = this.k > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD && (l = this.m.l((int) x, (int) y)) != null && k(l);
            this.x = false;
        }
        if (!w && !z) {
            int childCount = getChildCount();
            int i3 = 0;
            while (true) {
                if (i3 >= childCount) {
                    z2 = false;
                    break;
                } else if (((e) getChildAt(i3).getLayoutParams()).f283c) {
                    z2 = true;
                    break;
                } else {
                    i3++;
                }
            }
            if (!z2 && !this.x) {
                return false;
            }
        }
        return true;
    }

    @Override // android.view.View, android.view.KeyEvent.Callback
    public boolean onKeyDown(int i, KeyEvent keyEvent) {
        if (i == 4) {
            if (f() != null) {
                keyEvent.startTracking();
                return true;
            }
        }
        return super.onKeyDown(i, keyEvent);
    }

    @Override // android.view.View, android.view.KeyEvent.Callback
    public boolean onKeyUp(int i, KeyEvent keyEvent) {
        if (i == 4) {
            View f2 = f();
            if (f2 != null && g(f2) == 0) {
                c(false);
            }
            return f2 != null;
        }
        return super.onKeyUp(i, keyEvent);
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        WindowInsets rootWindowInsets;
        int i5;
        float f2;
        int i6;
        boolean z2 = true;
        this.r = true;
        int i7 = i3 - i;
        int childCount = getChildCount();
        int i8 = 0;
        while (i8 < childCount) {
            View childAt = getChildAt(i8);
            if (childAt.getVisibility() != 8) {
                e eVar = (e) childAt.getLayoutParams();
                if (k(childAt)) {
                    int i9 = ((ViewGroup.MarginLayoutParams) eVar).leftMargin;
                    childAt.layout(i9, ((ViewGroup.MarginLayoutParams) eVar).topMargin, childAt.getMeasuredWidth() + i9, childAt.getMeasuredHeight() + ((ViewGroup.MarginLayoutParams) eVar).topMargin);
                } else {
                    int measuredWidth = childAt.getMeasuredWidth();
                    int measuredHeight = childAt.getMeasuredHeight();
                    if (a(childAt, 3)) {
                        float f3 = measuredWidth;
                        i6 = (-measuredWidth) + ((int) (eVar.f282b * f3));
                        f2 = (measuredWidth + i6) / f3;
                    } else {
                        float f4 = measuredWidth;
                        f2 = (i7 - i5) / f4;
                        i6 = i7 - ((int) (eVar.f282b * f4));
                    }
                    boolean z3 = f2 != eVar.f282b ? z2 : false;
                    int i10 = eVar.f281a & 112;
                    if (i10 == 16) {
                        int i11 = i4 - i2;
                        int i12 = (i11 - measuredHeight) / 2;
                        int i13 = ((ViewGroup.MarginLayoutParams) eVar).topMargin;
                        if (i12 < i13) {
                            i12 = i13;
                        } else {
                            int i14 = i12 + measuredHeight;
                            int i15 = i11 - ((ViewGroup.MarginLayoutParams) eVar).bottomMargin;
                            if (i14 > i15) {
                                i12 = i15 - measuredHeight;
                            }
                        }
                        childAt.layout(i6, i12, measuredWidth + i6, measuredHeight + i12);
                    } else if (i10 != 80) {
                        int i16 = ((ViewGroup.MarginLayoutParams) eVar).topMargin;
                        childAt.layout(i6, i16, measuredWidth + i6, measuredHeight + i16);
                    } else {
                        int i17 = i4 - i2;
                        childAt.layout(i6, (i17 - ((ViewGroup.MarginLayoutParams) eVar).bottomMargin) - childAt.getMeasuredHeight(), measuredWidth + i6, i17 - ((ViewGroup.MarginLayoutParams) eVar).bottomMargin);
                    }
                    if (z3) {
                        q(childAt, f2);
                    }
                    int i18 = eVar.f282b > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? 0 : 4;
                    if (childAt.getVisibility() != i18) {
                        childAt.setVisibility(i18);
                    }
                }
            }
            i8++;
            z2 = true;
        }
        if (f275f && (rootWindowInsets = getRootWindowInsets()) != null) {
            b.j.d.b h2 = w.k(rootWindowInsets, null).f2238b.h();
            b.l.b.e eVar2 = this.m;
            eVar2.p = Math.max(eVar2.q, h2.f2096b);
            b.l.b.e eVar3 = this.n;
            eVar3.p = Math.max(eVar3.q, h2.f2098d);
        }
        this.r = false;
        this.s = false;
    }

    /* JADX WARN: Removed duplicated region for block: B:19:0x0048  */
    @Override // android.view.View
    @SuppressLint({"WrongConstant"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void onMeasure(int i, int i2) {
        boolean z;
        int childCount;
        int mode = View.MeasureSpec.getMode(i);
        int mode2 = View.MeasureSpec.getMode(i2);
        int size = View.MeasureSpec.getSize(i);
        int size2 = View.MeasureSpec.getSize(i2);
        if (mode != 1073741824 || mode2 != 1073741824) {
            if (!isInEditMode()) {
                throw new IllegalArgumentException("DrawerLayout must be measured with MeasureSpec.EXACTLY.");
            }
            if (mode == 0) {
                size = 300;
            }
            if (mode2 == 0) {
                size2 = 300;
            }
        }
        setMeasuredDimension(size, size2);
        int i3 = 0;
        if (this.D != null) {
            AtomicInteger atomicInteger = q.f2214a;
            if (getFitsSystemWindows()) {
                z = true;
                AtomicInteger atomicInteger2 = q.f2214a;
                int layoutDirection = getLayoutDirection();
                childCount = getChildCount();
                int i4 = 0;
                boolean z2 = false;
                boolean z3 = false;
                while (i3 < childCount) {
                    View childAt = getChildAt(i3);
                    if (childAt.getVisibility() != 8) {
                        e eVar = (e) childAt.getLayoutParams();
                        if (z) {
                            int absoluteGravity = Gravity.getAbsoluteGravity(eVar.f281a, layoutDirection);
                            if (childAt.getFitsSystemWindows()) {
                                WindowInsets windowInsets = (WindowInsets) this.D;
                                if (absoluteGravity == 3) {
                                    windowInsets = windowInsets.replaceSystemWindowInsets(windowInsets.getSystemWindowInsetLeft(), windowInsets.getSystemWindowInsetTop(), i4, windowInsets.getSystemWindowInsetBottom());
                                } else if (absoluteGravity == 5) {
                                    windowInsets = windowInsets.replaceSystemWindowInsets(i4, windowInsets.getSystemWindowInsetTop(), windowInsets.getSystemWindowInsetRight(), windowInsets.getSystemWindowInsetBottom());
                                }
                                childAt.dispatchApplyWindowInsets(windowInsets);
                            } else {
                                WindowInsets windowInsets2 = (WindowInsets) this.D;
                                if (absoluteGravity == 3) {
                                    windowInsets2 = windowInsets2.replaceSystemWindowInsets(windowInsets2.getSystemWindowInsetLeft(), windowInsets2.getSystemWindowInsetTop(), i4, windowInsets2.getSystemWindowInsetBottom());
                                } else if (absoluteGravity == 5) {
                                    windowInsets2 = windowInsets2.replaceSystemWindowInsets(i4, windowInsets2.getSystemWindowInsetTop(), windowInsets2.getSystemWindowInsetRight(), windowInsets2.getSystemWindowInsetBottom());
                                }
                                ((ViewGroup.MarginLayoutParams) eVar).leftMargin = windowInsets2.getSystemWindowInsetLeft();
                                ((ViewGroup.MarginLayoutParams) eVar).topMargin = windowInsets2.getSystemWindowInsetTop();
                                ((ViewGroup.MarginLayoutParams) eVar).rightMargin = windowInsets2.getSystemWindowInsetRight();
                                ((ViewGroup.MarginLayoutParams) eVar).bottomMargin = windowInsets2.getSystemWindowInsetBottom();
                            }
                        }
                        if (k(childAt)) {
                            childAt.measure(View.MeasureSpec.makeMeasureSpec((size - ((ViewGroup.MarginLayoutParams) eVar).leftMargin) - ((ViewGroup.MarginLayoutParams) eVar).rightMargin, 1073741824), View.MeasureSpec.makeMeasureSpec((size2 - ((ViewGroup.MarginLayoutParams) eVar).topMargin) - ((ViewGroup.MarginLayoutParams) eVar).bottomMargin, 1073741824));
                        } else if (m(childAt)) {
                            if (f274e) {
                                float elevation = childAt.getElevation();
                                float f2 = this.f277h;
                                if (elevation != f2) {
                                    childAt.setElevation(f2);
                                }
                            }
                            int h2 = h(childAt) & 7;
                            if (h2 == 3) {
                                i4 = 1;
                            }
                            if ((i4 != 0 && z2) || (i4 == 0 && z3)) {
                                StringBuilder x = c.b.a.a.a.x("Child drawer has absolute gravity ");
                                x.append(i(h2));
                                x.append(" but this ");
                                x.append("DrawerLayout");
                                x.append(" already has a drawer view along that edge");
                                throw new IllegalStateException(x.toString());
                            }
                            if (i4 != 0) {
                                z2 = true;
                            } else {
                                z3 = true;
                            }
                            childAt.measure(ViewGroup.getChildMeasureSpec(i, this.i + ((ViewGroup.MarginLayoutParams) eVar).leftMargin + ((ViewGroup.MarginLayoutParams) eVar).rightMargin, ((ViewGroup.MarginLayoutParams) eVar).width), ViewGroup.getChildMeasureSpec(i2, ((ViewGroup.MarginLayoutParams) eVar).topMargin + ((ViewGroup.MarginLayoutParams) eVar).bottomMargin, ((ViewGroup.MarginLayoutParams) eVar).height));
                            i3++;
                            i4 = 0;
                        } else {
                            throw new IllegalStateException("Child " + childAt + " at index " + i3 + " does not have a valid layout_gravity - must be Gravity.LEFT, Gravity.RIGHT or Gravity.NO_GRAVITY");
                        }
                    }
                    i3++;
                    i4 = 0;
                }
            }
        }
        z = false;
        AtomicInteger atomicInteger22 = q.f2214a;
        int layoutDirection2 = getLayoutDirection();
        childCount = getChildCount();
        int i42 = 0;
        boolean z22 = false;
        boolean z32 = false;
        while (i3 < childCount) {
        }
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        View d2;
        if (!(parcelable instanceof f)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        f fVar = (f) parcelable;
        super.onRestoreInstanceState(fVar.getSuperState());
        int i = fVar.f285b;
        if (i != 0 && (d2 = d(i)) != null) {
            o(d2, true);
        }
        int i2 = fVar.f286c;
        if (i2 != 3) {
            p(i2, 3);
        }
        int i3 = fVar.f287d;
        if (i3 != 3) {
            p(i3, 5);
        }
        int i4 = fVar.f288e;
        if (i4 != 3) {
            p(i4, 8388611);
        }
        int i5 = fVar.f289f;
        if (i5 != 3) {
            p(i5, 8388613);
        }
    }

    @Override // android.view.View
    public void onRtlPropertiesChanged(int i) {
        if (f274e) {
            return;
        }
        AtomicInteger atomicInteger = q.f2214a;
        getLayoutDirection();
        getLayoutDirection();
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        f fVar = new f(super.onSaveInstanceState());
        int childCount = getChildCount();
        for (int i = 0; i < childCount; i++) {
            e eVar = (e) getChildAt(i).getLayoutParams();
            int i2 = eVar.f284d;
            boolean z = i2 == 1;
            boolean z2 = i2 == 2;
            if (z || z2) {
                fVar.f285b = eVar.f281a;
                break;
            }
        }
        fVar.f286c = this.t;
        fVar.f287d = this.u;
        fVar.f288e = this.v;
        fVar.f289f = this.w;
        return fVar;
    }

    /* JADX WARN: Code restructure failed: missing block: B:18:0x0056, code lost:
        if (g(r7) != 2) goto L19;
     */
    @Override // android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouchEvent(MotionEvent motionEvent) {
        this.m.p(motionEvent);
        this.n.p(motionEvent);
        int action = motionEvent.getAction() & 255;
        boolean z = false;
        if (action == 0) {
            float x = motionEvent.getX();
            float y = motionEvent.getY();
            this.A = x;
            this.B = y;
            this.x = false;
        } else if (action == 1) {
            float x2 = motionEvent.getX();
            float y2 = motionEvent.getY();
            View l = this.m.l((int) x2, (int) y2);
            if (l != null && k(l)) {
                float f2 = x2 - this.A;
                float f3 = y2 - this.B;
                int i = this.m.f2321c;
                if ((f3 * f3) + (f2 * f2) < i * i) {
                    View e2 = e();
                    if (e2 != null) {
                    }
                }
            }
            z = true;
            c(z);
        } else if (action == 3) {
            c(true);
            this.x = false;
        }
        return true;
    }

    public void p(int i, int i2) {
        View d2;
        AtomicInteger atomicInteger = q.f2214a;
        int absoluteGravity = Gravity.getAbsoluteGravity(i2, getLayoutDirection());
        if (i2 == 3) {
            this.t = i;
        } else if (i2 == 5) {
            this.u = i;
        } else if (i2 == 8388611) {
            this.v = i;
        } else if (i2 == 8388613) {
            this.w = i;
        }
        if (i != 0) {
            (absoluteGravity == 3 ? this.m : this.n).a();
        }
        if (i != 1) {
            if (i == 2 && (d2 = d(absoluteGravity)) != null) {
                o(d2, true);
                return;
            }
            return;
        }
        View d3 = d(absoluteGravity);
        if (d3 != null) {
            b(d3);
        }
    }

    public void q(View view, float f2) {
        e eVar = (e) view.getLayoutParams();
        if (f2 == eVar.f282b) {
            return;
        }
        eVar.f282b = f2;
        List<d> list = this.z;
        if (list == null) {
            return;
        }
        int size = list.size();
        while (true) {
            size--;
            if (size < 0) {
                return;
            }
            this.z.get(size).d(view, f2);
        }
    }

    public final void r(View view) {
        b.a aVar = b.a.f2269h;
        q.k(aVar.a(), view);
        q.i(view, 0);
        if (!l(view) || g(view) == 2) {
            return;
        }
        q.l(view, aVar, null, this.I);
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public void requestDisallowInterceptTouchEvent(boolean z) {
        super.requestDisallowInterceptTouchEvent(z);
        if (z) {
            c(true);
        }
    }

    @Override // android.view.View, android.view.ViewParent
    public void requestLayout() {
        if (this.r) {
            return;
        }
        super.requestLayout();
    }

    public final void s(View view, boolean z) {
        int childCount = getChildCount();
        for (int i = 0; i < childCount; i++) {
            View childAt = getChildAt(i);
            if ((!z && !m(childAt)) || (z && childAt == view)) {
                AtomicInteger atomicInteger = q.f2214a;
                childAt.setImportantForAccessibility(1);
            } else {
                AtomicInteger atomicInteger2 = q.f2214a;
                childAt.setImportantForAccessibility(4);
            }
        }
    }

    public void setDrawerElevation(float f2) {
        this.f277h = f2;
        for (int i = 0; i < getChildCount(); i++) {
            View childAt = getChildAt(i);
            if (m(childAt)) {
                float f3 = this.f277h;
                AtomicInteger atomicInteger = q.f2214a;
                childAt.setElevation(f3);
            }
        }
    }

    @Deprecated
    public void setDrawerListener(d dVar) {
        List<d> list;
        d dVar2 = this.y;
        if (dVar2 != null && dVar2 != null && (list = this.z) != null) {
            list.remove(dVar2);
        }
        if (dVar != null) {
            if (this.z == null) {
                this.z = new ArrayList();
            }
            this.z.add(dVar);
        }
        this.y = dVar;
    }

    public void setDrawerLockMode(int i) {
        p(i, 3);
        p(i, 5);
    }

    public void setScrimColor(int i) {
        this.j = i;
        invalidate();
    }

    public void setStatusBarBackground(Drawable drawable) {
        this.C = drawable;
        invalidate();
    }

    public void setStatusBarBackgroundColor(int i) {
        this.C = new ColorDrawable(i);
        invalidate();
    }

    public void t(int i, View view) {
        View rootView;
        int i2 = this.m.f2320b;
        int i3 = this.n.f2320b;
        int i4 = 2;
        if (i2 == 1 || i3 == 1) {
            i4 = 1;
        } else if (i2 != 2 && i3 != 2) {
            i4 = 0;
        }
        if (view != null && i == 0) {
            float f2 = ((e) view.getLayoutParams()).f282b;
            if (f2 == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                e eVar = (e) view.getLayoutParams();
                if ((eVar.f284d & 1) == 1) {
                    eVar.f284d = 0;
                    List<d> list = this.z;
                    if (list != null) {
                        for (int size = list.size() - 1; size >= 0; size--) {
                            this.z.get(size).b(view);
                        }
                    }
                    s(view, false);
                    r(view);
                    if (hasWindowFocus() && (rootView = getRootView()) != null) {
                        rootView.sendAccessibilityEvent(32);
                    }
                }
            } else if (f2 == 1.0f) {
                e eVar2 = (e) view.getLayoutParams();
                if ((eVar2.f284d & 1) == 0) {
                    eVar2.f284d = 1;
                    List<d> list2 = this.z;
                    if (list2 != null) {
                        for (int size2 = list2.size() - 1; size2 >= 0; size2--) {
                            this.z.get(size2).a(view);
                        }
                    }
                    s(view, true);
                    r(view);
                    if (hasWindowFocus()) {
                        sendAccessibilityEvent(32);
                    }
                }
            }
        }
        if (i4 != this.q) {
            this.q = i4;
            List<d> list3 = this.z;
            if (list3 != null) {
                for (int size3 = list3.size() - 1; size3 >= 0; size3--) {
                    this.z.get(size3).c(i4);
                }
            }
        }
    }

    public void setStatusBarBackground(int i) {
        Drawable drawable;
        if (i != 0) {
            Context context = getContext();
            Object obj = b.j.c.a.f2074a;
            drawable = context.getDrawable(i);
        } else {
            drawable = null;
        }
        this.C = drawable;
        invalidate();
    }

    /* loaded from: classes.dex */
    public static class e extends ViewGroup.MarginLayoutParams {

        /* renamed from: a  reason: collision with root package name */
        public int f281a;

        /* renamed from: b  reason: collision with root package name */
        public float f282b;

        /* renamed from: c  reason: collision with root package name */
        public boolean f283c;

        /* renamed from: d  reason: collision with root package name */
        public int f284d;

        public e(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f281a = 0;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, DrawerLayout.f272c);
            this.f281a = obtainStyledAttributes.getInt(0, 0);
            obtainStyledAttributes.recycle();
        }

        public e(int i, int i2) {
            super(i, i2);
            this.f281a = 0;
        }

        public e(e eVar) {
            super((ViewGroup.MarginLayoutParams) eVar);
            this.f281a = 0;
            this.f281a = eVar.f281a;
        }

        public e(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f281a = 0;
        }

        public e(ViewGroup.MarginLayoutParams marginLayoutParams) {
            super(marginLayoutParams);
            this.f281a = 0;
        }
    }

    @Override // android.view.ViewGroup
    public ViewGroup.LayoutParams generateLayoutParams(AttributeSet attributeSet) {
        return new e(getContext(), attributeSet);
    }

    /* loaded from: classes.dex */
    public static class f extends b.l.a.a {
        public static final Parcelable.Creator<f> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f285b;

        /* renamed from: c  reason: collision with root package name */
        public int f286c;

        /* renamed from: d  reason: collision with root package name */
        public int f287d;

        /* renamed from: e  reason: collision with root package name */
        public int f288e;

        /* renamed from: f  reason: collision with root package name */
        public int f289f;

        /* loaded from: classes.dex */
        public class a implements Parcelable.ClassLoaderCreator<f> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.ClassLoaderCreator
            public f createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new f(parcel, classLoader);
            }

            @Override // android.os.Parcelable.Creator
            public Object[] newArray(int i) {
                return new f[i];
            }

            @Override // android.os.Parcelable.Creator
            public Object createFromParcel(Parcel parcel) {
                return new f(parcel, null);
            }
        }

        public f(Parcel parcel, ClassLoader classLoader) {
            super(parcel, classLoader);
            this.f285b = 0;
            this.f285b = parcel.readInt();
            this.f286c = parcel.readInt();
            this.f287d = parcel.readInt();
            this.f288e = parcel.readInt();
            this.f289f = parcel.readInt();
        }

        @Override // b.l.a.a, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeInt(this.f285b);
            parcel.writeInt(this.f286c);
            parcel.writeInt(this.f287d);
            parcel.writeInt(this.f288e);
            parcel.writeInt(this.f289f);
        }

        public f(Parcelable parcelable) {
            super(parcelable);
            this.f285b = 0;
        }
    }
}