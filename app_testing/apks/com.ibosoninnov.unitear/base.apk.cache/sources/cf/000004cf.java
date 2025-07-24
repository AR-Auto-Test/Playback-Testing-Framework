package b.l.b;

import android.content.Context;
import android.util.Log;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.animation.Interpolator;
import android.widget.OverScroller;
import b.j.j.q;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ViewDragHelper.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public static final Interpolator f2319a = new a();

    /* renamed from: b  reason: collision with root package name */
    public int f2320b;

    /* renamed from: c  reason: collision with root package name */
    public int f2321c;

    /* renamed from: e  reason: collision with root package name */
    public float[] f2323e;

    /* renamed from: f  reason: collision with root package name */
    public float[] f2324f;

    /* renamed from: g  reason: collision with root package name */
    public float[] f2325g;

    /* renamed from: h  reason: collision with root package name */
    public float[] f2326h;
    public int[] i;
    public int[] j;
    public int[] k;
    public int l;
    public VelocityTracker m;
    public float n;
    public float o;
    public int p;
    public final int q;
    public int r;
    public OverScroller s;
    public final c t;
    public View u;
    public boolean v;
    public final ViewGroup w;

    /* renamed from: d  reason: collision with root package name */
    public int f2322d = -1;
    public final Runnable x = new b();

    /* compiled from: ViewDragHelper.java */
    /* loaded from: classes.dex */
    public class a implements Interpolator {
        @Override // android.animation.TimeInterpolator
        public float getInterpolation(float f2) {
            float f3 = f2 - 1.0f;
            return (f3 * f3 * f3 * f3 * f3) + 1.0f;
        }
    }

    /* compiled from: ViewDragHelper.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            e.this.u(0);
        }
    }

    /* compiled from: ViewDragHelper.java */
    /* loaded from: classes.dex */
    public static abstract class c {
        public abstract int clampViewPositionHorizontal(View view, int i, int i2);

        public abstract int clampViewPositionVertical(View view, int i, int i2);

        public int getOrderedChildIndex(int i) {
            return i;
        }

        public int getViewHorizontalDragRange(View view) {
            return 0;
        }

        public int getViewVerticalDragRange(View view) {
            return 0;
        }

        public void onEdgeDragStarted(int i, int i2) {
        }

        public boolean onEdgeLock(int i) {
            return false;
        }

        public void onEdgeTouched(int i, int i2) {
        }

        public void onViewCaptured(View view, int i) {
        }

        public abstract void onViewDragStateChanged(int i);

        public abstract void onViewPositionChanged(View view, int i, int i2, int i3, int i4);

        public abstract void onViewReleased(View view, float f2, float f3);

        public abstract boolean tryCaptureView(View view, int i);
    }

    public e(Context context, ViewGroup viewGroup, c cVar) {
        if (cVar != null) {
            this.w = viewGroup;
            this.t = cVar;
            ViewConfiguration viewConfiguration = ViewConfiguration.get(context);
            int i = (int) ((context.getResources().getDisplayMetrics().density * 20.0f) + 0.5f);
            this.q = i;
            this.p = i;
            this.f2321c = viewConfiguration.getScaledTouchSlop();
            this.n = viewConfiguration.getScaledMaximumFlingVelocity();
            this.o = viewConfiguration.getScaledMinimumFlingVelocity();
            this.s = new OverScroller(context, f2319a);
            return;
        }
        throw new IllegalArgumentException("Callback may not be null");
    }

    public static e j(ViewGroup viewGroup, float f2, c cVar) {
        e eVar = new e(viewGroup.getContext(), viewGroup, cVar);
        eVar.f2321c = (int) ((1.0f / f2) * eVar.f2321c);
        return eVar;
    }

    public void a() {
        this.f2322d = -1;
        float[] fArr = this.f2323e;
        if (fArr != null) {
            Arrays.fill(fArr, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Arrays.fill(this.f2324f, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Arrays.fill(this.f2325g, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Arrays.fill(this.f2326h, (float) StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            Arrays.fill(this.i, 0);
            Arrays.fill(this.j, 0);
            Arrays.fill(this.k, 0);
            this.l = 0;
        }
        VelocityTracker velocityTracker = this.m;
        if (velocityTracker != null) {
            velocityTracker.recycle();
            this.m = null;
        }
    }

    public void b(View view, int i) {
        if (view.getParent() == this.w) {
            this.u = view;
            this.f2322d = i;
            this.t.onViewCaptured(view, i);
            u(1);
            return;
        }
        StringBuilder x = c.b.a.a.a.x("captureChildView: parameter must be a descendant of the ViewDragHelper's tracked parent view (");
        x.append(this.w);
        x.append(")");
        throw new IllegalArgumentException(x.toString());
    }

    public final boolean c(float f2, float f3, int i, int i2) {
        float abs = Math.abs(f2);
        float abs2 = Math.abs(f3);
        if ((this.i[i] & i2) != i2 || (this.r & i2) == 0 || (this.k[i] & i2) == i2 || (this.j[i] & i2) == i2) {
            return false;
        }
        int i3 = this.f2321c;
        if (abs > i3 || abs2 > i3) {
            if (abs >= abs2 * 0.5f || !this.t.onEdgeLock(i2)) {
                return (this.j[i] & i2) == 0 && abs > ((float) this.f2321c);
            }
            int[] iArr = this.k;
            iArr[i] = iArr[i] | i2;
            return false;
        }
        return false;
    }

    public final boolean d(View view, float f2, float f3) {
        if (view == null) {
            return false;
        }
        boolean z = this.t.getViewHorizontalDragRange(view) > 0;
        boolean z2 = this.t.getViewVerticalDragRange(view) > 0;
        if (!z || !z2) {
            return z ? Math.abs(f2) > ((float) this.f2321c) : z2 && Math.abs(f3) > ((float) this.f2321c);
        }
        float f4 = f3 * f3;
        int i = this.f2321c;
        return f4 + (f2 * f2) > ((float) (i * i));
    }

    public final float e(float f2, float f3, float f4) {
        float abs = Math.abs(f2);
        return abs < f3 ? StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : abs > f4 ? f2 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD ? f4 : -f4 : f2;
    }

    public final int f(int i, int i2, int i3) {
        int abs = Math.abs(i);
        if (abs < i2) {
            return 0;
        }
        return abs > i3 ? i > 0 ? i3 : -i3 : i;
    }

    public final void g(int i) {
        if (this.f2323e == null || !n(i)) {
            return;
        }
        this.f2323e[i] = 0.0f;
        this.f2324f[i] = 0.0f;
        this.f2325g[i] = 0.0f;
        this.f2326h[i] = 0.0f;
        this.i[i] = 0;
        this.j[i] = 0;
        this.k[i] = 0;
        this.l = (~(1 << i)) & this.l;
    }

    public final int h(int i, int i2, int i3) {
        int width;
        int abs;
        if (i == 0) {
            return 0;
        }
        float width2 = this.w.getWidth() / 2;
        float sin = (((float) Math.sin((Math.min(1.0f, Math.abs(i) / width) - 0.5f) * 0.47123894f)) * width2) + width2;
        int abs2 = Math.abs(i2);
        if (abs2 > 0) {
            abs = Math.round(Math.abs(sin / abs2) * 1000.0f) * 4;
        } else {
            abs = (int) (((Math.abs(i) / i3) + 1.0f) * 256.0f);
        }
        return Math.min(abs, 600);
    }

    public boolean i(boolean z) {
        if (this.f2320b == 2) {
            boolean computeScrollOffset = this.s.computeScrollOffset();
            int currX = this.s.getCurrX();
            int currY = this.s.getCurrY();
            int left = currX - this.u.getLeft();
            int top = currY - this.u.getTop();
            if (left != 0) {
                View view = this.u;
                AtomicInteger atomicInteger = q.f2214a;
                view.offsetLeftAndRight(left);
            }
            if (top != 0) {
                View view2 = this.u;
                AtomicInteger atomicInteger2 = q.f2214a;
                view2.offsetTopAndBottom(top);
            }
            if (left != 0 || top != 0) {
                this.t.onViewPositionChanged(this.u, currX, currY, left, top);
            }
            if (computeScrollOffset && currX == this.s.getFinalX() && currY == this.s.getFinalY()) {
                this.s.abortAnimation();
                computeScrollOffset = false;
            }
            if (!computeScrollOffset) {
                if (z) {
                    this.w.post(this.x);
                } else {
                    u(0);
                }
            }
        }
        return this.f2320b == 2;
    }

    public final void k(float f2, float f3) {
        this.v = true;
        this.t.onViewReleased(this.u, f2, f3);
        this.v = false;
        if (this.f2320b == 1) {
            u(0);
        }
    }

    public View l(int i, int i2) {
        for (int childCount = this.w.getChildCount() - 1; childCount >= 0; childCount--) {
            View childAt = this.w.getChildAt(this.t.getOrderedChildIndex(childCount));
            if (i >= childAt.getLeft() && i < childAt.getRight() && i2 >= childAt.getTop() && i2 < childAt.getBottom()) {
                return childAt;
            }
        }
        return null;
    }

    public final boolean m(int i, int i2, int i3, int i4) {
        float f2;
        float f3;
        float f4;
        float f5;
        int left = this.u.getLeft();
        int top = this.u.getTop();
        int i5 = i - left;
        int i6 = i2 - top;
        if (i5 == 0 && i6 == 0) {
            this.s.abortAnimation();
            u(0);
            return false;
        }
        View view = this.u;
        int f6 = f(i3, (int) this.o, (int) this.n);
        int f7 = f(i4, (int) this.o, (int) this.n);
        int abs = Math.abs(i5);
        int abs2 = Math.abs(i6);
        int abs3 = Math.abs(f6);
        int abs4 = Math.abs(f7);
        int i7 = abs3 + abs4;
        int i8 = abs + abs2;
        if (f6 != 0) {
            f2 = abs3;
            f3 = i7;
        } else {
            f2 = abs;
            f3 = i8;
        }
        float f8 = f2 / f3;
        if (f7 != 0) {
            f4 = abs4;
            f5 = i7;
        } else {
            f4 = abs2;
            f5 = i8;
        }
        int h2 = h(i5, f6, this.t.getViewHorizontalDragRange(view));
        this.s.startScroll(left, top, i5, i6, (int) ((h(i6, f7, this.t.getViewVerticalDragRange(view)) * (f4 / f5)) + (h2 * f8)));
        u(2);
        return true;
    }

    public boolean n(int i) {
        return ((1 << i) & this.l) != 0;
    }

    public final boolean o(int i) {
        if (n(i)) {
            return true;
        }
        Log.e("ViewDragHelper", "Ignoring pointerId=" + i + " because ACTION_DOWN was not received for this pointer before ACTION_MOVE. It likely happened because  ViewDragHelper did not receive all the events in the event stream.");
        return false;
    }

    public void p(MotionEvent motionEvent) {
        int i;
        int actionMasked = motionEvent.getActionMasked();
        int actionIndex = motionEvent.getActionIndex();
        if (actionMasked == 0) {
            a();
        }
        if (this.m == null) {
            this.m = VelocityTracker.obtain();
        }
        this.m.addMovement(motionEvent);
        int i2 = 0;
        if (actionMasked == 0) {
            float x = motionEvent.getX();
            float y = motionEvent.getY();
            int pointerId = motionEvent.getPointerId(0);
            View l = l((int) x, (int) y);
            s(x, y, pointerId);
            y(l, pointerId);
            int i3 = this.i[pointerId];
            int i4 = this.r;
            if ((i3 & i4) != 0) {
                this.t.onEdgeTouched(i3 & i4, pointerId);
            }
        } else if (actionMasked == 1) {
            if (this.f2320b == 1) {
                q();
            }
            a();
        } else if (actionMasked == 2) {
            if (this.f2320b == 1) {
                if (o(this.f2322d)) {
                    int findPointerIndex = motionEvent.findPointerIndex(this.f2322d);
                    float x2 = motionEvent.getX(findPointerIndex);
                    float y2 = motionEvent.getY(findPointerIndex);
                    float[] fArr = this.f2325g;
                    int i5 = this.f2322d;
                    int i6 = (int) (x2 - fArr[i5]);
                    int i7 = (int) (y2 - this.f2326h[i5]);
                    int left = this.u.getLeft() + i6;
                    int top = this.u.getTop() + i7;
                    int left2 = this.u.getLeft();
                    int top2 = this.u.getTop();
                    if (i6 != 0) {
                        left = this.t.clampViewPositionHorizontal(this.u, left, i6);
                        AtomicInteger atomicInteger = q.f2214a;
                        this.u.offsetLeftAndRight(left - left2);
                    }
                    int i8 = left;
                    if (i7 != 0) {
                        top = this.t.clampViewPositionVertical(this.u, top, i7);
                        AtomicInteger atomicInteger2 = q.f2214a;
                        this.u.offsetTopAndBottom(top - top2);
                    }
                    int i9 = top;
                    if (i6 != 0 || i7 != 0) {
                        this.t.onViewPositionChanged(this.u, i8, i9, i8 - left2, i9 - top2);
                    }
                    t(motionEvent);
                    return;
                }
                return;
            }
            int pointerCount = motionEvent.getPointerCount();
            while (i2 < pointerCount) {
                int pointerId2 = motionEvent.getPointerId(i2);
                if (o(pointerId2)) {
                    float x3 = motionEvent.getX(i2);
                    float y3 = motionEvent.getY(i2);
                    float f2 = x3 - this.f2323e[pointerId2];
                    float f3 = y3 - this.f2324f[pointerId2];
                    r(f2, f3, pointerId2);
                    if (this.f2320b != 1) {
                        View l2 = l((int) x3, (int) y3);
                        if (d(l2, f2, f3) && y(l2, pointerId2)) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                i2++;
            }
            t(motionEvent);
        } else if (actionMasked == 3) {
            if (this.f2320b == 1) {
                k(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            a();
        } else if (actionMasked != 5) {
            if (actionMasked != 6) {
                return;
            }
            int pointerId3 = motionEvent.getPointerId(actionIndex);
            if (this.f2320b == 1 && pointerId3 == this.f2322d) {
                int pointerCount2 = motionEvent.getPointerCount();
                while (true) {
                    if (i2 >= pointerCount2) {
                        i = -1;
                        break;
                    }
                    int pointerId4 = motionEvent.getPointerId(i2);
                    if (pointerId4 != this.f2322d) {
                        View l3 = l((int) motionEvent.getX(i2), (int) motionEvent.getY(i2));
                        View view = this.u;
                        if (l3 == view && y(view, pointerId4)) {
                            i = this.f2322d;
                            break;
                        }
                    }
                    i2++;
                }
                if (i == -1) {
                    q();
                }
            }
            g(pointerId3);
        } else {
            int pointerId5 = motionEvent.getPointerId(actionIndex);
            float x4 = motionEvent.getX(actionIndex);
            float y4 = motionEvent.getY(actionIndex);
            s(x4, y4, pointerId5);
            if (this.f2320b == 0) {
                y(l((int) x4, (int) y4), pointerId5);
                int i10 = this.i[pointerId5];
                int i11 = this.r;
                if ((i10 & i11) != 0) {
                    this.t.onEdgeTouched(i10 & i11, pointerId5);
                    return;
                }
                return;
            }
            int i12 = (int) x4;
            int i13 = (int) y4;
            View view2 = this.u;
            if (view2 != null && i12 >= view2.getLeft() && i12 < view2.getRight() && i13 >= view2.getTop() && i13 < view2.getBottom()) {
                i2 = 1;
            }
            if (i2 != 0) {
                y(this.u, pointerId5);
            }
        }
    }

    public final void q() {
        this.m.computeCurrentVelocity(1000, this.n);
        k(e(this.m.getXVelocity(this.f2322d), this.o, this.n), e(this.m.getYVelocity(this.f2322d), this.o, this.n));
    }

    public final void r(float f2, float f3, int i) {
        int i2 = c(f2, f3, i, 1) ? 1 : 0;
        if (c(f3, f2, i, 4)) {
            i2 |= 4;
        }
        if (c(f2, f3, i, 2)) {
            i2 |= 2;
        }
        if (c(f3, f2, i, 8)) {
            i2 |= 8;
        }
        if (i2 != 0) {
            int[] iArr = this.j;
            iArr[i] = iArr[i] | i2;
            this.t.onEdgeDragStarted(i2, i);
        }
    }

    public final void s(float f2, float f3, int i) {
        float[] fArr = this.f2323e;
        if (fArr == null || fArr.length <= i) {
            int i2 = i + 1;
            float[] fArr2 = new float[i2];
            float[] fArr3 = new float[i2];
            float[] fArr4 = new float[i2];
            float[] fArr5 = new float[i2];
            int[] iArr = new int[i2];
            int[] iArr2 = new int[i2];
            int[] iArr3 = new int[i2];
            if (fArr != null) {
                System.arraycopy(fArr, 0, fArr2, 0, fArr.length);
                float[] fArr6 = this.f2324f;
                System.arraycopy(fArr6, 0, fArr3, 0, fArr6.length);
                float[] fArr7 = this.f2325g;
                System.arraycopy(fArr7, 0, fArr4, 0, fArr7.length);
                float[] fArr8 = this.f2326h;
                System.arraycopy(fArr8, 0, fArr5, 0, fArr8.length);
                int[] iArr4 = this.i;
                System.arraycopy(iArr4, 0, iArr, 0, iArr4.length);
                int[] iArr5 = this.j;
                System.arraycopy(iArr5, 0, iArr2, 0, iArr5.length);
                int[] iArr6 = this.k;
                System.arraycopy(iArr6, 0, iArr3, 0, iArr6.length);
            }
            this.f2323e = fArr2;
            this.f2324f = fArr3;
            this.f2325g = fArr4;
            this.f2326h = fArr5;
            this.i = iArr;
            this.j = iArr2;
            this.k = iArr3;
        }
        float[] fArr9 = this.f2323e;
        this.f2325g[i] = f2;
        fArr9[i] = f2;
        float[] fArr10 = this.f2324f;
        this.f2326h[i] = f3;
        fArr10[i] = f3;
        int[] iArr7 = this.i;
        int i3 = (int) f2;
        int i4 = (int) f3;
        int i5 = i3 < this.w.getLeft() + this.p ? 1 : 0;
        if (i4 < this.w.getTop() + this.p) {
            i5 |= 4;
        }
        if (i3 > this.w.getRight() - this.p) {
            i5 |= 2;
        }
        if (i4 > this.w.getBottom() - this.p) {
            i5 |= 8;
        }
        iArr7[i] = i5;
        this.l |= 1 << i;
    }

    public final void t(MotionEvent motionEvent) {
        int pointerCount = motionEvent.getPointerCount();
        for (int i = 0; i < pointerCount; i++) {
            int pointerId = motionEvent.getPointerId(i);
            if (o(pointerId)) {
                float x = motionEvent.getX(i);
                float y = motionEvent.getY(i);
                this.f2325g[pointerId] = x;
                this.f2326h[pointerId] = y;
            }
        }
    }

    public void u(int i) {
        this.w.removeCallbacks(this.x);
        if (this.f2320b != i) {
            this.f2320b = i;
            this.t.onViewDragStateChanged(i);
            if (this.f2320b == 0) {
                this.u = null;
            }
        }
    }

    public boolean v(int i, int i2) {
        if (this.v) {
            return m(i, i2, (int) this.m.getXVelocity(this.f2322d), (int) this.m.getYVelocity(this.f2322d));
        }
        throw new IllegalStateException("Cannot settleCapturedViewAt outside of a call to Callback#onViewReleased");
    }

    /* JADX WARN: Code restructure failed: missing block: B:49:0x00de, code lost:
        if (r12 != r11) goto L58;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean w(MotionEvent motionEvent) {
        boolean z;
        View l;
        int actionMasked = motionEvent.getActionMasked();
        int actionIndex = motionEvent.getActionIndex();
        if (actionMasked == 0) {
            a();
        }
        if (this.m == null) {
            this.m = VelocityTracker.obtain();
        }
        this.m.addMovement(motionEvent);
        if (actionMasked != 0) {
            if (actionMasked != 1) {
                if (actionMasked != 2) {
                    if (actionMasked != 3) {
                        if (actionMasked == 5) {
                            int pointerId = motionEvent.getPointerId(actionIndex);
                            float x = motionEvent.getX(actionIndex);
                            float y = motionEvent.getY(actionIndex);
                            s(x, y, pointerId);
                            int i = this.f2320b;
                            if (i == 0) {
                                int i2 = this.i[pointerId] & this.r;
                                if (i2 != 0) {
                                    this.t.onEdgeTouched(i2, pointerId);
                                }
                            } else if (i == 2 && (l = l((int) x, (int) y)) == this.u) {
                                y(l, pointerId);
                            }
                        } else if (actionMasked == 6) {
                            g(motionEvent.getPointerId(actionIndex));
                        }
                    }
                } else if (this.f2323e != null && this.f2324f != null) {
                    int pointerCount = motionEvent.getPointerCount();
                    for (int i3 = 0; i3 < pointerCount; i3++) {
                        int pointerId2 = motionEvent.getPointerId(i3);
                        if (o(pointerId2)) {
                            float x2 = motionEvent.getX(i3);
                            float y2 = motionEvent.getY(i3);
                            float f2 = x2 - this.f2323e[pointerId2];
                            float f3 = y2 - this.f2324f[pointerId2];
                            View l2 = l((int) x2, (int) y2);
                            boolean z2 = l2 != null && d(l2, f2, f3);
                            if (z2) {
                                int left = l2.getLeft();
                                int i4 = (int) f2;
                                int clampViewPositionHorizontal = this.t.clampViewPositionHorizontal(l2, left + i4, i4);
                                int top = l2.getTop();
                                int i5 = (int) f3;
                                int clampViewPositionVertical = this.t.clampViewPositionVertical(l2, top + i5, i5);
                                int viewHorizontalDragRange = this.t.getViewHorizontalDragRange(l2);
                                int viewVerticalDragRange = this.t.getViewVerticalDragRange(l2);
                                if (viewHorizontalDragRange != 0) {
                                    if (viewHorizontalDragRange > 0) {
                                    }
                                }
                                if (viewVerticalDragRange == 0) {
                                    break;
                                } else if (viewVerticalDragRange > 0 && clampViewPositionVertical == top) {
                                    break;
                                }
                            }
                            r(f2, f3, pointerId2);
                            if (this.f2320b != 1) {
                                if (z2 && y(l2, pointerId2)) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    t(motionEvent);
                }
                z = false;
            }
            a();
            z = false;
        } else {
            float x3 = motionEvent.getX();
            float y3 = motionEvent.getY();
            z = false;
            int pointerId3 = motionEvent.getPointerId(0);
            s(x3, y3, pointerId3);
            View l3 = l((int) x3, (int) y3);
            if (l3 == this.u && this.f2320b == 2) {
                y(l3, pointerId3);
            }
            int i6 = this.i[pointerId3] & this.r;
            if (i6 != 0) {
                this.t.onEdgeTouched(i6, pointerId3);
            }
        }
        if (this.f2320b == 1) {
            return true;
        }
        return z;
    }

    public boolean x(View view, int i, int i2) {
        this.u = view;
        this.f2322d = -1;
        boolean m = m(i, i2, 0, 0);
        if (!m && this.f2320b == 0 && this.u != null) {
            this.u = null;
        }
        return m;
    }

    public boolean y(View view, int i) {
        if (view == this.u && this.f2322d == i) {
            return true;
        }
        if (view == null || !this.t.tryCaptureView(view, i)) {
            return false;
        }
        this.f2322d = i;
        b(view, i);
        return true;
    }
}