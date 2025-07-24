package b.b.h;

import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewParent;
import com.google.android.material.internal.StaticLayoutBuilderCompat;

/* compiled from: ForwardingListener.java */
/* loaded from: classes.dex */
public abstract class h0 implements View.OnTouchListener, View.OnAttachStateChangeListener {

    /* renamed from: b  reason: collision with root package name */
    public final float f845b;

    /* renamed from: c  reason: collision with root package name */
    public final int f846c;

    /* renamed from: d  reason: collision with root package name */
    public final int f847d;

    /* renamed from: e  reason: collision with root package name */
    public final View f848e;

    /* renamed from: f  reason: collision with root package name */
    public Runnable f849f;

    /* renamed from: g  reason: collision with root package name */
    public Runnable f850g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f851h;
    public int i;
    public final int[] j = new int[2];

    /* compiled from: ForwardingListener.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            ViewParent parent = h0.this.f848e.getParent();
            if (parent != null) {
                parent.requestDisallowInterceptTouchEvent(true);
            }
        }
    }

    /* compiled from: ForwardingListener.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            h0 h0Var = h0.this;
            h0Var.a();
            View view = h0Var.f848e;
            if (view.isEnabled() && !view.isLongClickable() && h0Var.c()) {
                view.getParent().requestDisallowInterceptTouchEvent(true);
                long uptimeMillis = SystemClock.uptimeMillis();
                MotionEvent obtain = MotionEvent.obtain(uptimeMillis, uptimeMillis, 3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0);
                view.onTouchEvent(obtain);
                obtain.recycle();
                h0Var.f851h = true;
            }
        }
    }

    public h0(View view) {
        this.f848e = view;
        view.setLongClickable(true);
        view.addOnAttachStateChangeListener(this);
        this.f845b = ViewConfiguration.get(view.getContext()).getScaledTouchSlop();
        int tapTimeout = ViewConfiguration.getTapTimeout();
        this.f846c = tapTimeout;
        this.f847d = (ViewConfiguration.getLongPressTimeout() + tapTimeout) / 2;
    }

    public final void a() {
        Runnable runnable = this.f850g;
        if (runnable != null) {
            this.f848e.removeCallbacks(runnable);
        }
        Runnable runnable2 = this.f849f;
        if (runnable2 != null) {
            this.f848e.removeCallbacks(runnable2);
        }
    }

    public abstract b.b.g.i.p b();

    public abstract boolean c();

    public boolean d() {
        b.b.g.i.p b2 = b();
        if (b2 == null || !b2.a()) {
            return true;
        }
        b2.dismiss();
        return true;
    }

    /* JADX WARN: Code restructure failed: missing block: B:37:0x0087, code lost:
        if (r4 != 3) goto L35;
     */
    /* JADX WARN: Removed duplicated region for block: B:67:0x0113  */
    @Override // android.view.View.OnTouchListener
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouch(View view, MotionEvent motionEvent) {
        boolean z;
        boolean z2;
        boolean z3;
        f0 f0Var;
        int[] iArr;
        int[] iArr2;
        boolean z4 = this.f851h;
        if (z4) {
            View view2 = this.f848e;
            b.b.g.i.p b2 = b();
            if (b2 != null && b2.a() && (f0Var = (f0) b2.h()) != null && f0Var.isShown()) {
                MotionEvent obtainNoHistory = MotionEvent.obtainNoHistory(motionEvent);
                view2.getLocationOnScreen(this.j);
                obtainNoHistory.offsetLocation(iArr[0], iArr[1]);
                f0Var.getLocationOnScreen(this.j);
                obtainNoHistory.offsetLocation(-iArr2[0], -iArr2[1]);
                boolean b3 = f0Var.b(obtainNoHistory, this.i);
                obtainNoHistory.recycle();
                int actionMasked = motionEvent.getActionMasked();
                boolean z5 = (actionMasked == 1 || actionMasked == 3) ? false : true;
                if (b3 && z5) {
                    z3 = true;
                    z2 = (z3 && d()) ? false : true;
                }
            }
            z3 = false;
            if (z3) {
            }
        } else {
            View view3 = this.f848e;
            if (view3.isEnabled()) {
                int actionMasked2 = motionEvent.getActionMasked();
                if (actionMasked2 != 0) {
                    if (actionMasked2 != 1) {
                        if (actionMasked2 == 2) {
                            int findPointerIndex = motionEvent.findPointerIndex(this.i);
                            if (findPointerIndex >= 0) {
                                float x = motionEvent.getX(findPointerIndex);
                                float y = motionEvent.getY(findPointerIndex);
                                float f2 = this.f845b;
                                float f3 = -f2;
                                if (!(x >= f3 && y >= f3 && x < ((float) (view3.getRight() - view3.getLeft())) + f2 && y < ((float) (view3.getBottom() - view3.getTop())) + f2)) {
                                    a();
                                    view3.getParent().requestDisallowInterceptTouchEvent(true);
                                    z = true;
                                    z2 = !z && c();
                                    if (z2) {
                                        long uptimeMillis = SystemClock.uptimeMillis();
                                        MotionEvent obtain = MotionEvent.obtain(uptimeMillis, uptimeMillis, 3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 0);
                                        this.f848e.onTouchEvent(obtain);
                                        obtain.recycle();
                                    }
                                }
                            }
                        }
                    }
                    a();
                } else {
                    this.i = motionEvent.getPointerId(0);
                    if (this.f849f == null) {
                        this.f849f = new a();
                    }
                    view3.postDelayed(this.f849f, this.f846c);
                    if (this.f850g == null) {
                        this.f850g = new b();
                    }
                    view3.postDelayed(this.f850g, this.f847d);
                }
            }
            z = false;
            if (z) {
            }
            if (z2) {
            }
        }
        this.f851h = z2;
        return z2 || z4;
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewAttachedToWindow(View view) {
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewDetachedFromWindow(View view) {
        this.f851h = false;
        this.i = -1;
        Runnable runnable = this.f849f;
        if (runnable != null) {
            this.f848e.removeCallbacks(runnable);
        }
    }
}