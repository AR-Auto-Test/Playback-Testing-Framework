package b.w.b;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.graphics.Canvas;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.StateListDrawable;
import android.view.MotionEvent;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: FastScroller.java */
/* loaded from: classes.dex */
public class l extends RecyclerView.n implements RecyclerView.s {

    /* renamed from: a  reason: collision with root package name */
    public static final int[] f2757a = {16842919};

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f2758b = new int[0];
    public final ValueAnimator B;
    public int C;
    public final Runnable D;
    public final RecyclerView.t E;

    /* renamed from: c  reason: collision with root package name */
    public final int f2759c;

    /* renamed from: d  reason: collision with root package name */
    public final int f2760d;

    /* renamed from: e  reason: collision with root package name */
    public final StateListDrawable f2761e;

    /* renamed from: f  reason: collision with root package name */
    public final Drawable f2762f;

    /* renamed from: g  reason: collision with root package name */
    public final int f2763g;

    /* renamed from: h  reason: collision with root package name */
    public final int f2764h;
    public final StateListDrawable i;
    public final Drawable j;
    public final int k;
    public final int l;
    public int m;
    public int n;
    public float o;
    public int p;
    public int q;
    public float r;
    public RecyclerView u;
    public int s = 0;
    public int t = 0;
    public boolean v = false;
    public boolean w = false;
    public int x = 0;
    public int y = 0;
    public final int[] z = new int[2];
    public final int[] A = new int[2];

    /* compiled from: FastScroller.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            l lVar = l.this;
            int i = lVar.C;
            if (i == 1) {
                lVar.B.cancel();
            } else if (i != 2) {
                return;
            }
            lVar.C = 3;
            ValueAnimator valueAnimator = lVar.B;
            valueAnimator.setFloatValues(((Float) valueAnimator.getAnimatedValue()).floatValue(), StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            lVar.B.setDuration(500);
            lVar.B.start();
        }
    }

    /* compiled from: FastScroller.java */
    /* loaded from: classes.dex */
    public class b extends RecyclerView.t {
        public b() {
        }

        @Override // androidx.recyclerview.widget.RecyclerView.t
        public void onScrolled(RecyclerView recyclerView, int i, int i2) {
            l lVar = l.this;
            int computeHorizontalScrollOffset = recyclerView.computeHorizontalScrollOffset();
            int computeVerticalScrollOffset = recyclerView.computeVerticalScrollOffset();
            int computeVerticalScrollRange = lVar.u.computeVerticalScrollRange();
            int i3 = lVar.t;
            lVar.v = computeVerticalScrollRange - i3 > 0 && i3 >= lVar.f2759c;
            int computeHorizontalScrollRange = lVar.u.computeHorizontalScrollRange();
            int i4 = lVar.s;
            boolean z = computeHorizontalScrollRange - i4 > 0 && i4 >= lVar.f2759c;
            lVar.w = z;
            boolean z2 = lVar.v;
            if (!z2 && !z) {
                if (lVar.x != 0) {
                    lVar.h(0);
                    return;
                }
                return;
            }
            if (z2) {
                float f2 = i3;
                lVar.n = (int) ((((f2 / 2.0f) + computeVerticalScrollOffset) * f2) / computeVerticalScrollRange);
                lVar.m = Math.min(i3, (i3 * i3) / computeVerticalScrollRange);
            }
            if (lVar.w) {
                float f3 = computeHorizontalScrollOffset;
                float f4 = i4;
                lVar.q = (int) ((((f4 / 2.0f) + f3) * f4) / computeHorizontalScrollRange);
                lVar.p = Math.min(i4, (i4 * i4) / computeHorizontalScrollRange);
            }
            int i5 = lVar.x;
            if (i5 == 0 || i5 == 1) {
                lVar.h(1);
            }
        }
    }

    /* compiled from: FastScroller.java */
    /* loaded from: classes.dex */
    public class c extends AnimatorListenerAdapter {

        /* renamed from: a  reason: collision with root package name */
        public boolean f2767a = false;

        public c() {
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationCancel(Animator animator) {
            this.f2767a = true;
        }

        @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
        public void onAnimationEnd(Animator animator) {
            if (this.f2767a) {
                this.f2767a = false;
            } else if (((Float) l.this.B.getAnimatedValue()).floatValue() == StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                l lVar = l.this;
                lVar.C = 0;
                lVar.h(0);
            } else {
                l lVar2 = l.this;
                lVar2.C = 2;
                lVar2.u.invalidate();
            }
        }
    }

    /* compiled from: FastScroller.java */
    /* loaded from: classes.dex */
    public class d implements ValueAnimator.AnimatorUpdateListener {
        public d() {
        }

        @Override // android.animation.ValueAnimator.AnimatorUpdateListener
        public void onAnimationUpdate(ValueAnimator valueAnimator) {
            int floatValue = (int) (((Float) valueAnimator.getAnimatedValue()).floatValue() * 255.0f);
            l.this.f2761e.setAlpha(floatValue);
            l.this.f2762f.setAlpha(floatValue);
            l.this.u.invalidate();
        }
    }

    public l(RecyclerView recyclerView, StateListDrawable stateListDrawable, Drawable drawable, StateListDrawable stateListDrawable2, Drawable drawable2, int i, int i2, int i3) {
        ValueAnimator ofFloat = ValueAnimator.ofFloat(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f);
        this.B = ofFloat;
        this.C = 0;
        this.D = new a();
        b bVar = new b();
        this.E = bVar;
        this.f2761e = stateListDrawable;
        this.f2762f = drawable;
        this.i = stateListDrawable2;
        this.j = drawable2;
        this.f2763g = Math.max(i, stateListDrawable.getIntrinsicWidth());
        this.f2764h = Math.max(i, drawable.getIntrinsicWidth());
        this.k = Math.max(i, stateListDrawable2.getIntrinsicWidth());
        this.l = Math.max(i, drawable2.getIntrinsicWidth());
        this.f2759c = i2;
        this.f2760d = i3;
        stateListDrawable.setAlpha(255);
        drawable.setAlpha(255);
        ofFloat.addListener(new c());
        ofFloat.addUpdateListener(new d());
        RecyclerView recyclerView2 = this.u;
        if (recyclerView2 == recyclerView) {
            return;
        }
        if (recyclerView2 != null) {
            recyclerView2.removeItemDecoration(this);
            this.u.removeOnItemTouchListener(this);
            this.u.removeOnScrollListener(bVar);
            d();
        }
        this.u = recyclerView;
        recyclerView.addItemDecoration(this);
        this.u.addOnItemTouchListener(this);
        this.u.addOnScrollListener(bVar);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public void a(RecyclerView recyclerView, MotionEvent motionEvent) {
        if (this.x == 0) {
            return;
        }
        if (motionEvent.getAction() == 0) {
            boolean f2 = f(motionEvent.getX(), motionEvent.getY());
            boolean e2 = e(motionEvent.getX(), motionEvent.getY());
            if (f2 || e2) {
                if (e2) {
                    this.y = 1;
                    this.r = (int) motionEvent.getX();
                } else if (f2) {
                    this.y = 2;
                    this.o = (int) motionEvent.getY();
                }
                h(2);
            }
        } else if (motionEvent.getAction() == 1 && this.x == 2) {
            this.o = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            this.r = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            h(1);
            this.y = 0;
        } else if (motionEvent.getAction() == 2 && this.x == 2) {
            i();
            if (this.y == 1) {
                float x = motionEvent.getX();
                int[] iArr = this.A;
                int i = this.f2760d;
                iArr[0] = i;
                iArr[1] = this.s - i;
                float max = Math.max(iArr[0], Math.min(iArr[1], x));
                if (Math.abs(this.q - max) >= 2.0f) {
                    int g2 = g(this.r, max, iArr, this.u.computeHorizontalScrollRange(), this.u.computeHorizontalScrollOffset(), this.s);
                    if (g2 != 0) {
                        this.u.scrollBy(g2, 0);
                    }
                    this.r = max;
                }
            }
            if (this.y == 2) {
                float y = motionEvent.getY();
                int[] iArr2 = this.z;
                int i2 = this.f2760d;
                iArr2[0] = i2;
                iArr2[1] = this.t - i2;
                float max2 = Math.max(iArr2[0], Math.min(iArr2[1], y));
                if (Math.abs(this.n - max2) < 2.0f) {
                    return;
                }
                int g3 = g(this.o, max2, iArr2, this.u.computeVerticalScrollRange(), this.u.computeVerticalScrollOffset(), this.t);
                if (g3 != 0) {
                    this.u.scrollBy(0, g3);
                }
                this.o = max2;
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public boolean b(RecyclerView recyclerView, MotionEvent motionEvent) {
        int i = this.x;
        if (i == 1) {
            boolean f2 = f(motionEvent.getX(), motionEvent.getY());
            boolean e2 = e(motionEvent.getX(), motionEvent.getY());
            if (motionEvent.getAction() == 0 && (f2 || e2)) {
                if (e2) {
                    this.y = 1;
                    this.r = (int) motionEvent.getX();
                } else if (f2) {
                    this.y = 2;
                    this.o = (int) motionEvent.getY();
                }
                h(2);
                return true;
            }
        } else if (i == 2) {
            return true;
        }
        return false;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.s
    public void c(boolean z) {
    }

    public final void d() {
        this.u.removeCallbacks(this.D);
    }

    public boolean e(float f2, float f3) {
        if (f3 >= this.t - this.k) {
            int i = this.q;
            int i2 = this.p;
            if (f2 >= i - (i2 / 2) && f2 <= (i2 / 2) + i) {
                return true;
            }
        }
        return false;
    }

    public boolean f(float f2, float f3) {
        RecyclerView recyclerView = this.u;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (recyclerView.getLayoutDirection() == 1) {
            if (f2 > this.f2763g / 2) {
                return false;
            }
        } else if (f2 < this.s - this.f2763g) {
            return false;
        }
        int i = this.n;
        int i2 = this.m / 2;
        return f3 >= ((float) (i - i2)) && f3 <= ((float) (i2 + i));
    }

    public final int g(float f2, float f3, int[] iArr, int i, int i2, int i3) {
        int i4 = iArr[1] - iArr[0];
        if (i4 == 0) {
            return 0;
        }
        int i5 = i - i3;
        int i6 = (int) (((f3 - f2) / i4) * i5);
        int i7 = i2 + i6;
        if (i7 >= i5 || i7 < 0) {
            return 0;
        }
        return i6;
    }

    public void h(int i) {
        if (i == 2 && this.x != 2) {
            this.f2761e.setState(f2757a);
            d();
        }
        if (i == 0) {
            this.u.invalidate();
        } else {
            i();
        }
        if (this.x == 2 && i != 2) {
            this.f2761e.setState(f2758b);
            d();
            this.u.postDelayed(this.D, 1200);
        } else if (i == 1) {
            d();
            this.u.postDelayed(this.D, 1500);
        }
        this.x = i;
    }

    public void i() {
        int i = this.C;
        if (i != 0) {
            if (i != 3) {
                return;
            }
            this.B.cancel();
        }
        this.C = 1;
        ValueAnimator valueAnimator = this.B;
        valueAnimator.setFloatValues(((Float) valueAnimator.getAnimatedValue()).floatValue(), 1.0f);
        this.B.setDuration(500L);
        this.B.setStartDelay(0L);
        this.B.start();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.n
    public void onDrawOver(Canvas canvas, RecyclerView recyclerView, RecyclerView.a0 a0Var) {
        if (this.s == this.u.getWidth() && this.t == this.u.getHeight()) {
            if (this.C != 0) {
                if (this.v) {
                    int i = this.s;
                    int i2 = this.f2763g;
                    int i3 = i - i2;
                    int i4 = this.n;
                    int i5 = this.m;
                    int i6 = i4 - (i5 / 2);
                    this.f2761e.setBounds(0, 0, i2, i5);
                    this.f2762f.setBounds(0, 0, this.f2764h, this.t);
                    RecyclerView recyclerView2 = this.u;
                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                    if (recyclerView2.getLayoutDirection() == 1) {
                        this.f2762f.draw(canvas);
                        canvas.translate(this.f2763g, i6);
                        canvas.scale(-1.0f, 1.0f);
                        this.f2761e.draw(canvas);
                        canvas.scale(1.0f, 1.0f);
                        canvas.translate(-this.f2763g, -i6);
                    } else {
                        canvas.translate(i3, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                        this.f2762f.draw(canvas);
                        canvas.translate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, i6);
                        this.f2761e.draw(canvas);
                        canvas.translate(-i3, -i6);
                    }
                }
                if (this.w) {
                    int i7 = this.t;
                    int i8 = this.k;
                    int i9 = i7 - i8;
                    int i10 = this.q;
                    int i11 = this.p;
                    int i12 = i10 - (i11 / 2);
                    this.i.setBounds(0, 0, i11, i8);
                    this.j.setBounds(0, 0, this.s, this.l);
                    canvas.translate(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, i9);
                    this.j.draw(canvas);
                    canvas.translate(i12, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    this.i.draw(canvas);
                    canvas.translate(-i12, -i9);
                    return;
                }
                return;
            }
            return;
        }
        this.s = this.u.getWidth();
        this.t = this.u.getHeight();
        h(0);
    }
}