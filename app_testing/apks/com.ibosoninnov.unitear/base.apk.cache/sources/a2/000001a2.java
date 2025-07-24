package b.b.h;

import android.app.Activity;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.Resources;
import android.graphics.Rect;
import android.os.Build;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityManager;
import com.google.firebase.crashlytics.internal.settings.DefaultSettingsSpiCall;
import com.ibosoninnov.unitear.R;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: TooltipCompatHandler.java */
/* loaded from: classes.dex */
public class b1 implements View.OnLongClickListener, View.OnHoverListener, View.OnAttachStateChangeListener {

    /* renamed from: b  reason: collision with root package name */
    public static b1 f800b;

    /* renamed from: c  reason: collision with root package name */
    public static b1 f801c;

    /* renamed from: d  reason: collision with root package name */
    public final View f802d;

    /* renamed from: e  reason: collision with root package name */
    public final CharSequence f803e;

    /* renamed from: f  reason: collision with root package name */
    public final int f804f;

    /* renamed from: g  reason: collision with root package name */
    public final Runnable f805g = new a();

    /* renamed from: h  reason: collision with root package name */
    public final Runnable f806h = new b();
    public int i;
    public int j;
    public c1 k;
    public boolean l;

    /* compiled from: TooltipCompatHandler.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            b1.this.d(false);
        }
    }

    /* compiled from: TooltipCompatHandler.java */
    /* loaded from: classes.dex */
    public class b implements Runnable {
        public b() {
        }

        @Override // java.lang.Runnable
        public void run() {
            b1.this.b();
        }
    }

    public b1(View view, CharSequence charSequence) {
        int scaledTouchSlop;
        this.f802d = view;
        this.f803e = charSequence;
        ViewConfiguration viewConfiguration = ViewConfiguration.get(view.getContext());
        Method method = b.j.j.r.f2230a;
        if (Build.VERSION.SDK_INT >= 28) {
            scaledTouchSlop = viewConfiguration.getScaledHoverSlop();
        } else {
            scaledTouchSlop = viewConfiguration.getScaledTouchSlop() / 2;
        }
        this.f804f = scaledTouchSlop;
        a();
        view.setOnLongClickListener(this);
        view.setOnHoverListener(this);
    }

    public static void c(b1 b1Var) {
        b1 b1Var2 = f800b;
        if (b1Var2 != null) {
            b1Var2.f802d.removeCallbacks(b1Var2.f805g);
        }
        f800b = b1Var;
        if (b1Var != null) {
            b1Var.f802d.postDelayed(b1Var.f805g, ViewConfiguration.getLongPressTimeout());
        }
    }

    public final void a() {
        this.i = Integer.MAX_VALUE;
        this.j = Integer.MAX_VALUE;
    }

    public void b() {
        if (f801c == this) {
            f801c = null;
            c1 c1Var = this.k;
            if (c1Var != null) {
                c1Var.a();
                this.k = null;
                a();
                this.f802d.removeOnAttachStateChangeListener(this);
            } else {
                Log.e("TooltipCompatHandler", "sActiveHandler.mPopup == null");
            }
        }
        if (f800b == this) {
            c(null);
        }
        this.f802d.removeCallbacks(this.f806h);
    }

    public void d(boolean z) {
        int height;
        int i;
        long j;
        int longPressTimeout;
        long j2;
        View view = this.f802d;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (view.isAttachedToWindow()) {
            c(null);
            b1 b1Var = f801c;
            if (b1Var != null) {
                b1Var.b();
            }
            f801c = this;
            this.l = z;
            c1 c1Var = new c1(this.f802d.getContext());
            this.k = c1Var;
            View view2 = this.f802d;
            int i2 = this.i;
            int i3 = this.j;
            boolean z2 = this.l;
            CharSequence charSequence = this.f803e;
            if (c1Var.f816b.getParent() != null) {
                c1Var.a();
            }
            c1Var.f817c.setText(charSequence);
            WindowManager.LayoutParams layoutParams = c1Var.f818d;
            layoutParams.token = view2.getApplicationWindowToken();
            int dimensionPixelOffset = c1Var.f815a.getResources().getDimensionPixelOffset(R.dimen.tooltip_precise_anchor_threshold);
            if (view2.getWidth() < dimensionPixelOffset) {
                i2 = view2.getWidth() / 2;
            }
            if (view2.getHeight() >= dimensionPixelOffset) {
                int dimensionPixelOffset2 = c1Var.f815a.getResources().getDimensionPixelOffset(R.dimen.tooltip_precise_anchor_extra_offset);
                height = i3 + dimensionPixelOffset2;
                i = i3 - dimensionPixelOffset2;
            } else {
                height = view2.getHeight();
                i = 0;
            }
            layoutParams.gravity = 49;
            int dimensionPixelOffset3 = c1Var.f815a.getResources().getDimensionPixelOffset(z2 ? R.dimen.tooltip_y_offset_touch : R.dimen.tooltip_y_offset_non_touch);
            View rootView = view2.getRootView();
            ViewGroup.LayoutParams layoutParams2 = rootView.getLayoutParams();
            if (!(layoutParams2 instanceof WindowManager.LayoutParams) || ((WindowManager.LayoutParams) layoutParams2).type != 2) {
                Context context = view2.getContext();
                while (true) {
                    if (!(context instanceof ContextWrapper)) {
                        break;
                    } else if (context instanceof Activity) {
                        rootView = ((Activity) context).getWindow().getDecorView();
                        break;
                    } else {
                        context = ((ContextWrapper) context).getBaseContext();
                    }
                }
            }
            if (rootView == null) {
                Log.e("TooltipPopup", "Cannot find app view");
            } else {
                rootView.getWindowVisibleDisplayFrame(c1Var.f819e);
                Rect rect = c1Var.f819e;
                if (rect.left < 0 && rect.top < 0) {
                    Resources resources = c1Var.f815a.getResources();
                    int identifier = resources.getIdentifier("status_bar_height", "dimen", DefaultSettingsSpiCall.ANDROID_CLIENT_TYPE);
                    int dimensionPixelSize = identifier != 0 ? resources.getDimensionPixelSize(identifier) : 0;
                    DisplayMetrics displayMetrics = resources.getDisplayMetrics();
                    c1Var.f819e.set(0, dimensionPixelSize, displayMetrics.widthPixels, displayMetrics.heightPixels);
                }
                rootView.getLocationOnScreen(c1Var.f821g);
                view2.getLocationOnScreen(c1Var.f820f);
                int[] iArr = c1Var.f820f;
                int i4 = iArr[0];
                int[] iArr2 = c1Var.f821g;
                iArr[0] = i4 - iArr2[0];
                iArr[1] = iArr[1] - iArr2[1];
                layoutParams.x = (iArr[0] + i2) - (rootView.getWidth() / 2);
                int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, 0);
                c1Var.f816b.measure(makeMeasureSpec, makeMeasureSpec);
                int measuredHeight = c1Var.f816b.getMeasuredHeight();
                int[] iArr3 = c1Var.f820f;
                int i5 = ((iArr3[1] + i) - dimensionPixelOffset3) - measuredHeight;
                int i6 = iArr3[1] + height + dimensionPixelOffset3;
                if (z2) {
                    if (i5 >= 0) {
                        layoutParams.y = i5;
                    } else {
                        layoutParams.y = i6;
                    }
                } else if (measuredHeight + i6 <= c1Var.f819e.height()) {
                    layoutParams.y = i6;
                } else {
                    layoutParams.y = i5;
                }
            }
            ((WindowManager) c1Var.f815a.getSystemService("window")).addView(c1Var.f816b, c1Var.f818d);
            this.f802d.addOnAttachStateChangeListener(this);
            if (this.l) {
                j2 = 2500;
            } else {
                if ((this.f802d.getWindowSystemUiVisibility() & 1) == 1) {
                    j = 3000;
                    longPressTimeout = ViewConfiguration.getLongPressTimeout();
                } else {
                    j = 15000;
                    longPressTimeout = ViewConfiguration.getLongPressTimeout();
                }
                j2 = j - longPressTimeout;
            }
            this.f802d.removeCallbacks(this.f806h);
            this.f802d.postDelayed(this.f806h, j2);
        }
    }

    @Override // android.view.View.OnHoverListener
    public boolean onHover(View view, MotionEvent motionEvent) {
        boolean z;
        if (this.k == null || !this.l) {
            AccessibilityManager accessibilityManager = (AccessibilityManager) this.f802d.getContext().getSystemService("accessibility");
            if (accessibilityManager.isEnabled() && accessibilityManager.isTouchExplorationEnabled()) {
                return false;
            }
            int action = motionEvent.getAction();
            if (action != 7) {
                if (action == 10) {
                    a();
                    b();
                }
            } else if (this.f802d.isEnabled() && this.k == null) {
                int x = (int) motionEvent.getX();
                int y = (int) motionEvent.getY();
                if (Math.abs(x - this.i) > this.f804f || Math.abs(y - this.j) > this.f804f) {
                    this.i = x;
                    this.j = y;
                    z = true;
                } else {
                    z = false;
                }
                if (z) {
                    c(this);
                }
            }
            return false;
        }
        return false;
    }

    @Override // android.view.View.OnLongClickListener
    public boolean onLongClick(View view) {
        this.i = view.getWidth() / 2;
        this.j = view.getHeight() / 2;
        d(true);
        return true;
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewAttachedToWindow(View view) {
    }

    @Override // android.view.View.OnAttachStateChangeListener
    public void onViewDetachedFromWindow(View view) {
        b();
    }
}