package b.b.h;

import android.content.Context;
import android.content.res.TypedArray;
import android.database.DataSetObserver;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Handler;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.AdapterView;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.PopupWindow;
import java.lang.reflect.Method;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ListPopupWindow.java */
/* loaded from: classes.dex */
public class k0 implements b.b.g.i.p {

    /* renamed from: b  reason: collision with root package name */
    public static Method f872b;

    /* renamed from: c  reason: collision with root package name */
    public static Method f873c;
    public Rect A;
    public boolean B;
    public PopupWindow C;

    /* renamed from: d  reason: collision with root package name */
    public Context f874d;

    /* renamed from: e  reason: collision with root package name */
    public ListAdapter f875e;

    /* renamed from: f  reason: collision with root package name */
    public f0 f876f;
    public int i;
    public int j;
    public boolean l;
    public boolean m;
    public boolean n;
    public DataSetObserver r;
    public View s;
    public AdapterView.OnItemClickListener t;
    public final Handler y;

    /* renamed from: g  reason: collision with root package name */
    public int f877g = -2;

    /* renamed from: h  reason: collision with root package name */
    public int f878h = -2;
    public int k = 1002;
    public int o = 0;
    public int p = Integer.MAX_VALUE;
    public int q = 0;
    public final e u = new e();
    public final d v = new d();
    public final c w = new c();
    public final a x = new a();
    public final Rect z = new Rect();

    /* compiled from: ListPopupWindow.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            f0 f0Var = k0.this.f876f;
            if (f0Var != null) {
                f0Var.setListSelectionHidden(true);
                f0Var.requestLayout();
            }
        }
    }

    /* compiled from: ListPopupWindow.java */
    /* loaded from: classes.dex */
    public class b extends DataSetObserver {
        public b() {
        }

        @Override // android.database.DataSetObserver
        public void onChanged() {
            if (k0.this.a()) {
                k0.this.show();
            }
        }

        @Override // android.database.DataSetObserver
        public void onInvalidated() {
            k0.this.dismiss();
        }
    }

    /* compiled from: ListPopupWindow.java */
    /* loaded from: classes.dex */
    public class c implements AbsListView.OnScrollListener {
        public c() {
        }

        @Override // android.widget.AbsListView.OnScrollListener
        public void onScroll(AbsListView absListView, int i, int i2, int i3) {
        }

        @Override // android.widget.AbsListView.OnScrollListener
        public void onScrollStateChanged(AbsListView absListView, int i) {
            if (i == 1) {
                if ((k0.this.C.getInputMethodMode() == 2) || k0.this.C.getContentView() == null) {
                    return;
                }
                k0 k0Var = k0.this;
                k0Var.y.removeCallbacks(k0Var.u);
                k0.this.u.run();
            }
        }
    }

    /* compiled from: ListPopupWindow.java */
    /* loaded from: classes.dex */
    public class d implements View.OnTouchListener {
        public d() {
        }

        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            PopupWindow popupWindow;
            int action = motionEvent.getAction();
            int x = (int) motionEvent.getX();
            int y = (int) motionEvent.getY();
            if (action == 0 && (popupWindow = k0.this.C) != null && popupWindow.isShowing() && x >= 0 && x < k0.this.C.getWidth() && y >= 0 && y < k0.this.C.getHeight()) {
                k0 k0Var = k0.this;
                k0Var.y.postDelayed(k0Var.u, 250L);
                return false;
            } else if (action == 1) {
                k0 k0Var2 = k0.this;
                k0Var2.y.removeCallbacks(k0Var2.u);
                return false;
            } else {
                return false;
            }
        }
    }

    /* compiled from: ListPopupWindow.java */
    /* loaded from: classes.dex */
    public class e implements Runnable {
        public e() {
        }

        @Override // java.lang.Runnable
        public void run() {
            f0 f0Var = k0.this.f876f;
            if (f0Var != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                if (!f0Var.isAttachedToWindow() || k0.this.f876f.getCount() <= k0.this.f876f.getChildCount()) {
                    return;
                }
                int childCount = k0.this.f876f.getChildCount();
                k0 k0Var = k0.this;
                if (childCount <= k0Var.p) {
                    k0Var.C.setInputMethodMode(2);
                    k0.this.show();
                }
            }
        }
    }

    static {
        if (Build.VERSION.SDK_INT <= 28) {
            try {
                f872b = PopupWindow.class.getDeclaredMethod("setClipToScreenEnabled", Boolean.TYPE);
            } catch (NoSuchMethodException unused) {
                Log.i("ListPopupWindow", "Could not find method setClipToScreenEnabled() on PopupWindow. Oh well.");
            }
            try {
                f873c = PopupWindow.class.getDeclaredMethod("setEpicenterBounds", Rect.class);
            } catch (NoSuchMethodException unused2) {
                Log.i("ListPopupWindow", "Could not find method setEpicenterBounds(Rect) on PopupWindow. Oh well.");
            }
        }
    }

    public k0(Context context, AttributeSet attributeSet, int i, int i2) {
        this.f874d = context;
        this.y = new Handler(context.getMainLooper());
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.o, i, i2);
        this.i = obtainStyledAttributes.getDimensionPixelOffset(0, 0);
        int dimensionPixelOffset = obtainStyledAttributes.getDimensionPixelOffset(1, 0);
        this.j = dimensionPixelOffset;
        if (dimensionPixelOffset != 0) {
            this.l = true;
        }
        obtainStyledAttributes.recycle();
        p pVar = new p(context, attributeSet, i, i2);
        this.C = pVar;
        pVar.setInputMethodMode(1);
    }

    @Override // b.b.g.i.p
    public boolean a() {
        return this.C.isShowing();
    }

    public int b() {
        return this.i;
    }

    public void d(int i) {
        this.i = i;
    }

    @Override // b.b.g.i.p
    public void dismiss() {
        this.C.dismiss();
        this.C.setContentView(null);
        this.f876f = null;
        this.y.removeCallbacks(this.u);
    }

    public Drawable g() {
        return this.C.getBackground();
    }

    @Override // b.b.g.i.p
    public ListView h() {
        return this.f876f;
    }

    public void j(int i) {
        this.j = i;
        this.l = true;
    }

    public int m() {
        if (this.l) {
            return this.j;
        }
        return 0;
    }

    public void n(ListAdapter listAdapter) {
        DataSetObserver dataSetObserver = this.r;
        if (dataSetObserver == null) {
            this.r = new b();
        } else {
            ListAdapter listAdapter2 = this.f875e;
            if (listAdapter2 != null) {
                listAdapter2.unregisterDataSetObserver(dataSetObserver);
            }
        }
        this.f875e = listAdapter;
        if (listAdapter != null) {
            listAdapter.registerDataSetObserver(this.r);
        }
        f0 f0Var = this.f876f;
        if (f0Var != null) {
            f0Var.setAdapter(this.f875e);
        }
    }

    public f0 o(Context context, boolean z) {
        return new f0(context, z);
    }

    public void p(int i) {
        Drawable background = this.C.getBackground();
        if (background != null) {
            background.getPadding(this.z);
            Rect rect = this.z;
            this.f878h = rect.left + rect.right + i;
            return;
        }
        this.f878h = i;
    }

    public void q(boolean z) {
        this.B = z;
        this.C.setFocusable(z);
    }

    public void setBackgroundDrawable(Drawable drawable) {
        this.C.setBackgroundDrawable(drawable);
    }

    @Override // b.b.g.i.p
    public void show() {
        int i;
        int makeMeasureSpec;
        int paddingBottom;
        f0 f0Var;
        if (this.f876f == null) {
            f0 o = o(this.f874d, !this.B);
            this.f876f = o;
            o.setAdapter(this.f875e);
            this.f876f.setOnItemClickListener(this.t);
            this.f876f.setFocusable(true);
            this.f876f.setFocusableInTouchMode(true);
            this.f876f.setOnItemSelectedListener(new j0(this));
            this.f876f.setOnScrollListener(this.w);
            this.C.setContentView(this.f876f);
        } else {
            ViewGroup viewGroup = (ViewGroup) this.C.getContentView();
        }
        Drawable background = this.C.getBackground();
        if (background != null) {
            background.getPadding(this.z);
            Rect rect = this.z;
            int i2 = rect.top;
            i = rect.bottom + i2;
            if (!this.l) {
                this.j = -i2;
            }
        } else {
            this.z.setEmpty();
            i = 0;
        }
        int maxAvailableHeight = this.C.getMaxAvailableHeight(this.s, this.j, this.C.getInputMethodMode() == 2);
        if (this.f877g == -1) {
            paddingBottom = maxAvailableHeight + i;
        } else {
            int i3 = this.f878h;
            if (i3 == -2) {
                int i4 = this.f874d.getResources().getDisplayMetrics().widthPixels;
                Rect rect2 = this.z;
                makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i4 - (rect2.left + rect2.right), Integer.MIN_VALUE);
            } else if (i3 != -1) {
                makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i3, 1073741824);
            } else {
                int i5 = this.f874d.getResources().getDisplayMetrics().widthPixels;
                Rect rect3 = this.z;
                makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i5 - (rect3.left + rect3.right), 1073741824);
            }
            int a2 = this.f876f.a(makeMeasureSpec, maxAvailableHeight - 0, -1);
            paddingBottom = a2 + (a2 > 0 ? this.f876f.getPaddingBottom() + this.f876f.getPaddingTop() + i + 0 : 0);
        }
        boolean z = this.C.getInputMethodMode() == 2;
        this.C.setWindowLayoutType(this.k);
        if (this.C.isShowing()) {
            View view = this.s;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if (view.isAttachedToWindow()) {
                int i6 = this.f878h;
                if (i6 == -1) {
                    i6 = -1;
                } else if (i6 == -2) {
                    i6 = this.s.getWidth();
                }
                int i7 = this.f877g;
                if (i7 == -1) {
                    if (!z) {
                        paddingBottom = -1;
                    }
                    if (z) {
                        this.C.setWidth(this.f878h == -1 ? -1 : 0);
                        this.C.setHeight(0);
                    } else {
                        this.C.setWidth(this.f878h == -1 ? -1 : 0);
                        this.C.setHeight(-1);
                    }
                } else if (i7 != -2) {
                    paddingBottom = i7;
                }
                this.C.setOutsideTouchable(true);
                this.C.update(this.s, this.i, this.j, i6 < 0 ? -1 : i6, paddingBottom < 0 ? -1 : paddingBottom);
                return;
            }
            return;
        }
        int i8 = this.f878h;
        if (i8 == -1) {
            i8 = -1;
        } else if (i8 == -2) {
            i8 = this.s.getWidth();
        }
        int i9 = this.f877g;
        if (i9 == -1) {
            paddingBottom = -1;
        } else if (i9 != -2) {
            paddingBottom = i9;
        }
        this.C.setWidth(i8);
        this.C.setHeight(paddingBottom);
        if (Build.VERSION.SDK_INT <= 28) {
            Method method = f872b;
            if (method != null) {
                try {
                    method.invoke(this.C, Boolean.TRUE);
                } catch (Exception unused) {
                    Log.i("ListPopupWindow", "Could not call setClipToScreenEnabled() on PopupWindow. Oh well.");
                }
            }
        } else {
            this.C.setIsClippedToScreen(true);
        }
        this.C.setOutsideTouchable(true);
        this.C.setTouchInterceptor(this.v);
        if (this.n) {
            this.C.setOverlapAnchor(this.m);
        }
        if (Build.VERSION.SDK_INT <= 28) {
            Method method2 = f873c;
            if (method2 != null) {
                try {
                    method2.invoke(this.C, this.A);
                } catch (Exception e2) {
                    Log.e("ListPopupWindow", "Could not invoke setEpicenterBounds on PopupWindow", e2);
                }
            }
        } else {
            this.C.setEpicenterBounds(this.A);
        }
        this.C.showAsDropDown(this.s, this.i, this.j, this.o);
        this.f876f.setSelection(-1);
        if ((!this.B || this.f876f.isInTouchMode()) && (f0Var = this.f876f) != null) {
            f0Var.setListSelectionHidden(true);
            f0Var.requestLayout();
        }
        if (this.B) {
            return;
        }
        this.y.post(this.x);
    }
}