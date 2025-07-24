package b.b.g.i;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Rect;
import android.os.Build;
import android.os.Handler;
import android.os.Parcelable;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.FrameLayout;
import android.widget.HeaderViewListAdapter;
import android.widget.ListAdapter;
import android.widget.ListView;
import android.widget.PopupWindow;
import android.widget.TextView;
import b.b.g.i.m;
import b.b.h.f0;
import b.b.h.l0;
import b.b.h.m0;
import com.ibosoninnov.unitear.R;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: CascadingMenuPopup.java */
/* loaded from: classes.dex */
public final class d extends k implements m, View.OnKeyListener, PopupWindow.OnDismissListener {
    public PopupWindow.OnDismissListener A;
    public boolean B;

    /* renamed from: c  reason: collision with root package name */
    public final Context f697c;

    /* renamed from: d  reason: collision with root package name */
    public final int f698d;

    /* renamed from: e  reason: collision with root package name */
    public final int f699e;

    /* renamed from: f  reason: collision with root package name */
    public final int f700f;

    /* renamed from: g  reason: collision with root package name */
    public final boolean f701g;

    /* renamed from: h  reason: collision with root package name */
    public final Handler f702h;
    public View p;
    public View q;
    public int r;
    public boolean s;
    public boolean t;
    public int u;
    public int v;
    public boolean x;
    public m.a y;
    public ViewTreeObserver z;
    public final List<g> i = new ArrayList();
    public final List<C0008d> j = new ArrayList();
    public final ViewTreeObserver.OnGlobalLayoutListener k = new a();
    public final View.OnAttachStateChangeListener l = new b();
    public final l0 m = new c();
    public int n = 0;
    public int o = 0;
    public boolean w = false;

    /* compiled from: CascadingMenuPopup.java */
    /* loaded from: classes.dex */
    public class a implements ViewTreeObserver.OnGlobalLayoutListener {
        public a() {
        }

        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
        public void onGlobalLayout() {
            if (!d.this.a() || d.this.j.size() <= 0 || d.this.j.get(0).f710a.B) {
                return;
            }
            View view = d.this.q;
            if (view != null && view.isShown()) {
                for (C0008d c0008d : d.this.j) {
                    c0008d.f710a.show();
                }
                return;
            }
            d.this.dismiss();
        }
    }

    /* compiled from: CascadingMenuPopup.java */
    /* loaded from: classes.dex */
    public class b implements View.OnAttachStateChangeListener {
        public b() {
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewAttachedToWindow(View view) {
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewDetachedFromWindow(View view) {
            ViewTreeObserver viewTreeObserver = d.this.z;
            if (viewTreeObserver != null) {
                if (!viewTreeObserver.isAlive()) {
                    d.this.z = view.getViewTreeObserver();
                }
                d dVar = d.this;
                dVar.z.removeGlobalOnLayoutListener(dVar.k);
            }
            view.removeOnAttachStateChangeListener(this);
        }
    }

    /* compiled from: CascadingMenuPopup.java */
    /* loaded from: classes.dex */
    public class c implements l0 {

        /* compiled from: CascadingMenuPopup.java */
        /* loaded from: classes.dex */
        public class a implements Runnable {

            /* renamed from: b  reason: collision with root package name */
            public final /* synthetic */ C0008d f706b;

            /* renamed from: c  reason: collision with root package name */
            public final /* synthetic */ MenuItem f707c;

            /* renamed from: d  reason: collision with root package name */
            public final /* synthetic */ g f708d;

            public a(C0008d c0008d, MenuItem menuItem, g gVar) {
                this.f706b = c0008d;
                this.f707c = menuItem;
                this.f708d = gVar;
            }

            @Override // java.lang.Runnable
            public void run() {
                C0008d c0008d = this.f706b;
                if (c0008d != null) {
                    d.this.B = true;
                    c0008d.f711b.close(false);
                    d.this.B = false;
                }
                if (this.f707c.isEnabled() && this.f707c.hasSubMenu()) {
                    this.f708d.performItemAction(this.f707c, 4);
                }
            }
        }

        public c() {
        }

        @Override // b.b.h.l0
        public void c(g gVar, MenuItem menuItem) {
            d.this.f702h.removeCallbacksAndMessages(null);
            int size = d.this.j.size();
            int i = 0;
            while (true) {
                if (i >= size) {
                    i = -1;
                    break;
                } else if (gVar == d.this.j.get(i).f711b) {
                    break;
                } else {
                    i++;
                }
            }
            if (i == -1) {
                return;
            }
            int i2 = i + 1;
            d.this.f702h.postAtTime(new a(i2 < d.this.j.size() ? d.this.j.get(i2) : null, menuItem, gVar), gVar, SystemClock.uptimeMillis() + 200);
        }

        @Override // b.b.h.l0
        public void f(g gVar, MenuItem menuItem) {
            d.this.f702h.removeCallbacksAndMessages(gVar);
        }
    }

    /* compiled from: CascadingMenuPopup.java */
    /* renamed from: b.b.g.i.d$d  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0008d {

        /* renamed from: a  reason: collision with root package name */
        public final m0 f710a;

        /* renamed from: b  reason: collision with root package name */
        public final g f711b;

        /* renamed from: c  reason: collision with root package name */
        public final int f712c;

        public C0008d(m0 m0Var, g gVar, int i) {
            this.f710a = m0Var;
            this.f711b = gVar;
            this.f712c = i;
        }
    }

    public d(Context context, View view, int i, int i2, boolean z) {
        this.f697c = context;
        this.p = view;
        this.f699e = i;
        this.f700f = i2;
        this.f701g = z;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        this.r = view.getLayoutDirection() != 1 ? 1 : 0;
        Resources resources = context.getResources();
        this.f698d = Math.max(resources.getDisplayMetrics().widthPixels / 2, resources.getDimensionPixelSize(R.dimen.abc_config_prefDialogWidth));
        this.f702h = new Handler();
    }

    @Override // b.b.g.i.p
    public boolean a() {
        return this.j.size() > 0 && this.j.get(0).f710a.a();
    }

    @Override // b.b.g.i.k
    public void b(g gVar) {
        gVar.addMenuPresenter(this, this.f697c);
        if (a()) {
            m(gVar);
        } else {
            this.i.add(gVar);
        }
    }

    @Override // b.b.g.i.k
    public void d(View view) {
        if (this.p != view) {
            this.p = view;
            int i = this.n;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            this.o = Gravity.getAbsoluteGravity(i, view.getLayoutDirection());
        }
    }

    @Override // b.b.g.i.p
    public void dismiss() {
        int size = this.j.size();
        if (size > 0) {
            C0008d[] c0008dArr = (C0008d[]) this.j.toArray(new C0008d[size]);
            for (int i = size - 1; i >= 0; i--) {
                C0008d c0008d = c0008dArr[i];
                if (c0008d.f710a.a()) {
                    c0008d.f710a.dismiss();
                }
            }
        }
    }

    @Override // b.b.g.i.k
    public void e(boolean z) {
        this.w = z;
    }

    @Override // b.b.g.i.k
    public void f(int i) {
        if (this.n != i) {
            this.n = i;
            View view = this.p;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            this.o = Gravity.getAbsoluteGravity(i, view.getLayoutDirection());
        }
    }

    @Override // b.b.g.i.m
    public boolean flagActionItems() {
        return false;
    }

    @Override // b.b.g.i.k
    public void g(int i) {
        this.s = true;
        this.u = i;
    }

    @Override // b.b.g.i.p
    public ListView h() {
        if (this.j.isEmpty()) {
            return null;
        }
        List<C0008d> list = this.j;
        return list.get(list.size() - 1).f710a.f876f;
    }

    @Override // b.b.g.i.k
    public void i(PopupWindow.OnDismissListener onDismissListener) {
        this.A = onDismissListener;
    }

    @Override // b.b.g.i.k
    public void j(boolean z) {
        this.x = z;
    }

    @Override // b.b.g.i.k
    public void k(int i) {
        this.t = true;
        this.v = i;
    }

    /* JADX WARN: Removed duplicated region for block: B:46:0x00ea  */
    /* JADX WARN: Removed duplicated region for block: B:83:0x01b1  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void m(g gVar) {
        C0008d c0008d;
        int i;
        int i2;
        int i3;
        MenuItem menuItem;
        f fVar;
        int i4;
        int firstVisiblePosition;
        LayoutInflater from = LayoutInflater.from(this.f697c);
        f fVar2 = new f(gVar, from, this.f701g, R.layout.abc_cascading_menu_item_layout);
        if (!a() && this.w) {
            fVar2.f723d = true;
        } else if (a()) {
            fVar2.f723d = k.l(gVar);
        }
        View view = null;
        int c2 = k.c(fVar2, null, this.f697c, this.f698d);
        m0 m0Var = new m0(this.f697c, null, this.f699e, this.f700f);
        m0Var.E = this.m;
        m0Var.t = this;
        m0Var.C.setOnDismissListener(this);
        m0Var.s = this.p;
        m0Var.o = this.o;
        m0Var.q(true);
        m0Var.C.setInputMethodMode(2);
        m0Var.n(fVar2);
        m0Var.p(c2);
        m0Var.o = this.o;
        if (this.j.size() > 0) {
            List<C0008d> list = this.j;
            c0008d = list.get(list.size() - 1);
            g gVar2 = c0008d.f711b;
            int size = gVar2.size();
            int i5 = 0;
            while (true) {
                if (i5 >= size) {
                    menuItem = null;
                    break;
                }
                menuItem = gVar2.getItem(i5);
                if (menuItem.hasSubMenu() && gVar == menuItem.getSubMenu()) {
                    break;
                }
                i5++;
            }
            if (menuItem != null) {
                f0 f0Var = c0008d.f710a.f876f;
                ListAdapter adapter = f0Var.getAdapter();
                if (adapter instanceof HeaderViewListAdapter) {
                    HeaderViewListAdapter headerViewListAdapter = (HeaderViewListAdapter) adapter;
                    i4 = headerViewListAdapter.getHeadersCount();
                    fVar = (f) headerViewListAdapter.getWrappedAdapter();
                } else {
                    fVar = (f) adapter;
                    i4 = 0;
                }
                int count = fVar.getCount();
                int i6 = 0;
                while (true) {
                    if (i6 >= count) {
                        i6 = -1;
                        break;
                    } else if (menuItem == fVar.getItem(i6)) {
                        break;
                    } else {
                        i6++;
                    }
                }
                if (i6 != -1 && (firstVisiblePosition = (i6 + i4) - f0Var.getFirstVisiblePosition()) >= 0 && firstVisiblePosition < f0Var.getChildCount()) {
                    view = f0Var.getChildAt(firstVisiblePosition);
                }
            }
            if (view == null) {
                if (Build.VERSION.SDK_INT <= 28) {
                    Method method = m0.D;
                    if (method != null) {
                        try {
                            method.invoke(m0Var.C, Boolean.FALSE);
                        } catch (Exception unused) {
                            Log.i("MenuPopupWindow", "Could not invoke setTouchModal() on PopupWindow. Oh well.");
                        }
                    }
                } else {
                    m0Var.C.setTouchModal(false);
                }
                m0Var.C.setEnterTransition(null);
                List<C0008d> list2 = this.j;
                f0 f0Var2 = list2.get(list2.size() - 1).f710a.f876f;
                int[] iArr = new int[2];
                f0Var2.getLocationOnScreen(iArr);
                Rect rect = new Rect();
                this.q.getWindowVisibleDisplayFrame(rect);
                int i7 = (this.r != 1 ? iArr[0] - c2 >= 0 : (f0Var2.getWidth() + iArr[0]) + c2 > rect.right) ? 0 : 1;
                boolean z = i7 == 1;
                this.r = i7;
                if (Build.VERSION.SDK_INT >= 26) {
                    m0Var.s = view;
                    i2 = 0;
                    i = 0;
                } else {
                    int[] iArr2 = new int[2];
                    this.p.getLocationOnScreen(iArr2);
                    int[] iArr3 = new int[2];
                    view.getLocationOnScreen(iArr3);
                    if ((this.o & 7) == 5) {
                        iArr2[0] = this.p.getWidth() + iArr2[0];
                        iArr3[0] = view.getWidth() + iArr3[0];
                    }
                    i = iArr3[0] - iArr2[0];
                    i2 = iArr3[1] - iArr2[1];
                }
                if ((this.o & 5) == 5) {
                    if (!z) {
                        c2 = view.getWidth();
                        i3 = i - c2;
                    }
                    i3 = i + c2;
                } else {
                    if (z) {
                        c2 = view.getWidth();
                        i3 = i + c2;
                    }
                    i3 = i - c2;
                }
                m0Var.i = i3;
                m0Var.n = true;
                m0Var.m = true;
                m0Var.j(i2);
            } else {
                if (this.s) {
                    m0Var.i = this.u;
                }
                if (this.t) {
                    m0Var.j(this.v);
                }
                Rect rect2 = this.f749b;
                m0Var.A = rect2 != null ? new Rect(rect2) : null;
            }
            this.j.add(new C0008d(m0Var, gVar, this.r));
            m0Var.show();
            f0 f0Var3 = m0Var.f876f;
            f0Var3.setOnKeyListener(this);
            if (c0008d == null || !this.x || gVar.getHeaderTitle() == null) {
                return;
            }
            FrameLayout frameLayout = (FrameLayout) from.inflate(R.layout.abc_popup_menu_header_item_layout, (ViewGroup) f0Var3, false);
            frameLayout.setEnabled(false);
            ((TextView) frameLayout.findViewById(16908310)).setText(gVar.getHeaderTitle());
            f0Var3.addHeaderView(frameLayout, null, false);
            m0Var.show();
            return;
        }
        c0008d = null;
        view = null;
        if (view == null) {
        }
        this.j.add(new C0008d(m0Var, gVar, this.r));
        m0Var.show();
        f0 f0Var32 = m0Var.f876f;
        f0Var32.setOnKeyListener(this);
        if (c0008d == null) {
        }
    }

    @Override // b.b.g.i.m
    public void onCloseMenu(g gVar, boolean z) {
        int size = this.j.size();
        int i = 0;
        while (true) {
            if (i >= size) {
                i = -1;
                break;
            } else if (gVar == this.j.get(i).f711b) {
                break;
            } else {
                i++;
            }
        }
        if (i < 0) {
            return;
        }
        int i2 = i + 1;
        if (i2 < this.j.size()) {
            this.j.get(i2).f711b.close(false);
        }
        C0008d remove = this.j.remove(i);
        remove.f711b.removeMenuPresenter(this);
        if (this.B) {
            remove.f710a.C.setExitTransition(null);
            remove.f710a.C.setAnimationStyle(0);
        }
        remove.f710a.dismiss();
        int size2 = this.j.size();
        if (size2 > 0) {
            this.r = this.j.get(size2 - 1).f712c;
        } else {
            View view = this.p;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            this.r = view.getLayoutDirection() == 1 ? 0 : 1;
        }
        if (size2 != 0) {
            if (z) {
                this.j.get(0).f711b.close(false);
                return;
            }
            return;
        }
        dismiss();
        m.a aVar = this.y;
        if (aVar != null) {
            aVar.onCloseMenu(gVar, true);
        }
        ViewTreeObserver viewTreeObserver = this.z;
        if (viewTreeObserver != null) {
            if (viewTreeObserver.isAlive()) {
                this.z.removeGlobalOnLayoutListener(this.k);
            }
            this.z = null;
        }
        this.q.removeOnAttachStateChangeListener(this.l);
        this.A.onDismiss();
    }

    @Override // android.widget.PopupWindow.OnDismissListener
    public void onDismiss() {
        C0008d c0008d;
        int size = this.j.size();
        int i = 0;
        while (true) {
            if (i >= size) {
                c0008d = null;
                break;
            }
            c0008d = this.j.get(i);
            if (!c0008d.f710a.a()) {
                break;
            }
            i++;
        }
        if (c0008d != null) {
            c0008d.f711b.close(false);
        }
    }

    @Override // android.view.View.OnKeyListener
    public boolean onKey(View view, int i, KeyEvent keyEvent) {
        if (keyEvent.getAction() == 1 && i == 82) {
            dismiss();
            return true;
        }
        return false;
    }

    @Override // b.b.g.i.m
    public void onRestoreInstanceState(Parcelable parcelable) {
    }

    @Override // b.b.g.i.m
    public Parcelable onSaveInstanceState() {
        return null;
    }

    @Override // b.b.g.i.m
    public boolean onSubMenuSelected(r rVar) {
        for (C0008d c0008d : this.j) {
            if (rVar == c0008d.f711b) {
                c0008d.f710a.f876f.requestFocus();
                return true;
            }
        }
        if (rVar.hasVisibleItems()) {
            rVar.addMenuPresenter(this, this.f697c);
            if (a()) {
                m(rVar);
            } else {
                this.i.add(rVar);
            }
            m.a aVar = this.y;
            if (aVar != null) {
                aVar.a(rVar);
            }
            return true;
        }
        return false;
    }

    @Override // b.b.g.i.m
    public void setCallback(m.a aVar) {
        this.y = aVar;
    }

    @Override // b.b.g.i.p
    public void show() {
        if (a()) {
            return;
        }
        for (g gVar : this.i) {
            m(gVar);
        }
        this.i.clear();
        View view = this.p;
        this.q = view;
        if (view != null) {
            boolean z = this.z == null;
            ViewTreeObserver viewTreeObserver = view.getViewTreeObserver();
            this.z = viewTreeObserver;
            if (z) {
                viewTreeObserver.addOnGlobalLayoutListener(this.k);
            }
            this.q.addOnAttachStateChangeListener(this.l);
        }
    }

    @Override // b.b.g.i.m
    public void updateMenuView(boolean z) {
        for (C0008d c0008d : this.j) {
            ListAdapter adapter = c0008d.f710a.f876f.getAdapter();
            if (adapter instanceof HeaderViewListAdapter) {
                adapter = ((HeaderViewListAdapter) adapter).getWrappedAdapter();
            }
            ((f) adapter).notifyDataSetChanged();
        }
    }
}