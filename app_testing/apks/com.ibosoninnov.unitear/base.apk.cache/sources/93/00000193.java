package b.b.g.i;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Rect;
import android.os.Parcelable;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.AdapterView;
import android.widget.FrameLayout;
import android.widget.ListView;
import android.widget.PopupWindow;
import android.widget.TextView;
import b.b.g.i.m;
import b.b.h.f0;
import b.b.h.m0;
import com.ibosoninnov.unitear.R;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: StandardMenuPopup.java */
/* loaded from: classes.dex */
public final class q extends k implements PopupWindow.OnDismissListener, AdapterView.OnItemClickListener, m, View.OnKeyListener {

    /* renamed from: c  reason: collision with root package name */
    public final Context f760c;

    /* renamed from: d  reason: collision with root package name */
    public final g f761d;

    /* renamed from: e  reason: collision with root package name */
    public final f f762e;

    /* renamed from: f  reason: collision with root package name */
    public final boolean f763f;

    /* renamed from: g  reason: collision with root package name */
    public final int f764g;

    /* renamed from: h  reason: collision with root package name */
    public final int f765h;
    public final int i;
    public final m0 j;
    public PopupWindow.OnDismissListener m;
    public View n;
    public View o;
    public m.a p;
    public ViewTreeObserver q;
    public boolean r;
    public boolean s;
    public int t;
    public boolean v;
    public final ViewTreeObserver.OnGlobalLayoutListener k = new a();
    public final View.OnAttachStateChangeListener l = new b();
    public int u = 0;

    /* compiled from: StandardMenuPopup.java */
    /* loaded from: classes.dex */
    public class a implements ViewTreeObserver.OnGlobalLayoutListener {
        public a() {
        }

        @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
        public void onGlobalLayout() {
            if (q.this.a()) {
                q qVar = q.this;
                if (qVar.j.B) {
                    return;
                }
                View view = qVar.o;
                if (view != null && view.isShown()) {
                    q.this.j.show();
                } else {
                    q.this.dismiss();
                }
            }
        }
    }

    /* compiled from: StandardMenuPopup.java */
    /* loaded from: classes.dex */
    public class b implements View.OnAttachStateChangeListener {
        public b() {
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewAttachedToWindow(View view) {
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewDetachedFromWindow(View view) {
            ViewTreeObserver viewTreeObserver = q.this.q;
            if (viewTreeObserver != null) {
                if (!viewTreeObserver.isAlive()) {
                    q.this.q = view.getViewTreeObserver();
                }
                q qVar = q.this;
                qVar.q.removeGlobalOnLayoutListener(qVar.k);
            }
            view.removeOnAttachStateChangeListener(this);
        }
    }

    public q(Context context, g gVar, View view, int i, int i2, boolean z) {
        this.f760c = context;
        this.f761d = gVar;
        this.f763f = z;
        this.f762e = new f(gVar, LayoutInflater.from(context), z, R.layout.abc_popup_menu_item_layout);
        this.f765h = i;
        this.i = i2;
        Resources resources = context.getResources();
        this.f764g = Math.max(resources.getDisplayMetrics().widthPixels / 2, resources.getDimensionPixelSize(R.dimen.abc_config_prefDialogWidth));
        this.n = view;
        this.j = new m0(context, null, i, i2);
        gVar.addMenuPresenter(this, context);
    }

    @Override // b.b.g.i.p
    public boolean a() {
        return !this.r && this.j.a();
    }

    @Override // b.b.g.i.k
    public void b(g gVar) {
    }

    @Override // b.b.g.i.k
    public void d(View view) {
        this.n = view;
    }

    @Override // b.b.g.i.p
    public void dismiss() {
        if (a()) {
            this.j.dismiss();
        }
    }

    @Override // b.b.g.i.k
    public void e(boolean z) {
        this.f762e.f723d = z;
    }

    @Override // b.b.g.i.k
    public void f(int i) {
        this.u = i;
    }

    @Override // b.b.g.i.m
    public boolean flagActionItems() {
        return false;
    }

    @Override // b.b.g.i.k
    public void g(int i) {
        this.j.i = i;
    }

    @Override // b.b.g.i.p
    public ListView h() {
        return this.j.f876f;
    }

    @Override // b.b.g.i.k
    public void i(PopupWindow.OnDismissListener onDismissListener) {
        this.m = onDismissListener;
    }

    @Override // b.b.g.i.k
    public void j(boolean z) {
        this.v = z;
    }

    @Override // b.b.g.i.k
    public void k(int i) {
        m0 m0Var = this.j;
        m0Var.j = i;
        m0Var.l = true;
    }

    @Override // b.b.g.i.m
    public void onCloseMenu(g gVar, boolean z) {
        if (gVar != this.f761d) {
            return;
        }
        dismiss();
        m.a aVar = this.p;
        if (aVar != null) {
            aVar.onCloseMenu(gVar, z);
        }
    }

    @Override // android.widget.PopupWindow.OnDismissListener
    public void onDismiss() {
        this.r = true;
        this.f761d.close();
        ViewTreeObserver viewTreeObserver = this.q;
        if (viewTreeObserver != null) {
            if (!viewTreeObserver.isAlive()) {
                this.q = this.o.getViewTreeObserver();
            }
            this.q.removeGlobalOnLayoutListener(this.k);
            this.q = null;
        }
        this.o.removeOnAttachStateChangeListener(this.l);
        PopupWindow.OnDismissListener onDismissListener = this.m;
        if (onDismissListener != null) {
            onDismissListener.onDismiss();
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

    /* JADX WARN: Removed duplicated region for block: B:23:0x0070  */
    @Override // b.b.g.i.m
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onSubMenuSelected(r rVar) {
        boolean z;
        if (rVar.hasVisibleItems()) {
            l lVar = new l(this.f760c, rVar, this.o, this.f763f, this.f765h, this.i);
            lVar.d(this.p);
            boolean l = k.l(rVar);
            lVar.f757h = l;
            k kVar = lVar.j;
            if (kVar != null) {
                kVar.e(l);
            }
            lVar.k = this.m;
            this.m = null;
            this.f761d.close(false);
            m0 m0Var = this.j;
            int i = m0Var.i;
            int i2 = !m0Var.l ? 0 : m0Var.j;
            int i3 = this.u;
            View view = this.n;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if ((Gravity.getAbsoluteGravity(i3, view.getLayoutDirection()) & 7) == 5) {
                i += this.n.getWidth();
            }
            if (!lVar.b()) {
                if (lVar.f755f == null) {
                    z = false;
                    if (z) {
                        m.a aVar = this.p;
                        if (aVar != null) {
                            aVar.a(rVar);
                        }
                        return true;
                    }
                } else {
                    lVar.e(i, i2, true, true);
                }
            }
            z = true;
            if (z) {
            }
        }
        return false;
    }

    @Override // b.b.g.i.m
    public void setCallback(m.a aVar) {
        this.p = aVar;
    }

    @Override // b.b.g.i.p
    public void show() {
        View view;
        boolean z = true;
        if (!a()) {
            if (this.r || (view = this.n) == null) {
                z = false;
            } else {
                this.o = view;
                this.j.C.setOnDismissListener(this);
                m0 m0Var = this.j;
                m0Var.t = this;
                m0Var.q(true);
                View view2 = this.o;
                boolean z2 = this.q == null;
                ViewTreeObserver viewTreeObserver = view2.getViewTreeObserver();
                this.q = viewTreeObserver;
                if (z2) {
                    viewTreeObserver.addOnGlobalLayoutListener(this.k);
                }
                view2.addOnAttachStateChangeListener(this.l);
                m0 m0Var2 = this.j;
                m0Var2.s = view2;
                m0Var2.o = this.u;
                if (!this.s) {
                    this.t = k.c(this.f762e, null, this.f760c, this.f764g);
                    this.s = true;
                }
                this.j.p(this.t);
                this.j.C.setInputMethodMode(2);
                m0 m0Var3 = this.j;
                Rect rect = this.f749b;
                Objects.requireNonNull(m0Var3);
                m0Var3.A = rect != null ? new Rect(rect) : null;
                this.j.show();
                f0 f0Var = this.j.f876f;
                f0Var.setOnKeyListener(this);
                if (this.v && this.f761d.getHeaderTitle() != null) {
                    FrameLayout frameLayout = (FrameLayout) LayoutInflater.from(this.f760c).inflate(R.layout.abc_popup_menu_header_item_layout, (ViewGroup) f0Var, false);
                    TextView textView = (TextView) frameLayout.findViewById(16908310);
                    if (textView != null) {
                        textView.setText(this.f761d.getHeaderTitle());
                    }
                    frameLayout.setEnabled(false);
                    f0Var.addHeaderView(frameLayout, null, false);
                }
                this.j.n(this.f762e);
                this.j.show();
            }
        }
        if (!z) {
            throw new IllegalStateException("StandardMenuPopup cannot be used without an anchor");
        }
    }

    @Override // b.b.g.i.m
    public void updateMenuView(boolean z) {
        this.s = false;
        f fVar = this.f762e;
        if (fVar != null) {
            fVar.notifyDataSetChanged();
        }
    }
}