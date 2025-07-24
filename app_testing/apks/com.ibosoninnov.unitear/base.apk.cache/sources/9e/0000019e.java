package b.b.h;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.Window;
import androidx.appcompat.widget.Toolbar;
import b.b.g.i.m;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;

/* compiled from: ToolbarWidgetWrapper.java */
/* loaded from: classes.dex */
public class a1 implements d0 {

    /* renamed from: a  reason: collision with root package name */
    public Toolbar f787a;

    /* renamed from: b  reason: collision with root package name */
    public int f788b;

    /* renamed from: c  reason: collision with root package name */
    public View f789c;

    /* renamed from: d  reason: collision with root package name */
    public View f790d;

    /* renamed from: e  reason: collision with root package name */
    public Drawable f791e;

    /* renamed from: f  reason: collision with root package name */
    public Drawable f792f;

    /* renamed from: g  reason: collision with root package name */
    public Drawable f793g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f794h;
    public CharSequence i;
    public CharSequence j;
    public CharSequence k;
    public Window.Callback l;
    public boolean m;
    public c n;
    public int o;
    public Drawable p;

    /* compiled from: ToolbarWidgetWrapper.java */
    /* loaded from: classes.dex */
    public class a extends b.j.j.u {

        /* renamed from: a  reason: collision with root package name */
        public boolean f795a = false;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f796b;

        public a(int i) {
            this.f796b = i;
        }

        @Override // b.j.j.u, b.j.j.t
        public void a(View view) {
            this.f795a = true;
        }

        @Override // b.j.j.t
        public void b(View view) {
            if (this.f795a) {
                return;
            }
            a1.this.f787a.setVisibility(this.f796b);
        }

        @Override // b.j.j.u, b.j.j.t
        public void c(View view) {
            a1.this.f787a.setVisibility(0);
        }
    }

    public a1(Toolbar toolbar, boolean z) {
        Drawable drawable;
        this.o = 0;
        this.f787a = toolbar;
        this.i = toolbar.getTitle();
        this.j = toolbar.getSubtitle();
        this.f794h = this.i != null;
        this.f793g = toolbar.getNavigationIcon();
        y0 r = y0.r(toolbar.getContext(), null, b.b.b.f541a, R.attr.actionBarStyle, 0);
        int i = 15;
        this.p = r.g(15);
        if (z) {
            CharSequence o = r.o(27);
            if (!TextUtils.isEmpty(o)) {
                this.f794h = true;
                this.i = o;
                if ((this.f788b & 8) != 0) {
                    this.f787a.setTitle(o);
                }
            }
            CharSequence o2 = r.o(25);
            if (!TextUtils.isEmpty(o2)) {
                this.j = o2;
                if ((this.f788b & 8) != 0) {
                    this.f787a.setSubtitle(o2);
                }
            }
            Drawable g2 = r.g(20);
            if (g2 != null) {
                this.f792f = g2;
                y();
            }
            Drawable g3 = r.g(17);
            if (g3 != null) {
                this.f791e = g3;
                y();
            }
            if (this.f793g == null && (drawable = this.p) != null) {
                this.f793g = drawable;
                x();
            }
            k(r.j(10, 0));
            int m = r.m(9, 0);
            if (m != 0) {
                View inflate = LayoutInflater.from(this.f787a.getContext()).inflate(m, (ViewGroup) this.f787a, false);
                View view = this.f790d;
                if (view != null && (this.f788b & 16) != 0) {
                    this.f787a.removeView(view);
                }
                this.f790d = inflate;
                if (inflate != null && (this.f788b & 16) != 0) {
                    this.f787a.addView(inflate);
                }
                k(this.f788b | 16);
            }
            int l = r.l(13, 0);
            if (l > 0) {
                ViewGroup.LayoutParams layoutParams = this.f787a.getLayoutParams();
                layoutParams.height = l;
                this.f787a.setLayoutParams(layoutParams);
            }
            int e2 = r.e(7, -1);
            int e3 = r.e(3, -1);
            if (e2 >= 0 || e3 >= 0) {
                this.f787a.setContentInsetsRelative(Math.max(e2, 0), Math.max(e3, 0));
            }
            int m2 = r.m(28, 0);
            if (m2 != 0) {
                Toolbar toolbar2 = this.f787a;
                toolbar2.setTitleTextAppearance(toolbar2.getContext(), m2);
            }
            int m3 = r.m(26, 0);
            if (m3 != 0) {
                Toolbar toolbar3 = this.f787a;
                toolbar3.setSubtitleTextAppearance(toolbar3.getContext(), m3);
            }
            int m4 = r.m(22, 0);
            if (m4 != 0) {
                this.f787a.setPopupTheme(m4);
            }
        } else {
            if (this.f787a.getNavigationIcon() != null) {
                this.p = this.f787a.getNavigationIcon();
            } else {
                i = 11;
            }
            this.f788b = i;
        }
        r.f972b.recycle();
        if (R.string.abc_action_bar_up_description != this.o) {
            this.o = R.string.abc_action_bar_up_description;
            if (TextUtils.isEmpty(this.f787a.getNavigationContentDescription())) {
                int i2 = this.o;
                this.k = i2 != 0 ? r().getString(i2) : null;
                w();
            }
        }
        this.k = this.f787a.getNavigationContentDescription();
        this.f787a.setNavigationOnClickListener(new z0(this));
    }

    @Override // b.b.h.d0
    public void a(Menu menu, m.a aVar) {
        if (this.n == null) {
            c cVar = new c(this.f787a.getContext());
            this.n = cVar;
            cVar.j = R.id.action_menu_presenter;
        }
        c cVar2 = this.n;
        cVar2.f691f = aVar;
        this.f787a.setMenu((b.b.g.i.g) menu, cVar2);
    }

    @Override // b.b.h.d0
    public boolean b() {
        return this.f787a.isOverflowMenuShowing();
    }

    @Override // b.b.h.d0
    public void c() {
        this.m = true;
    }

    @Override // b.b.h.d0
    public void collapseActionView() {
        this.f787a.collapseActionView();
    }

    @Override // b.b.h.d0
    public boolean d() {
        return this.f787a.canShowOverflowMenu();
    }

    @Override // b.b.h.d0
    public boolean e() {
        return this.f787a.isOverflowMenuShowPending();
    }

    @Override // b.b.h.d0
    public boolean f() {
        return this.f787a.hideOverflowMenu();
    }

    @Override // b.b.h.d0
    public boolean g() {
        return this.f787a.showOverflowMenu();
    }

    @Override // b.b.h.d0
    public CharSequence getTitle() {
        return this.f787a.getTitle();
    }

    @Override // b.b.h.d0
    public void h() {
        this.f787a.dismissPopupMenus();
    }

    @Override // b.b.h.d0
    public void i(q0 q0Var) {
        View view = this.f789c;
        if (view != null) {
            ViewParent parent = view.getParent();
            Toolbar toolbar = this.f787a;
            if (parent == toolbar) {
                toolbar.removeView(this.f789c);
            }
        }
        this.f789c = null;
    }

    @Override // b.b.h.d0
    public boolean j() {
        return this.f787a.hasExpandedActionView();
    }

    @Override // b.b.h.d0
    public void k(int i) {
        View view;
        int i2 = this.f788b ^ i;
        this.f788b = i;
        if (i2 != 0) {
            if ((i2 & 4) != 0) {
                if ((i & 4) != 0) {
                    w();
                }
                x();
            }
            if ((i2 & 3) != 0) {
                y();
            }
            if ((i2 & 8) != 0) {
                if ((i & 8) != 0) {
                    this.f787a.setTitle(this.i);
                    this.f787a.setSubtitle(this.j);
                } else {
                    this.f787a.setTitle((CharSequence) null);
                    this.f787a.setSubtitle((CharSequence) null);
                }
            }
            if ((i2 & 16) == 0 || (view = this.f790d) == null) {
                return;
            }
            if ((i & 16) != 0) {
                this.f787a.addView(view);
            } else {
                this.f787a.removeView(view);
            }
        }
    }

    @Override // b.b.h.d0
    public void l(int i) {
        this.f792f = i != 0 ? b.b.d.a.a.a(r(), i) : null;
        y();
    }

    @Override // b.b.h.d0
    public int m() {
        return 0;
    }

    @Override // b.b.h.d0
    public b.j.j.s n(int i, long j) {
        b.j.j.s b2 = b.j.j.q.b(this.f787a);
        b2.a(i == 0 ? 1.0f : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        b2.c(j);
        a aVar = new a(i);
        View view = b2.f2231a.get();
        if (view != null) {
            b2.e(view, aVar);
        }
        return b2;
    }

    @Override // b.b.h.d0
    public void o(int i) {
        this.f787a.setVisibility(i);
    }

    @Override // b.b.h.d0
    public ViewGroup p() {
        return this.f787a;
    }

    @Override // b.b.h.d0
    public void q(boolean z) {
    }

    @Override // b.b.h.d0
    public Context r() {
        return this.f787a.getContext();
    }

    @Override // b.b.h.d0
    public int s() {
        return this.f788b;
    }

    @Override // b.b.h.d0
    public void setIcon(int i) {
        this.f791e = i != 0 ? b.b.d.a.a.a(r(), i) : null;
        y();
    }

    @Override // b.b.h.d0
    public void setTitle(CharSequence charSequence) {
        this.f794h = true;
        this.i = charSequence;
        if ((this.f788b & 8) != 0) {
            this.f787a.setTitle(charSequence);
        }
    }

    @Override // b.b.h.d0
    public void setWindowCallback(Window.Callback callback) {
        this.l = callback;
    }

    @Override // b.b.h.d0
    public void setWindowTitle(CharSequence charSequence) {
        if (this.f794h) {
            return;
        }
        this.i = charSequence;
        if ((this.f788b & 8) != 0) {
            this.f787a.setTitle(charSequence);
        }
    }

    @Override // b.b.h.d0
    public void t() {
        Log.i("ToolbarWidgetWrapper", "Progress display unsupported");
    }

    @Override // b.b.h.d0
    public void u() {
        Log.i("ToolbarWidgetWrapper", "Progress display unsupported");
    }

    @Override // b.b.h.d0
    public void v(boolean z) {
        this.f787a.setCollapsible(z);
    }

    public final void w() {
        if ((this.f788b & 4) != 0) {
            if (TextUtils.isEmpty(this.k)) {
                this.f787a.setNavigationContentDescription(this.o);
            } else {
                this.f787a.setNavigationContentDescription(this.k);
            }
        }
    }

    public final void x() {
        if ((this.f788b & 4) != 0) {
            Toolbar toolbar = this.f787a;
            Drawable drawable = this.f793g;
            if (drawable == null) {
                drawable = this.p;
            }
            toolbar.setNavigationIcon(drawable);
            return;
        }
        this.f787a.setNavigationIcon((Drawable) null);
    }

    public final void y() {
        Drawable drawable;
        int i = this.f788b;
        if ((i & 2) == 0) {
            drawable = null;
        } else if ((i & 1) != 0) {
            drawable = this.f792f;
            if (drawable == null) {
                drawable = this.f791e;
            }
        } else {
            drawable = this.f791e;
        }
        this.f787a.setLogo(drawable);
    }

    @Override // b.b.h.d0
    public void setIcon(Drawable drawable) {
        this.f791e = drawable;
        y();
    }
}