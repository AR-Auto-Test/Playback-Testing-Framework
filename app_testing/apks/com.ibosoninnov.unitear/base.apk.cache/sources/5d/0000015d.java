package b.b.c;

import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.content.res.TypedArray;
import android.util.TypedValue;
import android.view.ContextThemeWrapper;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.animation.AccelerateInterpolator;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.Interpolator;
import androidx.appcompat.widget.ActionBarContainer;
import androidx.appcompat.widget.ActionBarContextView;
import androidx.appcompat.widget.ActionBarOverlayLayout;
import androidx.appcompat.widget.Toolbar;
import b.b.c.a;
import b.b.g.a;
import b.b.g.i.g;
import b.b.h.d0;
import b.j.j.v;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: WindowDecorActionBar.java */
/* loaded from: classes.dex */
public class u extends b.b.c.a implements ActionBarOverlayLayout.d {

    /* renamed from: a  reason: collision with root package name */
    public static final Interpolator f614a = new AccelerateInterpolator();

    /* renamed from: b  reason: collision with root package name */
    public static final Interpolator f615b = new DecelerateInterpolator();
    public final v A;

    /* renamed from: c  reason: collision with root package name */
    public Context f616c;

    /* renamed from: d  reason: collision with root package name */
    public Context f617d;

    /* renamed from: e  reason: collision with root package name */
    public ActionBarOverlayLayout f618e;

    /* renamed from: f  reason: collision with root package name */
    public ActionBarContainer f619f;

    /* renamed from: g  reason: collision with root package name */
    public d0 f620g;

    /* renamed from: h  reason: collision with root package name */
    public ActionBarContextView f621h;
    public View i;
    public boolean j;
    public d k;
    public b.b.g.a l;
    public a.InterfaceC0007a m;
    public boolean n;
    public ArrayList<a.b> o;
    public boolean p;
    public int q;
    public boolean r;
    public boolean s;
    public boolean t;
    public boolean u;
    public b.b.g.g v;
    public boolean w;
    public boolean x;
    public final b.j.j.t y;
    public final b.j.j.t z;

    /* compiled from: WindowDecorActionBar.java */
    /* loaded from: classes.dex */
    public class a extends b.j.j.u {
        public a() {
        }

        @Override // b.j.j.t
        public void b(View view) {
            View view2;
            u uVar = u.this;
            if (uVar.r && (view2 = uVar.i) != null) {
                view2.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                u.this.f619f.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            }
            u.this.f619f.setVisibility(8);
            u.this.f619f.setTransitioning(false);
            u uVar2 = u.this;
            uVar2.v = null;
            a.InterfaceC0007a interfaceC0007a = uVar2.m;
            if (interfaceC0007a != null) {
                interfaceC0007a.a(uVar2.l);
                uVar2.l = null;
                uVar2.m = null;
            }
            ActionBarOverlayLayout actionBarOverlayLayout = u.this.f618e;
            if (actionBarOverlayLayout != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                actionBarOverlayLayout.requestApplyInsets();
            }
        }
    }

    /* compiled from: WindowDecorActionBar.java */
    /* loaded from: classes.dex */
    public class b extends b.j.j.u {
        public b() {
        }

        @Override // b.j.j.t
        public void b(View view) {
            u uVar = u.this;
            uVar.v = null;
            uVar.f619f.requestLayout();
        }
    }

    /* compiled from: WindowDecorActionBar.java */
    /* loaded from: classes.dex */
    public class c implements v {
        public c() {
        }
    }

    /* compiled from: WindowDecorActionBar.java */
    /* loaded from: classes.dex */
    public class d extends b.b.g.a implements g.a {

        /* renamed from: d  reason: collision with root package name */
        public final Context f625d;

        /* renamed from: e  reason: collision with root package name */
        public final b.b.g.i.g f626e;

        /* renamed from: f  reason: collision with root package name */
        public a.InterfaceC0007a f627f;

        /* renamed from: g  reason: collision with root package name */
        public WeakReference<View> f628g;

        public d(Context context, a.InterfaceC0007a interfaceC0007a) {
            this.f625d = context;
            this.f627f = interfaceC0007a;
            b.b.g.i.g defaultShowAsAction = new b.b.g.i.g(context).setDefaultShowAsAction(1);
            this.f626e = defaultShowAsAction;
            defaultShowAsAction.setCallback(this);
        }

        @Override // b.b.g.a
        public void a() {
            u uVar = u.this;
            if (uVar.k != this) {
                return;
            }
            if (!(!uVar.s)) {
                uVar.l = this;
                uVar.m = this.f627f;
            } else {
                this.f627f.a(this);
            }
            this.f627f = null;
            u.this.d(false);
            ActionBarContextView actionBarContextView = u.this.f621h;
            if (actionBarContextView.l == null) {
                actionBarContextView.h();
            }
            u.this.f620g.p().sendAccessibilityEvent(32);
            u uVar2 = u.this;
            uVar2.f618e.setHideOnContentScrollEnabled(uVar2.x);
            u.this.k = null;
        }

        @Override // b.b.g.a
        public View b() {
            WeakReference<View> weakReference = this.f628g;
            if (weakReference != null) {
                return weakReference.get();
            }
            return null;
        }

        @Override // b.b.g.a
        public Menu c() {
            return this.f626e;
        }

        @Override // b.b.g.a
        public MenuInflater d() {
            return new b.b.g.f(this.f625d);
        }

        @Override // b.b.g.a
        public CharSequence e() {
            return u.this.f621h.getSubtitle();
        }

        @Override // b.b.g.a
        public CharSequence f() {
            return u.this.f621h.getTitle();
        }

        @Override // b.b.g.a
        public void g() {
            if (u.this.k != this) {
                return;
            }
            this.f626e.stopDispatchingItemsChanged();
            try {
                this.f627f.c(this, this.f626e);
            } finally {
                this.f626e.startDispatchingItemsChanged();
            }
        }

        @Override // b.b.g.a
        public boolean h() {
            return u.this.f621h.s;
        }

        @Override // b.b.g.a
        public void i(View view) {
            u.this.f621h.setCustomView(view);
            this.f628g = new WeakReference<>(view);
        }

        @Override // b.b.g.a
        public void j(int i) {
            u.this.f621h.setSubtitle(u.this.f616c.getResources().getString(i));
        }

        @Override // b.b.g.a
        public void k(CharSequence charSequence) {
            u.this.f621h.setSubtitle(charSequence);
        }

        @Override // b.b.g.a
        public void l(int i) {
            u.this.f621h.setTitle(u.this.f616c.getResources().getString(i));
        }

        @Override // b.b.g.a
        public void m(CharSequence charSequence) {
            u.this.f621h.setTitle(charSequence);
        }

        @Override // b.b.g.a
        public void n(boolean z) {
            this.f635c = z;
            u.this.f621h.setTitleOptional(z);
        }

        @Override // b.b.g.i.g.a
        public boolean onMenuItemSelected(b.b.g.i.g gVar, MenuItem menuItem) {
            a.InterfaceC0007a interfaceC0007a = this.f627f;
            if (interfaceC0007a != null) {
                return interfaceC0007a.d(this, menuItem);
            }
            return false;
        }

        @Override // b.b.g.i.g.a
        public void onMenuModeChange(b.b.g.i.g gVar) {
            if (this.f627f == null) {
                return;
            }
            g();
            b.b.h.c cVar = u.this.f621h.f772e;
            if (cVar != null) {
                cVar.f();
            }
        }
    }

    public u(Activity activity, boolean z) {
        new ArrayList();
        this.o = new ArrayList<>();
        this.q = 0;
        this.r = true;
        this.u = true;
        this.y = new a();
        this.z = new b();
        this.A = new c();
        View decorView = activity.getWindow().getDecorView();
        e(decorView);
        if (z) {
            return;
        }
        this.i = decorView.findViewById(16908290);
    }

    @Override // b.b.c.a
    public void a(boolean z) {
        if (z == this.n) {
            return;
        }
        this.n = z;
        int size = this.o.size();
        for (int i = 0; i < size; i++) {
            this.o.get(i).a(z);
        }
    }

    @Override // b.b.c.a
    public Context b() {
        if (this.f617d == null) {
            TypedValue typedValue = new TypedValue();
            this.f616c.getTheme().resolveAttribute(R.attr.actionBarWidgetTheme, typedValue, true);
            int i = typedValue.resourceId;
            if (i != 0) {
                this.f617d = new ContextThemeWrapper(this.f616c, i);
            } else {
                this.f617d = this.f616c;
            }
        }
        return this.f617d;
    }

    @Override // b.b.c.a
    public void c(boolean z) {
        int i = z ? 4 : 0;
        int s = this.f620g.s();
        this.j = true;
        this.f620g.k((i & 4) | ((-5) & s));
    }

    public void d(boolean z) {
        b.j.j.s n;
        b.j.j.s e2;
        if (z) {
            if (!this.t) {
                this.t = true;
                ActionBarOverlayLayout actionBarOverlayLayout = this.f618e;
                if (actionBarOverlayLayout != null) {
                    actionBarOverlayLayout.setShowingForActionMode(true);
                }
                g(false);
            }
        } else if (this.t) {
            this.t = false;
            ActionBarOverlayLayout actionBarOverlayLayout2 = this.f618e;
            if (actionBarOverlayLayout2 != null) {
                actionBarOverlayLayout2.setShowingForActionMode(false);
            }
            g(false);
        }
        ActionBarContainer actionBarContainer = this.f619f;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (!actionBarContainer.isLaidOut()) {
            if (z) {
                this.f620g.o(4);
                this.f621h.setVisibility(0);
                return;
            }
            this.f620g.o(0);
            this.f621h.setVisibility(8);
            return;
        }
        if (z) {
            e2 = this.f620g.n(4, 100L);
            n = this.f621h.e(0, 200L);
        } else {
            n = this.f620g.n(0, 200L);
            e2 = this.f621h.e(8, 100L);
        }
        b.b.g.g gVar = new b.b.g.g();
        gVar.f669a.add(e2);
        View view = e2.f2231a.get();
        long duration = view != null ? view.animate().getDuration() : 0L;
        View view2 = n.f2231a.get();
        if (view2 != null) {
            view2.animate().setStartDelay(duration);
        }
        gVar.f669a.add(n);
        gVar.b();
    }

    public final void e(View view) {
        d0 wrapper;
        ActionBarOverlayLayout actionBarOverlayLayout = (ActionBarOverlayLayout) view.findViewById(R.id.decor_content_parent);
        this.f618e = actionBarOverlayLayout;
        if (actionBarOverlayLayout != null) {
            actionBarOverlayLayout.setActionBarVisibilityCallback(this);
        }
        View findViewById = view.findViewById(R.id.action_bar);
        if (findViewById instanceof d0) {
            wrapper = (d0) findViewById;
        } else if (findViewById instanceof Toolbar) {
            wrapper = ((Toolbar) findViewById).getWrapper();
        } else {
            StringBuilder x = c.b.a.a.a.x("Can't make a decor toolbar out of ");
            x.append(findViewById != null ? findViewById.getClass().getSimpleName() : "null");
            throw new IllegalStateException(x.toString());
        }
        this.f620g = wrapper;
        this.f621h = (ActionBarContextView) view.findViewById(R.id.action_context_bar);
        ActionBarContainer actionBarContainer = (ActionBarContainer) view.findViewById(R.id.action_bar_container);
        this.f619f = actionBarContainer;
        d0 d0Var = this.f620g;
        if (d0Var != null && this.f621h != null && actionBarContainer != null) {
            this.f616c = d0Var.r();
            boolean z = (this.f620g.s() & 4) != 0;
            if (z) {
                this.j = true;
            }
            Context context = this.f616c;
            this.f620g.q((context.getApplicationInfo().targetSdkVersion < 14) || z);
            f(context.getResources().getBoolean(R.bool.abc_action_bar_embed_tabs));
            TypedArray obtainStyledAttributes = this.f616c.obtainStyledAttributes(null, b.b.b.f541a, R.attr.actionBarStyle, 0);
            if (obtainStyledAttributes.getBoolean(14, false)) {
                ActionBarOverlayLayout actionBarOverlayLayout2 = this.f618e;
                if (actionBarOverlayLayout2.j) {
                    this.x = true;
                    actionBarOverlayLayout2.setHideOnContentScrollEnabled(true);
                } else {
                    throw new IllegalStateException("Action bar must be in overlay mode (Window.FEATURE_OVERLAY_ACTION_BAR) to enable hide on content scroll");
                }
            }
            int dimensionPixelSize = obtainStyledAttributes.getDimensionPixelSize(12, 0);
            if (dimensionPixelSize != 0) {
                ActionBarContainer actionBarContainer2 = this.f619f;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                actionBarContainer2.setElevation(dimensionPixelSize);
            }
            obtainStyledAttributes.recycle();
            return;
        }
        throw new IllegalStateException(u.class.getSimpleName() + " can only be used with a compatible window decor layout");
    }

    public final void f(boolean z) {
        this.p = z;
        if (!z) {
            this.f620g.i(null);
            this.f619f.setTabContainer(null);
        } else {
            this.f619f.setTabContainer(null);
            this.f620g.i(null);
        }
        boolean z2 = true;
        boolean z3 = this.f620g.m() == 2;
        this.f620g.v(!this.p && z3);
        ActionBarOverlayLayout actionBarOverlayLayout = this.f618e;
        if (this.p || !z3) {
            z2 = false;
        }
        actionBarOverlayLayout.setHasNonEmbeddedTabs(z2);
    }

    public final void g(boolean z) {
        View view;
        int[] iArr;
        View view2;
        View view3;
        int[] iArr2;
        if (this.t || !this.s) {
            if (this.u) {
                return;
            }
            this.u = true;
            b.b.g.g gVar = this.v;
            if (gVar != null) {
                gVar.a();
            }
            this.f619f.setVisibility(0);
            if (this.q == 0 && (this.w || z)) {
                this.f619f.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                float f2 = -this.f619f.getHeight();
                if (z) {
                    this.f619f.getLocationInWindow(new int[]{0, 0});
                    f2 -= iArr2[1];
                }
                this.f619f.setTranslationY(f2);
                b.b.g.g gVar2 = new b.b.g.g();
                b.j.j.s b2 = b.j.j.q.b(this.f619f);
                b2.g(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                b2.f(this.A);
                if (!gVar2.f673e) {
                    gVar2.f669a.add(b2);
                }
                if (this.r && (view3 = this.i) != null) {
                    view3.setTranslationY(f2);
                    b.j.j.s b3 = b.j.j.q.b(this.i);
                    b3.g(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    if (!gVar2.f673e) {
                        gVar2.f669a.add(b3);
                    }
                }
                Interpolator interpolator = f615b;
                boolean z2 = gVar2.f673e;
                if (!z2) {
                    gVar2.f671c = interpolator;
                }
                if (!z2) {
                    gVar2.f670b = 250L;
                }
                b.j.j.t tVar = this.z;
                if (!z2) {
                    gVar2.f672d = tVar;
                }
                this.v = gVar2;
                gVar2.b();
            } else {
                this.f619f.setAlpha(1.0f);
                this.f619f.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                if (this.r && (view2 = this.i) != null) {
                    view2.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                }
                this.z.b(null);
            }
            ActionBarOverlayLayout actionBarOverlayLayout = this.f618e;
            if (actionBarOverlayLayout != null) {
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                actionBarOverlayLayout.requestApplyInsets();
            }
        } else if (this.u) {
            this.u = false;
            b.b.g.g gVar3 = this.v;
            if (gVar3 != null) {
                gVar3.a();
            }
            if (this.q == 0 && (this.w || z)) {
                this.f619f.setAlpha(1.0f);
                this.f619f.setTransitioning(true);
                b.b.g.g gVar4 = new b.b.g.g();
                float f3 = -this.f619f.getHeight();
                if (z) {
                    this.f619f.getLocationInWindow(new int[]{0, 0});
                    f3 -= iArr[1];
                }
                b.j.j.s b4 = b.j.j.q.b(this.f619f);
                b4.g(f3);
                b4.f(this.A);
                if (!gVar4.f673e) {
                    gVar4.f669a.add(b4);
                }
                if (this.r && (view = this.i) != null) {
                    b.j.j.s b5 = b.j.j.q.b(view);
                    b5.g(f3);
                    if (!gVar4.f673e) {
                        gVar4.f669a.add(b5);
                    }
                }
                Interpolator interpolator2 = f614a;
                boolean z3 = gVar4.f673e;
                if (!z3) {
                    gVar4.f671c = interpolator2;
                }
                if (!z3) {
                    gVar4.f670b = 250L;
                }
                b.j.j.t tVar2 = this.y;
                if (!z3) {
                    gVar4.f672d = tVar2;
                }
                this.v = gVar4;
                gVar4.b();
                return;
            }
            this.y.b(null);
        }
    }

    public u(Dialog dialog) {
        new ArrayList();
        this.o = new ArrayList<>();
        this.q = 0;
        this.r = true;
        this.u = true;
        this.y = new a();
        this.z = new b();
        this.A = new c();
        e(dialog.getWindow().getDecorView());
    }
}