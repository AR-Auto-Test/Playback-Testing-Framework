package b.b.c;

import android.app.Activity;
import android.app.Dialog;
import android.app.UiModeManager;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.location.Location;
import android.location.LocationManager;
import android.media.AudioManager;
import android.os.Build;
import android.os.Bundle;
import android.os.LocaleList;
import android.os.PowerManager;
import android.text.TextUtils;
import android.util.AndroidRuntimeException;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.ActionMode;
import android.view.ContextThemeWrapper;
import android.view.KeyCharacterMap;
import android.view.KeyEvent;
import android.view.KeyboardShortcutGroup;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ListAdapter;
import android.widget.PopupWindow;
import android.widget.TextView;
import androidx.appcompat.view.menu.ExpandedMenuView;
import androidx.appcompat.widget.ActionBarContextView;
import androidx.appcompat.widget.ContentFrameLayout;
import androidx.appcompat.widget.ViewStubCompat;
import b.b.c.t;
import b.b.c.u;
import b.b.g.a;
import b.b.g.e;
import b.b.g.i.e;
import b.b.g.i.g;
import b.b.g.i.m;
import b.b.h.c0;
import b.b.h.d0;
import b.b.h.d1;
import b.b.h.e1;
import b.b.h.n0;
import b.b.h.y0;
import b.j.c.b.f;
import b.j.j.d;
import b.j.j.q;
import b.j.j.w;
import b.t.e;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Calendar;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.ml.DTrees;

/* compiled from: AppCompatDelegateImpl.java */
/* loaded from: classes.dex */
public class k extends b.b.c.j implements g.a, LayoutInflater.Factory2 {

    /* renamed from: d  reason: collision with root package name */
    public static final b.f.h<String, Integer> f567d = new b.f.h<>();

    /* renamed from: e  reason: collision with root package name */
    public static final int[] f568e = {16842836};

    /* renamed from: f  reason: collision with root package name */
    public static final boolean f569f = !"robolectric".equals(Build.FINGERPRINT);

    /* renamed from: g  reason: collision with root package name */
    public static final boolean f570g = true;
    public View A;
    public boolean B;
    public boolean C;
    public boolean D;
    public boolean E;
    public boolean F;
    public boolean G;
    public boolean H;
    public boolean I;
    public i[] J;
    public i K;
    public boolean L;
    public boolean M;
    public boolean N;
    public boolean O;
    public boolean P;
    public int Q;
    public int R;
    public boolean S;
    public boolean T;
    public f U;
    public f V;
    public boolean W;
    public int X;
    public boolean Z;
    public Rect a0;
    public Rect b0;
    public r c0;

    /* renamed from: h  reason: collision with root package name */
    public final Object f571h;
    public final Context i;
    public Window j;
    public d k;
    public final b.b.c.i l;
    public b.b.c.a m;
    public MenuInflater n;
    public CharSequence o;
    public c0 p;
    public b q;
    public j r;
    public b.b.g.a s;
    public ActionBarContextView t;
    public PopupWindow u;
    public Runnable v;
    public boolean x;
    public ViewGroup y;
    public TextView z;
    public b.j.j.s w = null;
    public final Runnable Y = new a();

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            k kVar = k.this;
            if ((kVar.X & 1) != 0) {
                kVar.F(0);
            }
            k kVar2 = k.this;
            if ((kVar2.X & 4096) != 0) {
                kVar2.F(108);
            }
            k kVar3 = k.this;
            kVar3.W = false;
            kVar3.X = 0;
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public final class b implements m.a {
        public b() {
        }

        @Override // b.b.g.i.m.a
        public boolean a(b.b.g.i.g gVar) {
            Window.Callback M = k.this.M();
            if (M != null) {
                M.onMenuOpened(108, gVar);
                return true;
            }
            return true;
        }

        @Override // b.b.g.i.m.a
        public void onCloseMenu(b.b.g.i.g gVar, boolean z) {
            k.this.B(gVar);
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class c implements a.InterfaceC0007a {

        /* renamed from: a  reason: collision with root package name */
        public a.InterfaceC0007a f574a;

        /* compiled from: AppCompatDelegateImpl.java */
        /* loaded from: classes.dex */
        public class a extends b.j.j.u {
            public a() {
            }

            @Override // b.j.j.t
            public void b(View view) {
                k.this.t.setVisibility(8);
                k kVar = k.this;
                PopupWindow popupWindow = kVar.u;
                if (popupWindow != null) {
                    popupWindow.dismiss();
                } else if (kVar.t.getParent() instanceof View) {
                    AtomicInteger atomicInteger = b.j.j.q.f2214a;
                    ((View) k.this.t.getParent()).requestApplyInsets();
                }
                k.this.t.removeAllViews();
                k.this.w.d(null);
                k kVar2 = k.this;
                kVar2.w = null;
                ViewGroup viewGroup = kVar2.y;
                AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                viewGroup.requestApplyInsets();
            }
        }

        public c(a.InterfaceC0007a interfaceC0007a) {
            this.f574a = interfaceC0007a;
        }

        @Override // b.b.g.a.InterfaceC0007a
        public void a(b.b.g.a aVar) {
            this.f574a.a(aVar);
            k kVar = k.this;
            if (kVar.u != null) {
                kVar.j.getDecorView().removeCallbacks(k.this.v);
            }
            k kVar2 = k.this;
            if (kVar2.t != null) {
                kVar2.G();
                k kVar3 = k.this;
                b.j.j.s b2 = b.j.j.q.b(kVar3.t);
                b2.a(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                kVar3.w = b2;
                b.j.j.s sVar = k.this.w;
                a aVar2 = new a();
                View view = sVar.f2231a.get();
                if (view != null) {
                    sVar.e(view, aVar2);
                }
            }
            k kVar4 = k.this;
            b.b.c.i iVar = kVar4.l;
            if (iVar != null) {
                iVar.onSupportActionModeFinished(kVar4.s);
            }
            k kVar5 = k.this;
            kVar5.s = null;
            ViewGroup viewGroup = kVar5.y;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            viewGroup.requestApplyInsets();
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean b(b.b.g.a aVar, Menu menu) {
            return this.f574a.b(aVar, menu);
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean c(b.b.g.a aVar, Menu menu) {
            ViewGroup viewGroup = k.this.y;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            viewGroup.requestApplyInsets();
            return this.f574a.c(aVar, menu);
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean d(b.b.g.a aVar, MenuItem menuItem) {
            return this.f574a.d(aVar, menuItem);
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class d extends b.b.g.h {
        public d(Window.Callback callback) {
            super(callback);
        }

        /* JADX WARN: Removed duplicated region for block: B:37:0x009d  */
        /* JADX WARN: Removed duplicated region for block: B:38:0x00a1  */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public final ActionMode a(ActionMode.Callback callback) {
            b.b.g.a aVar;
            Context context;
            b.b.g.a aVar2;
            b.b.c.i iVar;
            b.b.c.i iVar2;
            e.a aVar3 = new e.a(k.this.i, callback);
            k kVar = k.this;
            Objects.requireNonNull(kVar);
            b.b.g.a aVar4 = kVar.s;
            if (aVar4 != null) {
                aVar4.a();
            }
            c cVar = new c(aVar3);
            kVar.N();
            b.b.c.a aVar5 = kVar.m;
            if (aVar5 != null) {
                u uVar = (u) aVar5;
                u.d dVar = uVar.k;
                if (dVar != null) {
                    dVar.a();
                }
                uVar.f618e.setHideOnContentScrollEnabled(false);
                uVar.f621h.h();
                u.d dVar2 = new u.d(uVar.f621h.getContext(), cVar);
                dVar2.f626e.stopDispatchingItemsChanged();
                try {
                    if (dVar2.f627f.b(dVar2, dVar2.f626e)) {
                        uVar.k = dVar2;
                        dVar2.g();
                        uVar.f621h.f(dVar2);
                        uVar.d(true);
                        uVar.f621h.sendAccessibilityEvent(32);
                    } else {
                        dVar2 = null;
                    }
                    kVar.s = dVar2;
                    if (dVar2 != null && (iVar2 = kVar.l) != null) {
                        iVar2.onSupportActionModeStarted(dVar2);
                    }
                } finally {
                    dVar2.f626e.startDispatchingItemsChanged();
                }
            }
            if (kVar.s == null) {
                kVar.G();
                b.b.g.a aVar6 = kVar.s;
                if (aVar6 != null) {
                    aVar6.a();
                }
                b.b.c.i iVar3 = kVar.l;
                if (iVar3 != null && !kVar.P) {
                    try {
                        aVar = iVar3.onWindowStartingSupportActionMode(cVar);
                    } catch (AbstractMethodError unused) {
                    }
                    if (aVar == null) {
                        kVar.s = aVar;
                    } else {
                        if (kVar.t == null) {
                            if (kVar.G) {
                                TypedValue typedValue = new TypedValue();
                                Resources.Theme theme = kVar.i.getTheme();
                                theme.resolveAttribute(R.attr.actionBarTheme, typedValue, true);
                                if (typedValue.resourceId != 0) {
                                    Resources.Theme newTheme = kVar.i.getResources().newTheme();
                                    newTheme.setTo(theme);
                                    newTheme.applyStyle(typedValue.resourceId, true);
                                    context = new b.b.g.c(kVar.i, 0);
                                    context.getTheme().setTo(newTheme);
                                } else {
                                    context = kVar.i;
                                }
                                kVar.t = new ActionBarContextView(context, null);
                                PopupWindow popupWindow = new PopupWindow(context, (AttributeSet) null, (int) R.attr.actionModePopupWindowStyle);
                                kVar.u = popupWindow;
                                popupWindow.setWindowLayoutType(2);
                                kVar.u.setContentView(kVar.t);
                                kVar.u.setWidth(-1);
                                context.getTheme().resolveAttribute(R.attr.actionBarSize, typedValue, true);
                                kVar.t.setContentHeight(TypedValue.complexToDimensionPixelSize(typedValue.data, context.getResources().getDisplayMetrics()));
                                kVar.u.setHeight(-2);
                                kVar.v = new n(kVar);
                            } else {
                                ViewStubCompat viewStubCompat = (ViewStubCompat) kVar.y.findViewById(R.id.action_mode_bar_stub);
                                if (viewStubCompat != null) {
                                    kVar.N();
                                    b.b.c.a aVar7 = kVar.m;
                                    Context b2 = aVar7 != null ? aVar7.b() : null;
                                    if (b2 == null) {
                                        b2 = kVar.i;
                                    }
                                    viewStubCompat.setLayoutInflater(LayoutInflater.from(b2));
                                    kVar.t = (ActionBarContextView) viewStubCompat.a();
                                }
                            }
                        }
                        if (kVar.t != null) {
                            kVar.G();
                            kVar.t.h();
                            b.b.g.d dVar3 = new b.b.g.d(kVar.t.getContext(), kVar.t, cVar, kVar.u == null);
                            if (cVar.b(dVar3, dVar3.i)) {
                                dVar3.g();
                                kVar.t.f(dVar3);
                                kVar.s = dVar3;
                                if (kVar.T()) {
                                    kVar.t.setAlpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                                    b.j.j.s b3 = b.j.j.q.b(kVar.t);
                                    b3.a(1.0f);
                                    kVar.w = b3;
                                    o oVar = new o(kVar);
                                    View view = b3.f2231a.get();
                                    if (view != null) {
                                        b3.e(view, oVar);
                                    }
                                } else {
                                    kVar.t.setAlpha(1.0f);
                                    kVar.t.setVisibility(0);
                                    kVar.t.sendAccessibilityEvent(32);
                                    if (kVar.t.getParent() instanceof View) {
                                        AtomicInteger atomicInteger = b.j.j.q.f2214a;
                                        ((View) kVar.t.getParent()).requestApplyInsets();
                                    }
                                }
                                if (kVar.u != null) {
                                    kVar.j.getDecorView().post(kVar.v);
                                }
                            } else {
                                kVar.s = null;
                            }
                        }
                    }
                    aVar2 = kVar.s;
                    if (aVar2 != null && (iVar = kVar.l) != null) {
                        iVar.onSupportActionModeStarted(aVar2);
                    }
                    kVar.s = kVar.s;
                }
                aVar = null;
                if (aVar == null) {
                }
                aVar2 = kVar.s;
                if (aVar2 != null) {
                    iVar.onSupportActionModeStarted(aVar2);
                }
                kVar.s = kVar.s;
            }
            b.b.g.a aVar8 = kVar.s;
            if (aVar8 != null) {
                return aVar3.e(aVar8);
            }
            return null;
        }

        @Override // android.view.Window.Callback
        public boolean dispatchKeyEvent(KeyEvent keyEvent) {
            return k.this.E(keyEvent) || this.f678b.dispatchKeyEvent(keyEvent);
        }

        /* JADX WARN: Code restructure failed: missing block: B:17:0x003c, code lost:
            if (r3 != false) goto L14;
         */
        /* JADX WARN: Code restructure failed: missing block: B:29:0x0069, code lost:
            if (r7 != false) goto L14;
         */
        /* JADX WARN: Removed duplicated region for block: B:35:? A[RETURN, SYNTHETIC] */
        @Override // android.view.Window.Callback
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean dispatchKeyShortcutEvent(KeyEvent keyEvent) {
            boolean z;
            boolean z2;
            b.b.g.i.g gVar;
            if (!this.f678b.dispatchKeyShortcutEvent(keyEvent)) {
                k kVar = k.this;
                int keyCode = keyEvent.getKeyCode();
                kVar.N();
                b.b.c.a aVar = kVar.m;
                if (aVar != null) {
                    u.d dVar = ((u) aVar).k;
                    if (dVar == null || (gVar = dVar.f626e) == null) {
                        z2 = false;
                    } else {
                        gVar.setQwertyMode(KeyCharacterMap.load(keyEvent.getDeviceId()).getKeyboardType() != 1);
                        z2 = gVar.performShortcut(keyCode, keyEvent, 0);
                    }
                }
                i iVar = kVar.K;
                if (iVar != null && kVar.R(iVar, keyEvent.getKeyCode(), keyEvent, 1)) {
                    i iVar2 = kVar.K;
                    if (iVar2 != null) {
                        iVar2.l = true;
                    }
                } else {
                    if (kVar.K == null) {
                        i L = kVar.L(0);
                        kVar.S(L, keyEvent);
                        boolean R = kVar.R(L, keyEvent.getKeyCode(), keyEvent, 1);
                        L.k = false;
                    }
                    z = false;
                    if (!z) {
                        return false;
                    }
                }
                z = true;
                if (!z) {
                }
            }
            return true;
        }

        @Override // android.view.Window.Callback
        public void onContentChanged() {
        }

        @Override // android.view.Window.Callback
        public boolean onCreatePanelMenu(int i, Menu menu) {
            if (i != 0 || (menu instanceof b.b.g.i.g)) {
                return this.f678b.onCreatePanelMenu(i, menu);
            }
            return false;
        }

        @Override // android.view.Window.Callback
        public boolean onMenuOpened(int i, Menu menu) {
            this.f678b.onMenuOpened(i, menu);
            k kVar = k.this;
            Objects.requireNonNull(kVar);
            if (i == 108) {
                kVar.N();
                b.b.c.a aVar = kVar.m;
                if (aVar != null) {
                    aVar.a(true);
                }
            }
            return true;
        }

        @Override // android.view.Window.Callback
        public void onPanelClosed(int i, Menu menu) {
            this.f678b.onPanelClosed(i, menu);
            k kVar = k.this;
            Objects.requireNonNull(kVar);
            if (i == 108) {
                kVar.N();
                b.b.c.a aVar = kVar.m;
                if (aVar != null) {
                    aVar.a(false);
                }
            } else if (i == 0) {
                i L = kVar.L(i);
                if (L.m) {
                    kVar.C(L, false);
                }
            }
        }

        @Override // android.view.Window.Callback
        public boolean onPreparePanel(int i, View view, Menu menu) {
            b.b.g.i.g gVar = menu instanceof b.b.g.i.g ? (b.b.g.i.g) menu : null;
            if (i == 0 && gVar == null) {
                return false;
            }
            if (gVar != null) {
                gVar.setOverrideVisibleItems(true);
            }
            boolean onPreparePanel = this.f678b.onPreparePanel(i, view, menu);
            if (gVar != null) {
                gVar.setOverrideVisibleItems(false);
            }
            return onPreparePanel;
        }

        @Override // android.view.Window.Callback
        public void onProvideKeyboardShortcuts(List<KeyboardShortcutGroup> list, Menu menu, int i) {
            b.b.g.i.g gVar = k.this.L(0).f592h;
            if (gVar != null) {
                this.f678b.onProvideKeyboardShortcuts(list, gVar, i);
            } else {
                this.f678b.onProvideKeyboardShortcuts(list, menu, i);
            }
        }

        @Override // android.view.Window.Callback
        public ActionMode onWindowStartingActionMode(ActionMode.Callback callback) {
            return null;
        }

        @Override // android.view.Window.Callback
        public ActionMode onWindowStartingActionMode(ActionMode.Callback callback, int i) {
            Objects.requireNonNull(k.this);
            if (i != 0) {
                return this.f678b.onWindowStartingActionMode(callback, i);
            }
            return a(callback);
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class e extends f {

        /* renamed from: c  reason: collision with root package name */
        public final PowerManager f578c;

        public e(Context context) {
            super();
            this.f578c = (PowerManager) context.getApplicationContext().getSystemService("power");
        }

        @Override // b.b.c.k.f
        public IntentFilter b() {
            IntentFilter intentFilter = new IntentFilter();
            intentFilter.addAction("android.os.action.POWER_SAVE_MODE_CHANGED");
            return intentFilter;
        }

        @Override // b.b.c.k.f
        public int c() {
            return this.f578c.isPowerSaveMode() ? 2 : 1;
        }

        @Override // b.b.c.k.f
        public void d() {
            k.this.x();
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public abstract class f {

        /* renamed from: a  reason: collision with root package name */
        public BroadcastReceiver f580a;

        /* compiled from: AppCompatDelegateImpl.java */
        /* loaded from: classes.dex */
        public class a extends BroadcastReceiver {
            public a() {
            }

            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context, Intent intent) {
                f.this.d();
            }
        }

        public f() {
        }

        public void a() {
            BroadcastReceiver broadcastReceiver = this.f580a;
            if (broadcastReceiver != null) {
                try {
                    k.this.i.unregisterReceiver(broadcastReceiver);
                } catch (IllegalArgumentException unused) {
                }
                this.f580a = null;
            }
        }

        public abstract IntentFilter b();

        public abstract int c();

        public abstract void d();

        public void e() {
            a();
            IntentFilter b2 = b();
            if (b2 == null || b2.countActions() == 0) {
                return;
            }
            if (this.f580a == null) {
                this.f580a = new a();
            }
            k.this.i.registerReceiver(this.f580a, b2);
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class g extends f {

        /* renamed from: c  reason: collision with root package name */
        public final t f583c;

        public g(t tVar) {
            super();
            this.f583c = tVar;
        }

        @Override // b.b.c.k.f
        public IntentFilter b() {
            IntentFilter intentFilter = new IntentFilter();
            intentFilter.addAction("android.intent.action.TIME_SET");
            intentFilter.addAction("android.intent.action.TIMEZONE_CHANGED");
            intentFilter.addAction("android.intent.action.TIME_TICK");
            return intentFilter;
        }

        @Override // b.b.c.k.f
        public int c() {
            boolean z;
            long j;
            t tVar = this.f583c;
            t.a aVar = tVar.f611d;
            if (aVar.f613b > System.currentTimeMillis()) {
                z = aVar.f612a;
            } else {
                Location a2 = b.j.b.d.j(tVar.f609b, "android.permission.ACCESS_COARSE_LOCATION") == 0 ? tVar.a("network") : null;
                Location a3 = b.j.b.d.j(tVar.f609b, "android.permission.ACCESS_FINE_LOCATION") == 0 ? tVar.a("gps") : null;
                if (a3 == null || a2 == null ? a3 != null : a3.getTime() > a2.getTime()) {
                    a2 = a3;
                }
                if (a2 != null) {
                    t.a aVar2 = tVar.f611d;
                    long currentTimeMillis = System.currentTimeMillis();
                    if (s.f604a == null) {
                        s.f604a = new s();
                    }
                    s sVar = s.f604a;
                    sVar.a(currentTimeMillis - 86400000, a2.getLatitude(), a2.getLongitude());
                    sVar.a(currentTimeMillis, a2.getLatitude(), a2.getLongitude());
                    boolean z2 = sVar.f607d == 1;
                    long j2 = sVar.f606c;
                    long j3 = sVar.f605b;
                    sVar.a(currentTimeMillis + 86400000, a2.getLatitude(), a2.getLongitude());
                    long j4 = sVar.f606c;
                    if (j2 == -1 || j3 == -1) {
                        j = currentTimeMillis + 43200000;
                    } else {
                        j = (currentTimeMillis > j3 ? j4 + 0 : currentTimeMillis > j2 ? j3 + 0 : j2 + 0) + 60000;
                    }
                    aVar2.f612a = z2;
                    aVar2.f613b = j;
                    z = aVar.f612a;
                } else {
                    Log.i("TwilightManager", "Could not get last known location. This is probably because the app does not have any location permissions. Falling back to hardcoded sunrise/sunset values.");
                    int i = Calendar.getInstance().get(11);
                    z = i < 6 || i >= 22;
                }
            }
            return z ? 2 : 1;
        }

        @Override // b.b.c.k.f
        public void d() {
            k.this.x();
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public class h extends ContentFrameLayout {
        public h(Context context) {
            super(context, null);
        }

        @Override // android.view.ViewGroup, android.view.View
        public boolean dispatchKeyEvent(KeyEvent keyEvent) {
            return k.this.E(keyEvent) || super.dispatchKeyEvent(keyEvent);
        }

        @Override // android.view.ViewGroup
        public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
            if (motionEvent.getAction() == 0) {
                int x = (int) motionEvent.getX();
                int y = (int) motionEvent.getY();
                if (x < -5 || y < -5 || x > getWidth() + 5 || y > getHeight() + 5) {
                    k kVar = k.this;
                    kVar.C(kVar.L(0), true);
                    return true;
                }
            }
            return super.onInterceptTouchEvent(motionEvent);
        }

        @Override // android.view.View
        public void setBackgroundResource(int i) {
            setBackgroundDrawable(b.b.d.a.a.a(getContext(), i));
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public static final class i {

        /* renamed from: a  reason: collision with root package name */
        public int f585a;

        /* renamed from: b  reason: collision with root package name */
        public int f586b;

        /* renamed from: c  reason: collision with root package name */
        public int f587c;

        /* renamed from: d  reason: collision with root package name */
        public int f588d;

        /* renamed from: e  reason: collision with root package name */
        public ViewGroup f589e;

        /* renamed from: f  reason: collision with root package name */
        public View f590f;

        /* renamed from: g  reason: collision with root package name */
        public View f591g;

        /* renamed from: h  reason: collision with root package name */
        public b.b.g.i.g f592h;
        public b.b.g.i.e i;
        public Context j;
        public boolean k;
        public boolean l;
        public boolean m;
        public boolean n;
        public boolean o = false;
        public boolean p;
        public Bundle q;

        public i(int i) {
            this.f585a = i;
        }

        public void a(b.b.g.i.g gVar) {
            b.b.g.i.e eVar;
            b.b.g.i.g gVar2 = this.f592h;
            if (gVar == gVar2) {
                return;
            }
            if (gVar2 != null) {
                gVar2.removeMenuPresenter(this.i);
            }
            this.f592h = gVar;
            if (gVar == null || (eVar = this.i) == null) {
                return;
            }
            gVar.addMenuPresenter(eVar);
        }
    }

    /* compiled from: AppCompatDelegateImpl.java */
    /* loaded from: classes.dex */
    public final class j implements m.a {
        public j() {
        }

        @Override // b.b.g.i.m.a
        public boolean a(b.b.g.i.g gVar) {
            Window.Callback M;
            if (gVar == gVar.getRootMenu()) {
                k kVar = k.this;
                if (!kVar.D || (M = kVar.M()) == null || k.this.P) {
                    return true;
                }
                M.onMenuOpened(108, gVar);
                return true;
            }
            return true;
        }

        @Override // b.b.g.i.m.a
        public void onCloseMenu(b.b.g.i.g gVar, boolean z) {
            b.b.g.i.g rootMenu = gVar.getRootMenu();
            boolean z2 = rootMenu != gVar;
            k kVar = k.this;
            if (z2) {
                gVar = rootMenu;
            }
            i J = kVar.J(gVar);
            if (J != null) {
                if (z2) {
                    k.this.A(J.f585a, J, rootMenu);
                    k.this.C(J, true);
                    return;
                }
                k.this.C(J, z);
            }
        }
    }

    public k(Context context, Window window, b.b.c.i iVar, Object obj) {
        b.f.h<String, Integer> hVar;
        Integer orDefault;
        b.b.c.h hVar2;
        this.Q = -100;
        this.i = context;
        this.l = iVar;
        this.f571h = obj;
        if (obj instanceof Dialog) {
            while (context != null) {
                if (context instanceof b.b.c.h) {
                    hVar2 = (b.b.c.h) context;
                    break;
                } else if (!(context instanceof ContextWrapper)) {
                    break;
                } else {
                    context = ((ContextWrapper) context).getBaseContext();
                }
            }
            hVar2 = null;
            if (hVar2 != null) {
                this.Q = hVar2.q().d();
            }
        }
        if (this.Q == -100 && (orDefault = (hVar = f567d).getOrDefault(this.f571h.getClass().getName(), null)) != null) {
            this.Q = orDefault.intValue();
            hVar.remove(this.f571h.getClass().getName());
        }
        if (window != null) {
            z(window);
        }
        b.b.h.j.e();
    }

    public void A(int i2, i iVar, Menu menu) {
        if (menu == null && iVar != null) {
            menu = iVar.f592h;
        }
        if ((iVar == null || iVar.m) && !this.P) {
            this.k.f678b.onPanelClosed(i2, menu);
        }
    }

    public void B(b.b.g.i.g gVar) {
        if (this.I) {
            return;
        }
        this.I = true;
        this.p.i();
        Window.Callback M = M();
        if (M != null && !this.P) {
            M.onPanelClosed(108, gVar);
        }
        this.I = false;
    }

    public void C(i iVar, boolean z) {
        ViewGroup viewGroup;
        c0 c0Var;
        if (z && iVar.f585a == 0 && (c0Var = this.p) != null && c0Var.b()) {
            B(iVar.f592h);
            return;
        }
        WindowManager windowManager = (WindowManager) this.i.getSystemService("window");
        if (windowManager != null && iVar.m && (viewGroup = iVar.f589e) != null) {
            windowManager.removeView(viewGroup);
            if (z) {
                A(iVar.f585a, iVar, null);
            }
        }
        iVar.k = false;
        iVar.l = false;
        iVar.m = false;
        iVar.f590f = null;
        iVar.o = true;
        if (this.K == iVar) {
            this.K = null;
        }
    }

    public final Configuration D(Context context, int i2, Configuration configuration) {
        int i3;
        if (i2 != 1) {
            i3 = i2 != 2 ? context.getApplicationContext().getResources().getConfiguration().uiMode & 48 : 32;
        } else {
            i3 = 16;
        }
        Configuration configuration2 = new Configuration();
        configuration2.fontScale = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        if (configuration != null) {
            configuration2.setTo(configuration);
        }
        configuration2.uiMode = i3 | (configuration2.uiMode & (-49));
        return configuration2;
    }

    /* JADX WARN: Code restructure failed: missing block: B:91:0x0123, code lost:
        if (r7 != false) goto L82;
     */
    /* JADX WARN: Removed duplicated region for block: B:104:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean E(KeyEvent keyEvent) {
        View decorView;
        boolean z;
        boolean z2;
        boolean z3;
        boolean z4;
        Object obj = this.f571h;
        if (((obj instanceof d.a) || (obj instanceof p)) && (decorView = this.j.getDecorView()) != null && b.j.j.d.a(decorView, keyEvent)) {
            return true;
        }
        if (keyEvent.getKeyCode() == 82 && this.k.f678b.dispatchKeyEvent(keyEvent)) {
            return true;
        }
        int keyCode = keyEvent.getKeyCode();
        if (keyEvent.getAction() == 0) {
            if (keyCode == 4) {
                this.L = (keyEvent.getFlags() & 128) != 0;
            } else if (keyCode == 82) {
                if (keyEvent.getRepeatCount() == 0) {
                    i L = L(0);
                    if (L.m) {
                        return true;
                    }
                    S(L, keyEvent);
                    return true;
                }
                return true;
            }
        } else if (keyCode == 4) {
            boolean z5 = this.L;
            this.L = false;
            i L2 = L(0);
            if (L2.m) {
                if (z5) {
                    return true;
                }
                C(L2, true);
                return true;
            }
            b.b.g.a aVar = this.s;
            if (aVar != null) {
                aVar.a();
            } else {
                N();
                b.b.c.a aVar2 = this.m;
                if (aVar2 != null) {
                    u uVar = (u) aVar2;
                    d0 d0Var = uVar.f620g;
                    if (d0Var == null || !d0Var.j()) {
                        z2 = false;
                    } else {
                        uVar.f620g.collapseActionView();
                        z2 = true;
                    }
                }
                z = false;
                if (z) {
                    return true;
                }
            }
            z = true;
            if (z) {
            }
        } else if (keyCode == 82) {
            if (this.s != null) {
                return true;
            }
            i L3 = L(0);
            c0 c0Var = this.p;
            if (c0Var != null && c0Var.d() && !ViewConfiguration.get(this.i).hasPermanentMenuKey()) {
                if (!this.p.b()) {
                    if (!this.P && S(L3, keyEvent)) {
                        z3 = this.p.g();
                    }
                    z3 = false;
                } else {
                    z3 = this.p.f();
                }
            } else {
                boolean z6 = L3.m;
                if (!z6 && !L3.l) {
                    if (L3.k) {
                        if (L3.p) {
                            L3.k = false;
                            z4 = S(L3, keyEvent);
                        } else {
                            z4 = true;
                        }
                        if (z4) {
                            Q(L3, keyEvent);
                            z3 = true;
                        }
                    }
                    z3 = false;
                } else {
                    C(L3, true);
                    z3 = z6;
                }
            }
            if (z3) {
                AudioManager audioManager = (AudioManager) this.i.getApplicationContext().getSystemService("audio");
                if (audioManager != null) {
                    audioManager.playSoundEffect(0);
                    return true;
                }
                Log.w("AppCompatDelegate", "Couldn't get audio manager");
                return true;
            }
            return true;
        }
        return false;
    }

    public void F(int i2) {
        i L = L(i2);
        if (L.f592h != null) {
            Bundle bundle = new Bundle();
            L.f592h.saveActionViewStates(bundle);
            if (bundle.size() > 0) {
                L.q = bundle;
            }
            L.f592h.stopDispatchingItemsChanged();
            L.f592h.clear();
        }
        L.p = true;
        L.o = true;
        if ((i2 == 108 || i2 == 0) && this.p != null) {
            i L2 = L(0);
            L2.k = false;
            S(L2, null);
        }
    }

    public void G() {
        b.j.j.s sVar = this.w;
        if (sVar != null) {
            sVar.b();
        }
    }

    public final void H() {
        ViewGroup viewGroup;
        CharSequence charSequence;
        Context context;
        if (this.x) {
            return;
        }
        TypedArray obtainStyledAttributes = this.i.obtainStyledAttributes(b.b.b.j);
        if (obtainStyledAttributes.hasValue(115)) {
            if (obtainStyledAttributes.getBoolean(124, false)) {
                r(1);
            } else if (obtainStyledAttributes.getBoolean(115, false)) {
                r(108);
            }
            if (obtainStyledAttributes.getBoolean(116, false)) {
                r(109);
            }
            if (obtainStyledAttributes.getBoolean(117, false)) {
                r(10);
            }
            this.G = obtainStyledAttributes.getBoolean(0, false);
            obtainStyledAttributes.recycle();
            I();
            this.j.getDecorView();
            LayoutInflater from = LayoutInflater.from(this.i);
            if (!this.H) {
                if (this.G) {
                    viewGroup = (ViewGroup) from.inflate(R.layout.abc_dialog_title_material, (ViewGroup) null);
                    this.E = false;
                    this.D = false;
                } else if (this.D) {
                    TypedValue typedValue = new TypedValue();
                    this.i.getTheme().resolveAttribute(R.attr.actionBarTheme, typedValue, true);
                    if (typedValue.resourceId != 0) {
                        context = new b.b.g.c(this.i, typedValue.resourceId);
                    } else {
                        context = this.i;
                    }
                    viewGroup = (ViewGroup) LayoutInflater.from(context).inflate(R.layout.abc_screen_toolbar, (ViewGroup) null);
                    c0 c0Var = (c0) viewGroup.findViewById(R.id.decor_content_parent);
                    this.p = c0Var;
                    c0Var.setWindowCallback(M());
                    if (this.E) {
                        this.p.h(109);
                    }
                    if (this.B) {
                        this.p.h(2);
                    }
                    if (this.C) {
                        this.p.h(5);
                    }
                } else {
                    viewGroup = null;
                }
            } else {
                viewGroup = this.F ? (ViewGroup) from.inflate(R.layout.abc_screen_simple_overlay_action_mode, (ViewGroup) null) : (ViewGroup) from.inflate(R.layout.abc_screen_simple, (ViewGroup) null);
            }
            if (viewGroup != null) {
                l lVar = new l(this);
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                q.b.c(viewGroup, lVar);
                if (this.p == null) {
                    this.z = (TextView) viewGroup.findViewById(R.id.title);
                }
                Method method = e1.f833a;
                try {
                    Method method2 = viewGroup.getClass().getMethod("makeOptionalFitsSystemWindows", new Class[0]);
                    if (!method2.isAccessible()) {
                        method2.setAccessible(true);
                    }
                    method2.invoke(viewGroup, new Object[0]);
                } catch (IllegalAccessException e2) {
                    Log.d("ViewUtils", "Could not invoke makeOptionalFitsSystemWindows", e2);
                } catch (NoSuchMethodException unused) {
                    Log.d("ViewUtils", "Could not find method makeOptionalFitsSystemWindows. Oh well...");
                } catch (InvocationTargetException e3) {
                    Log.d("ViewUtils", "Could not invoke makeOptionalFitsSystemWindows", e3);
                }
                ContentFrameLayout contentFrameLayout = (ContentFrameLayout) viewGroup.findViewById(R.id.action_bar_activity_content);
                ViewGroup viewGroup2 = (ViewGroup) this.j.findViewById(16908290);
                if (viewGroup2 != null) {
                    while (viewGroup2.getChildCount() > 0) {
                        View childAt = viewGroup2.getChildAt(0);
                        viewGroup2.removeViewAt(0);
                        contentFrameLayout.addView(childAt);
                    }
                    viewGroup2.setId(-1);
                    contentFrameLayout.setId(16908290);
                    if (viewGroup2 instanceof FrameLayout) {
                        ((FrameLayout) viewGroup2).setForeground(null);
                    }
                }
                this.j.setContentView(viewGroup);
                contentFrameLayout.setAttachListener(new m(this));
                this.y = viewGroup;
                Object obj = this.f571h;
                if (obj instanceof Activity) {
                    charSequence = ((Activity) obj).getTitle();
                } else {
                    charSequence = this.o;
                }
                if (!TextUtils.isEmpty(charSequence)) {
                    c0 c0Var2 = this.p;
                    if (c0Var2 != null) {
                        c0Var2.setWindowTitle(charSequence);
                    } else {
                        b.b.c.a aVar = this.m;
                        if (aVar != null) {
                            ((u) aVar).f620g.setWindowTitle(charSequence);
                        } else {
                            TextView textView = this.z;
                            if (textView != null) {
                                textView.setText(charSequence);
                            }
                        }
                    }
                }
                ContentFrameLayout contentFrameLayout2 = (ContentFrameLayout) this.y.findViewById(16908290);
                View decorView = this.j.getDecorView();
                contentFrameLayout2.f136h.set(decorView.getPaddingLeft(), decorView.getPaddingTop(), decorView.getPaddingRight(), decorView.getPaddingBottom());
                AtomicInteger atomicInteger2 = b.j.j.q.f2214a;
                if (contentFrameLayout2.isLaidOut()) {
                    contentFrameLayout2.requestLayout();
                }
                TypedArray obtainStyledAttributes2 = this.i.obtainStyledAttributes(b.b.b.j);
                obtainStyledAttributes2.getValue(122, contentFrameLayout2.getMinWidthMajor());
                obtainStyledAttributes2.getValue(123, contentFrameLayout2.getMinWidthMinor());
                if (obtainStyledAttributes2.hasValue(120)) {
                    obtainStyledAttributes2.getValue(120, contentFrameLayout2.getFixedWidthMajor());
                }
                if (obtainStyledAttributes2.hasValue(121)) {
                    obtainStyledAttributes2.getValue(121, contentFrameLayout2.getFixedWidthMinor());
                }
                if (obtainStyledAttributes2.hasValue(118)) {
                    obtainStyledAttributes2.getValue(118, contentFrameLayout2.getFixedHeightMajor());
                }
                if (obtainStyledAttributes2.hasValue(119)) {
                    obtainStyledAttributes2.getValue(119, contentFrameLayout2.getFixedHeightMinor());
                }
                obtainStyledAttributes2.recycle();
                contentFrameLayout2.requestLayout();
                this.x = true;
                i L = L(0);
                if (this.P || L.f592h != null) {
                    return;
                }
                O(108);
                return;
            }
            StringBuilder x = c.b.a.a.a.x("AppCompat does not support the current theme features: { windowActionBar: ");
            x.append(this.D);
            x.append(", windowActionBarOverlay: ");
            x.append(this.E);
            x.append(", android:windowIsFloating: ");
            x.append(this.G);
            x.append(", windowActionModeOverlay: ");
            x.append(this.F);
            x.append(", windowNoTitle: ");
            x.append(this.H);
            x.append(" }");
            throw new IllegalArgumentException(x.toString());
        }
        obtainStyledAttributes.recycle();
        throw new IllegalStateException("You need to use a Theme.AppCompat theme (or descendant) with this activity.");
    }

    public final void I() {
        if (this.j == null) {
            Object obj = this.f571h;
            if (obj instanceof Activity) {
                z(((Activity) obj).getWindow());
            }
        }
        if (this.j == null) {
            throw new IllegalStateException("We have not been given a Window");
        }
    }

    public i J(Menu menu) {
        i[] iVarArr = this.J;
        int length = iVarArr != null ? iVarArr.length : 0;
        for (int i2 = 0; i2 < length; i2++) {
            i iVar = iVarArr[i2];
            if (iVar != null && iVar.f592h == menu) {
                return iVar;
            }
        }
        return null;
    }

    public final f K(Context context) {
        if (this.U == null) {
            if (t.f608a == null) {
                Context applicationContext = context.getApplicationContext();
                t.f608a = new t(applicationContext, (LocationManager) applicationContext.getSystemService(FirebaseAnalytics.Param.LOCATION));
            }
            this.U = new g(t.f608a);
        }
        return this.U;
    }

    public i L(int i2) {
        i[] iVarArr = this.J;
        if (iVarArr == null || iVarArr.length <= i2) {
            i[] iVarArr2 = new i[i2 + 1];
            if (iVarArr != null) {
                System.arraycopy(iVarArr, 0, iVarArr2, 0, iVarArr.length);
            }
            this.J = iVarArr2;
            iVarArr = iVarArr2;
        }
        i iVar = iVarArr[i2];
        if (iVar == null) {
            i iVar2 = new i(i2);
            iVarArr[i2] = iVar2;
            return iVar2;
        }
        return iVar;
    }

    public final Window.Callback M() {
        return this.j.getCallback();
    }

    public final void N() {
        H();
        if (this.D && this.m == null) {
            Object obj = this.f571h;
            if (obj instanceof Activity) {
                this.m = new u((Activity) this.f571h, this.E);
            } else if (obj instanceof Dialog) {
                this.m = new u((Dialog) this.f571h);
            }
            b.b.c.a aVar = this.m;
            if (aVar != null) {
                boolean z = this.Z;
                u uVar = (u) aVar;
                if (uVar.j) {
                    return;
                }
                uVar.c(z);
            }
        }
    }

    public final void O(int i2) {
        this.X = (1 << i2) | this.X;
        if (this.W) {
            return;
        }
        View decorView = this.j.getDecorView();
        Runnable runnable = this.Y;
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        decorView.postOnAnimation(runnable);
        this.W = true;
    }

    public int P(Context context, int i2) {
        if (i2 != -100) {
            if (i2 != -1) {
                if (i2 == 0) {
                    if (((UiModeManager) context.getApplicationContext().getSystemService(UiModeManager.class)).getNightMode() == 0) {
                        return -1;
                    }
                    return K(context).c();
                } else if (i2 != 1 && i2 != 2) {
                    if (i2 == 3) {
                        if (this.V == null) {
                            this.V = new e(context);
                        }
                        return this.V.c();
                    }
                    throw new IllegalStateException("Unknown value set for night mode. Please use one of the MODE_NIGHT values from AppCompatDelegate.");
                }
            }
            return i2;
        }
        return -1;
    }

    /* JADX WARN: Code restructure failed: missing block: B:78:0x0155, code lost:
        if (r15 != null) goto L56;
     */
    /* JADX WARN: Removed duplicated region for block: B:82:0x015c  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void Q(i iVar, KeyEvent keyEvent) {
        boolean z;
        int i2;
        ViewGroup.LayoutParams layoutParams;
        if (iVar.m || this.P) {
            return;
        }
        if (iVar.f585a == 0) {
            if ((this.i.getResources().getConfiguration().screenLayout & 15) == 4) {
                return;
            }
        }
        Window.Callback M = M();
        if (M != null && !M.onMenuOpened(iVar.f585a, iVar.f592h)) {
            C(iVar, true);
            return;
        }
        WindowManager windowManager = (WindowManager) this.i.getSystemService("window");
        if (windowManager != null && S(iVar, keyEvent)) {
            ViewGroup viewGroup = iVar.f589e;
            if (viewGroup != null && !iVar.o) {
                View view = iVar.f591g;
                if (view != null && (layoutParams = view.getLayoutParams()) != null && layoutParams.width == -1) {
                    i2 = -1;
                    iVar.l = false;
                    WindowManager.LayoutParams layoutParams2 = new WindowManager.LayoutParams(i2, -2, 0, 0, 1002, 8519680, -3);
                    layoutParams2.gravity = iVar.f587c;
                    layoutParams2.windowAnimations = iVar.f588d;
                    windowManager.addView(iVar.f589e, layoutParams2);
                    iVar.m = true;
                }
            } else {
                if (viewGroup == null) {
                    N();
                    b.b.c.a aVar = this.m;
                    Context b2 = aVar != null ? aVar.b() : null;
                    if (b2 == null) {
                        b2 = this.i;
                    }
                    TypedValue typedValue = new TypedValue();
                    Resources.Theme newTheme = b2.getResources().newTheme();
                    newTheme.setTo(b2.getTheme());
                    newTheme.resolveAttribute(R.attr.actionBarPopupTheme, typedValue, true);
                    int i3 = typedValue.resourceId;
                    if (i3 != 0) {
                        newTheme.applyStyle(i3, true);
                    }
                    newTheme.resolveAttribute(R.attr.panelMenuListTheme, typedValue, true);
                    int i4 = typedValue.resourceId;
                    if (i4 != 0) {
                        newTheme.applyStyle(i4, true);
                    } else {
                        newTheme.applyStyle(2131886483, true);
                    }
                    b.b.g.c cVar = new b.b.g.c(b2, 0);
                    cVar.getTheme().setTo(newTheme);
                    iVar.j = cVar;
                    TypedArray obtainStyledAttributes = cVar.obtainStyledAttributes(b.b.b.j);
                    iVar.f586b = obtainStyledAttributes.getResourceId(84, 0);
                    iVar.f588d = obtainStyledAttributes.getResourceId(1, 0);
                    obtainStyledAttributes.recycle();
                    iVar.f589e = new h(iVar.j);
                    iVar.f587c = 81;
                } else if (iVar.o && viewGroup.getChildCount() > 0) {
                    iVar.f589e.removeAllViews();
                }
                View view2 = iVar.f591g;
                if (view2 != null) {
                    iVar.f590f = view2;
                } else {
                    if (iVar.f592h != null) {
                        if (this.r == null) {
                            this.r = new j();
                        }
                        j jVar = this.r;
                        if (iVar.i == null) {
                            b.b.g.i.e eVar = new b.b.g.i.e(iVar.j, R.layout.abc_list_menu_item_layout);
                            iVar.i = eVar;
                            eVar.f717f = jVar;
                            iVar.f592h.addMenuPresenter(eVar);
                        }
                        b.b.g.i.e eVar2 = iVar.i;
                        ViewGroup viewGroup2 = iVar.f589e;
                        if (eVar2.f716e == null) {
                            eVar2.f716e = (ExpandedMenuView) eVar2.f714c.inflate(R.layout.abc_expanded_menu_layout, viewGroup2, false);
                            if (eVar2.f718g == null) {
                                eVar2.f718g = new e.a();
                            }
                            eVar2.f716e.setAdapter((ListAdapter) eVar2.f718g);
                            eVar2.f716e.setOnItemClickListener(eVar2);
                        }
                        ExpandedMenuView expandedMenuView = eVar2.f716e;
                        iVar.f590f = expandedMenuView;
                    }
                    z = false;
                    if (z) {
                        if (iVar.f590f != null && (iVar.f591g != null || ((e.a) iVar.i.a()).getCount() > 0)) {
                            ViewGroup.LayoutParams layoutParams3 = iVar.f590f.getLayoutParams();
                            if (layoutParams3 == null) {
                                layoutParams3 = new ViewGroup.LayoutParams(-2, -2);
                            }
                            iVar.f589e.setBackgroundResource(iVar.f586b);
                            ViewParent parent = iVar.f590f.getParent();
                            if (parent instanceof ViewGroup) {
                                ((ViewGroup) parent).removeView(iVar.f590f);
                            }
                            iVar.f589e.addView(iVar.f590f, layoutParams3);
                            if (!iVar.f590f.hasFocus()) {
                                iVar.f590f.requestFocus();
                            }
                        }
                    }
                    iVar.o = true;
                    return;
                }
                z = true;
                if (z) {
                }
                iVar.o = true;
                return;
            }
            i2 = -2;
            iVar.l = false;
            WindowManager.LayoutParams layoutParams22 = new WindowManager.LayoutParams(i2, -2, 0, 0, 1002, 8519680, -3);
            layoutParams22.gravity = iVar.f587c;
            layoutParams22.windowAnimations = iVar.f588d;
            windowManager.addView(iVar.f589e, layoutParams22);
            iVar.m = true;
        }
    }

    public final boolean R(i iVar, int i2, KeyEvent keyEvent, int i3) {
        b.b.g.i.g gVar;
        boolean z = false;
        if (keyEvent.isSystem()) {
            return false;
        }
        if ((iVar.k || S(iVar, keyEvent)) && (gVar = iVar.f592h) != null) {
            z = gVar.performShortcut(i2, keyEvent, i3);
        }
        if (z && (i3 & 1) == 0 && this.p == null) {
            C(iVar, true);
        }
        return z;
    }

    public final boolean S(i iVar, KeyEvent keyEvent) {
        c0 c0Var;
        c0 c0Var2;
        Resources.Theme theme;
        c0 c0Var3;
        c0 c0Var4;
        if (this.P) {
            return false;
        }
        if (iVar.k) {
            return true;
        }
        i iVar2 = this.K;
        if (iVar2 != null && iVar2 != iVar) {
            C(iVar2, false);
        }
        Window.Callback M = M();
        if (M != null) {
            iVar.f591g = M.onCreatePanelView(iVar.f585a);
        }
        int i2 = iVar.f585a;
        boolean z = i2 == 0 || i2 == 108;
        if (z && (c0Var4 = this.p) != null) {
            c0Var4.c();
        }
        if (iVar.f591g == null) {
            b.b.g.i.g gVar = iVar.f592h;
            if (gVar == null || iVar.p) {
                if (gVar == null) {
                    Context context = this.i;
                    int i3 = iVar.f585a;
                    if ((i3 == 0 || i3 == 108) && this.p != null) {
                        TypedValue typedValue = new TypedValue();
                        Resources.Theme theme2 = context.getTheme();
                        theme2.resolveAttribute(R.attr.actionBarTheme, typedValue, true);
                        if (typedValue.resourceId != 0) {
                            theme = context.getResources().newTheme();
                            theme.setTo(theme2);
                            theme.applyStyle(typedValue.resourceId, true);
                            theme.resolveAttribute(R.attr.actionBarWidgetTheme, typedValue, true);
                        } else {
                            theme2.resolveAttribute(R.attr.actionBarWidgetTheme, typedValue, true);
                            theme = null;
                        }
                        if (typedValue.resourceId != 0) {
                            if (theme == null) {
                                theme = context.getResources().newTheme();
                                theme.setTo(theme2);
                            }
                            theme.applyStyle(typedValue.resourceId, true);
                        }
                        if (theme != null) {
                            b.b.g.c cVar = new b.b.g.c(context, 0);
                            cVar.getTheme().setTo(theme);
                            context = cVar;
                        }
                    }
                    b.b.g.i.g gVar2 = new b.b.g.i.g(context);
                    gVar2.setCallback(this);
                    iVar.a(gVar2);
                    if (iVar.f592h == null) {
                        return false;
                    }
                }
                if (z && (c0Var2 = this.p) != null) {
                    if (this.q == null) {
                        this.q = new b();
                    }
                    c0Var2.a(iVar.f592h, this.q);
                }
                iVar.f592h.stopDispatchingItemsChanged();
                if (!M.onCreatePanelMenu(iVar.f585a, iVar.f592h)) {
                    iVar.a(null);
                    if (z && (c0Var = this.p) != null) {
                        c0Var.a(null, this.q);
                    }
                    return false;
                }
                iVar.p = false;
            }
            iVar.f592h.stopDispatchingItemsChanged();
            Bundle bundle = iVar.q;
            if (bundle != null) {
                iVar.f592h.restoreActionViewStates(bundle);
                iVar.q = null;
            }
            if (!M.onPreparePanel(0, iVar.f591g, iVar.f592h)) {
                if (z && (c0Var3 = this.p) != null) {
                    c0Var3.a(null, this.q);
                }
                iVar.f592h.startDispatchingItemsChanged();
                return false;
            }
            boolean z2 = KeyCharacterMap.load(keyEvent != null ? keyEvent.getDeviceId() : -1).getKeyboardType() != 1;
            iVar.n = z2;
            iVar.f592h.setQwertyMode(z2);
            iVar.f592h.startDispatchingItemsChanged();
        }
        iVar.k = true;
        iVar.l = false;
        this.K = iVar;
        return true;
    }

    public final boolean T() {
        ViewGroup viewGroup;
        if (this.x && (viewGroup = this.y) != null) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if (viewGroup.isLaidOut()) {
                return true;
            }
        }
        return false;
    }

    public final void U() {
        if (this.x) {
            throw new AndroidRuntimeException("Window feature must be requested before adding content");
        }
    }

    public final int V(w wVar, Rect rect) {
        boolean z;
        boolean z2;
        int color;
        int e2 = wVar.e();
        ActionBarContextView actionBarContextView = this.t;
        if (actionBarContextView == null || !(actionBarContextView.getLayoutParams() instanceof ViewGroup.MarginLayoutParams)) {
            z = false;
        } else {
            ViewGroup.MarginLayoutParams marginLayoutParams = (ViewGroup.MarginLayoutParams) this.t.getLayoutParams();
            if (this.t.isShown()) {
                if (this.a0 == null) {
                    this.a0 = new Rect();
                    this.b0 = new Rect();
                }
                Rect rect2 = this.a0;
                Rect rect3 = this.b0;
                rect2.set(wVar.c(), wVar.e(), wVar.d(), wVar.b());
                e1.a(this.y, rect2, rect3);
                int i2 = rect2.top;
                int i3 = rect2.left;
                int i4 = rect2.right;
                ViewGroup viewGroup = this.y;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                w a2 = q.c.a(viewGroup);
                int c2 = a2 == null ? 0 : a2.c();
                int d2 = a2 == null ? 0 : a2.d();
                if (marginLayoutParams.topMargin == i2 && marginLayoutParams.leftMargin == i3 && marginLayoutParams.rightMargin == i4) {
                    z2 = false;
                } else {
                    marginLayoutParams.topMargin = i2;
                    marginLayoutParams.leftMargin = i3;
                    marginLayoutParams.rightMargin = i4;
                    z2 = true;
                }
                if (i2 > 0 && this.A == null) {
                    View view = new View(this.i);
                    this.A = view;
                    view.setVisibility(8);
                    FrameLayout.LayoutParams layoutParams = new FrameLayout.LayoutParams(-1, marginLayoutParams.topMargin, 51);
                    layoutParams.leftMargin = c2;
                    layoutParams.rightMargin = d2;
                    this.y.addView(this.A, -1, layoutParams);
                } else {
                    View view2 = this.A;
                    if (view2 != null) {
                        ViewGroup.MarginLayoutParams marginLayoutParams2 = (ViewGroup.MarginLayoutParams) view2.getLayoutParams();
                        int i5 = marginLayoutParams2.height;
                        int i6 = marginLayoutParams.topMargin;
                        if (i5 != i6 || marginLayoutParams2.leftMargin != c2 || marginLayoutParams2.rightMargin != d2) {
                            marginLayoutParams2.height = i6;
                            marginLayoutParams2.leftMargin = c2;
                            marginLayoutParams2.rightMargin = d2;
                            this.A.setLayoutParams(marginLayoutParams2);
                        }
                    }
                }
                View view3 = this.A;
                z = view3 != null;
                if (z && view3.getVisibility() != 0) {
                    View view4 = this.A;
                    if ((view4.getWindowSystemUiVisibility() & 8192) != 0) {
                        Context context = this.i;
                        Object obj = b.j.c.a.f2074a;
                        color = context.getColor(R.color.abc_decor_view_status_guard_light);
                    } else {
                        Context context2 = this.i;
                        Object obj2 = b.j.c.a.f2074a;
                        color = context2.getColor(R.color.abc_decor_view_status_guard);
                    }
                    view4.setBackgroundColor(color);
                }
                if (!this.F && z) {
                    e2 = 0;
                }
                r4 = z2;
            } else if (marginLayoutParams.topMargin != 0) {
                marginLayoutParams.topMargin = 0;
                z = false;
            } else {
                r4 = false;
                z = false;
            }
            if (r4) {
                this.t.setLayoutParams(marginLayoutParams);
            }
        }
        View view5 = this.A;
        if (view5 != null) {
            view5.setVisibility(z ? 0 : 8);
        }
        return e2;
    }

    @Override // b.b.c.j
    public void a(View view, ViewGroup.LayoutParams layoutParams) {
        H();
        ((ViewGroup) this.y.findViewById(16908290)).addView(view, layoutParams);
        this.k.f678b.onContentChanged();
    }

    /* JADX WARN: Removed duplicated region for block: B:104:0x0181  */
    @Override // b.b.c.j
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public Context b(Context context) {
        Configuration configuration;
        boolean z;
        this.M = true;
        int i2 = this.Q;
        if (i2 == -100) {
            i2 = -100;
        }
        int P = P(context, i2);
        if (f570g && (context instanceof ContextThemeWrapper)) {
            try {
                ((ContextThemeWrapper) context).applyOverrideConfiguration(D(context, P, null));
                return context;
            } catch (IllegalStateException unused) {
            }
        }
        if (context instanceof b.b.g.c) {
            try {
                ((b.b.g.c) context).a(D(context, P, null));
                return context;
            } catch (IllegalStateException unused2) {
            }
        }
        if (f569f) {
            try {
                Configuration configuration2 = context.getPackageManager().getResourcesForApplication(context.getApplicationInfo()).getConfiguration();
                Configuration configuration3 = context.getResources().getConfiguration();
                if (configuration2.equals(configuration3)) {
                    configuration = null;
                } else {
                    configuration = new Configuration();
                    configuration.fontScale = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                    if (configuration3 != null && configuration2.diff(configuration3) != 0) {
                        float f2 = configuration2.fontScale;
                        float f3 = configuration3.fontScale;
                        if (f2 != f3) {
                            configuration.fontScale = f3;
                        }
                        int i3 = configuration2.mcc;
                        int i4 = configuration3.mcc;
                        if (i3 != i4) {
                            configuration.mcc = i4;
                        }
                        int i5 = configuration2.mnc;
                        int i6 = configuration3.mnc;
                        if (i5 != i6) {
                            configuration.mnc = i6;
                        }
                        int i7 = Build.VERSION.SDK_INT;
                        LocaleList locales = configuration2.getLocales();
                        LocaleList locales2 = configuration3.getLocales();
                        if (!locales.equals(locales2)) {
                            configuration.setLocales(locales2);
                            configuration.locale = configuration3.locale;
                        }
                        int i8 = configuration2.touchscreen;
                        int i9 = configuration3.touchscreen;
                        if (i8 != i9) {
                            configuration.touchscreen = i9;
                        }
                        int i10 = configuration2.keyboard;
                        int i11 = configuration3.keyboard;
                        if (i10 != i11) {
                            configuration.keyboard = i11;
                        }
                        int i12 = configuration2.keyboardHidden;
                        int i13 = configuration3.keyboardHidden;
                        if (i12 != i13) {
                            configuration.keyboardHidden = i13;
                        }
                        int i14 = configuration2.navigation;
                        int i15 = configuration3.navigation;
                        if (i14 != i15) {
                            configuration.navigation = i15;
                        }
                        int i16 = configuration2.navigationHidden;
                        int i17 = configuration3.navigationHidden;
                        if (i16 != i17) {
                            configuration.navigationHidden = i17;
                        }
                        int i18 = configuration2.orientation;
                        int i19 = configuration3.orientation;
                        if (i18 != i19) {
                            configuration.orientation = i19;
                        }
                        int i20 = configuration2.screenLayout & 15;
                        int i21 = configuration3.screenLayout & 15;
                        if (i20 != i21) {
                            configuration.screenLayout |= i21;
                        }
                        int i22 = configuration2.screenLayout & 192;
                        int i23 = configuration3.screenLayout & 192;
                        if (i22 != i23) {
                            configuration.screenLayout |= i23;
                        }
                        int i24 = configuration2.screenLayout & 48;
                        int i25 = configuration3.screenLayout & 48;
                        if (i24 != i25) {
                            configuration.screenLayout |= i25;
                        }
                        int i26 = configuration2.screenLayout & DTrees.PREDICT_MASK;
                        int i27 = configuration3.screenLayout & DTrees.PREDICT_MASK;
                        if (i26 != i27) {
                            configuration.screenLayout |= i27;
                        }
                        if (i7 >= 26) {
                            int i28 = configuration2.colorMode & 3;
                            int i29 = configuration3.colorMode & 3;
                            if (i28 != i29) {
                                configuration.colorMode |= i29;
                            }
                            int i30 = configuration2.colorMode & 12;
                            int i31 = configuration3.colorMode & 12;
                            if (i30 != i31) {
                                configuration.colorMode |= i31;
                            }
                        }
                        int i32 = configuration2.uiMode & 15;
                        int i33 = configuration3.uiMode & 15;
                        if (i32 != i33) {
                            configuration.uiMode |= i33;
                        }
                        int i34 = configuration2.uiMode & 48;
                        int i35 = configuration3.uiMode & 48;
                        if (i34 != i35) {
                            configuration.uiMode |= i35;
                        }
                        int i36 = configuration2.screenWidthDp;
                        int i37 = configuration3.screenWidthDp;
                        if (i36 != i37) {
                            configuration.screenWidthDp = i37;
                        }
                        int i38 = configuration2.screenHeightDp;
                        int i39 = configuration3.screenHeightDp;
                        if (i38 != i39) {
                            configuration.screenHeightDp = i39;
                        }
                        int i40 = configuration2.smallestScreenWidthDp;
                        int i41 = configuration3.smallestScreenWidthDp;
                        if (i40 != i41) {
                            configuration.smallestScreenWidthDp = i41;
                        }
                        int i42 = configuration2.densityDpi;
                        int i43 = configuration3.densityDpi;
                        if (i42 != i43) {
                            configuration.densityDpi = i43;
                        }
                    }
                }
                Configuration D = D(context, P, configuration);
                b.b.g.c cVar = new b.b.g.c(context, (int) com.google.android.material.R.style.Theme_AppCompat_Empty);
                cVar.a(D);
                if (context.getTheme() != null) {
                    z = true;
                    if (z) {
                        Resources.Theme theme = cVar.getTheme();
                        if (Build.VERSION.SDK_INT >= 29) {
                            theme.rebase();
                        } else {
                            synchronized (f.a.f2091a) {
                                if (!f.a.f2093c) {
                                    try {
                                        Method declaredMethod = Resources.Theme.class.getDeclaredMethod("rebase", new Class[0]);
                                        f.a.f2092b = declaredMethod;
                                        declaredMethod.setAccessible(true);
                                    } catch (NoSuchMethodException e2) {
                                        Log.i("ResourcesCompat", "Failed to retrieve rebase() method", e2);
                                    }
                                    f.a.f2093c = true;
                                }
                                Method method = f.a.f2092b;
                                if (method != null) {
                                    try {
                                        method.invoke(theme, new Object[0]);
                                    } catch (IllegalAccessException | InvocationTargetException e3) {
                                        Log.i("ResourcesCompat", "Failed to invoke rebase() method via reflection", e3);
                                        f.a.f2092b = null;
                                    }
                                }
                            }
                        }
                    }
                    return cVar;
                }
                z = false;
                if (z) {
                }
                return cVar;
            } catch (PackageManager.NameNotFoundException e4) {
                throw new RuntimeException("Application failed to obtain resources from itself", e4);
            }
        }
        return context;
    }

    @Override // b.b.c.j
    public <T extends View> T c(int i2) {
        H();
        return (T) this.j.findViewById(i2);
    }

    @Override // b.b.c.j
    public int d() {
        return this.Q;
    }

    @Override // b.b.c.j
    public MenuInflater e() {
        if (this.n == null) {
            N();
            b.b.c.a aVar = this.m;
            this.n = new b.b.g.f(aVar != null ? aVar.b() : this.i);
        }
        return this.n;
    }

    @Override // b.b.c.j
    public b.b.c.a f() {
        N();
        return this.m;
    }

    @Override // b.b.c.j
    public void g() {
        LayoutInflater from = LayoutInflater.from(this.i);
        if (from.getFactory() == null) {
            from.setFactory2(this);
        } else if (from.getFactory2() instanceof k) {
        } else {
            Log.i("AppCompatDelegate", "The Activity's LayoutInflater already has a Factory installed so we can not install AppCompat's");
        }
    }

    @Override // b.b.c.j
    public void h() {
        N();
        b.b.c.a aVar = this.m;
        O(0);
    }

    @Override // b.b.c.j
    public void i(Configuration configuration) {
        if (this.D && this.x) {
            N();
            b.b.c.a aVar = this.m;
            if (aVar != null) {
                u uVar = (u) aVar;
                uVar.f(uVar.f616c.getResources().getBoolean(R.bool.abc_action_bar_embed_tabs));
            }
        }
        b.b.h.j a2 = b.b.h.j.a();
        Context context = this.i;
        synchronized (a2) {
            n0 n0Var = a2.f864c;
            synchronized (n0Var) {
                b.f.e<WeakReference<Drawable.ConstantState>> eVar = n0Var.f892g.get(context);
                if (eVar != null) {
                    eVar.a();
                }
            }
        }
        y(false);
    }

    @Override // b.b.c.j
    public void j(Bundle bundle) {
        this.M = true;
        y(false);
        I();
        Object obj = this.f571h;
        if (obj instanceof Activity) {
            String str = null;
            try {
                Activity activity = (Activity) obj;
                try {
                    str = b.j.b.d.w(activity, activity.getComponentName());
                } catch (PackageManager.NameNotFoundException e2) {
                    throw new IllegalArgumentException(e2);
                }
            } catch (IllegalArgumentException unused) {
            }
            if (str != null) {
                b.b.c.a aVar = this.m;
                if (aVar == null) {
                    this.Z = true;
                } else {
                    u uVar = (u) aVar;
                    if (!uVar.j) {
                        uVar.c(true);
                    }
                }
            }
            synchronized (b.b.c.j.f566c) {
                b.b.c.j.q(this);
                b.b.c.j.f565b.add(new WeakReference<>(this));
            }
        }
        this.N = true;
    }

    /* JADX WARN: Removed duplicated region for block: B:25:0x0063  */
    /* JADX WARN: Removed duplicated region for block: B:28:0x006a  */
    /* JADX WARN: Removed duplicated region for block: B:31:0x0071  */
    /* JADX WARN: Removed duplicated region for block: B:35:? A[RETURN, SYNTHETIC] */
    @Override // b.b.c.j
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void k() {
        b.b.c.a aVar;
        f fVar;
        f fVar2;
        if (this.f571h instanceof Activity) {
            synchronized (b.b.c.j.f566c) {
                b.b.c.j.q(this);
            }
        }
        if (this.W) {
            this.j.getDecorView().removeCallbacks(this.Y);
        }
        this.O = false;
        this.P = true;
        if (this.Q != -100) {
            Object obj = this.f571h;
            if ((obj instanceof Activity) && ((Activity) obj).isChangingConfigurations()) {
                f567d.put(this.f571h.getClass().getName(), Integer.valueOf(this.Q));
                aVar = this.m;
                if (aVar != null) {
                    Objects.requireNonNull(aVar);
                }
                fVar = this.U;
                if (fVar != null) {
                    fVar.a();
                }
                fVar2 = this.V;
                if (fVar2 == null) {
                    fVar2.a();
                    return;
                }
                return;
            }
        }
        f567d.remove(this.f571h.getClass().getName());
        aVar = this.m;
        if (aVar != null) {
        }
        fVar = this.U;
        if (fVar != null) {
        }
        fVar2 = this.V;
        if (fVar2 == null) {
        }
    }

    @Override // b.b.c.j
    public void l(Bundle bundle) {
        H();
    }

    @Override // b.b.c.j
    public void m() {
        N();
        b.b.c.a aVar = this.m;
        if (aVar != null) {
            ((u) aVar).w = true;
        }
    }

    @Override // b.b.c.j
    public void n(Bundle bundle) {
    }

    @Override // b.b.c.j
    public void o() {
        this.O = true;
        x();
    }

    @Override // android.view.LayoutInflater.Factory2
    public final View onCreateView(View view, String str, Context context, AttributeSet attributeSet) {
        if (this.c0 == null) {
            String string = this.i.obtainStyledAttributes(b.b.b.j).getString(114);
            if (string == null) {
                this.c0 = new r();
            } else {
                try {
                    this.c0 = (r) Class.forName(string).getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
                } catch (Throwable th) {
                    Log.i("AppCompatDelegate", "Failed to instantiate custom view inflater " + string + ". Falling back to default.", th);
                    this.c0 = new r();
                }
            }
        }
        r rVar = this.c0;
        int i2 = d1.f822a;
        return rVar.createView(view, str, context, attributeSet, false, false, true, false);
    }

    @Override // b.b.g.i.g.a
    public boolean onMenuItemSelected(b.b.g.i.g gVar, MenuItem menuItem) {
        i J;
        Window.Callback M = M();
        if (M == null || this.P || (J = J(gVar.getRootMenu())) == null) {
            return false;
        }
        return M.onMenuItemSelected(J.f585a, menuItem);
    }

    @Override // b.b.g.i.g.a
    public void onMenuModeChange(b.b.g.i.g gVar) {
        c0 c0Var = this.p;
        if (c0Var != null && c0Var.d() && (!ViewConfiguration.get(this.i).hasPermanentMenuKey() || this.p.e())) {
            Window.Callback M = M();
            if (this.p.b()) {
                this.p.f();
                if (this.P) {
                    return;
                }
                M.onPanelClosed(108, L(0).f592h);
                return;
            } else if (M == null || this.P) {
                return;
            } else {
                if (this.W && (1 & this.X) != 0) {
                    this.j.getDecorView().removeCallbacks(this.Y);
                    this.Y.run();
                }
                i L = L(0);
                b.b.g.i.g gVar2 = L.f592h;
                if (gVar2 == null || L.p || !M.onPreparePanel(0, L.f591g, gVar2)) {
                    return;
                }
                M.onMenuOpened(108, L.f592h);
                this.p.g();
                return;
            }
        }
        i L2 = L(0);
        L2.o = true;
        C(L2, false);
        Q(L2, null);
    }

    @Override // b.b.c.j
    public void p() {
        this.O = false;
        N();
        b.b.c.a aVar = this.m;
        if (aVar != null) {
            u uVar = (u) aVar;
            uVar.w = false;
            b.b.g.g gVar = uVar.v;
            if (gVar != null) {
                gVar.a();
            }
        }
    }

    @Override // b.b.c.j
    public boolean r(int i2) {
        if (i2 == 8) {
            Log.i("AppCompatDelegate", "You should now use the AppCompatDelegate.FEATURE_SUPPORT_ACTION_BAR id when requesting this feature.");
            i2 = 108;
        } else if (i2 == 9) {
            Log.i("AppCompatDelegate", "You should now use the AppCompatDelegate.FEATURE_SUPPORT_ACTION_BAR_OVERLAY id when requesting this feature.");
            i2 = 109;
        }
        if (this.H && i2 == 108) {
            return false;
        }
        if (this.D && i2 == 1) {
            this.D = false;
        }
        if (i2 == 1) {
            U();
            this.H = true;
            return true;
        } else if (i2 == 2) {
            U();
            this.B = true;
            return true;
        } else if (i2 == 5) {
            U();
            this.C = true;
            return true;
        } else if (i2 == 10) {
            U();
            this.F = true;
            return true;
        } else if (i2 == 108) {
            U();
            this.D = true;
            return true;
        } else if (i2 != 109) {
            return this.j.requestFeature(i2);
        } else {
            U();
            this.E = true;
            return true;
        }
    }

    @Override // b.b.c.j
    public void s(int i2) {
        H();
        ViewGroup viewGroup = (ViewGroup) this.y.findViewById(16908290);
        viewGroup.removeAllViews();
        LayoutInflater.from(this.i).inflate(i2, viewGroup);
        this.k.f678b.onContentChanged();
    }

    @Override // b.b.c.j
    public void t(View view) {
        H();
        ViewGroup viewGroup = (ViewGroup) this.y.findViewById(16908290);
        viewGroup.removeAllViews();
        viewGroup.addView(view);
        this.k.f678b.onContentChanged();
    }

    @Override // b.b.c.j
    public void u(View view, ViewGroup.LayoutParams layoutParams) {
        H();
        ViewGroup viewGroup = (ViewGroup) this.y.findViewById(16908290);
        viewGroup.removeAllViews();
        viewGroup.addView(view, layoutParams);
        this.k.f678b.onContentChanged();
    }

    @Override // b.b.c.j
    public void v(int i2) {
        this.R = i2;
    }

    @Override // b.b.c.j
    public final void w(CharSequence charSequence) {
        this.o = charSequence;
        c0 c0Var = this.p;
        if (c0Var != null) {
            c0Var.setWindowTitle(charSequence);
            return;
        }
        b.b.c.a aVar = this.m;
        if (aVar != null) {
            ((u) aVar).f620g.setWindowTitle(charSequence);
            return;
        }
        TextView textView = this.z;
        if (textView != null) {
            textView.setText(charSequence);
        }
    }

    public boolean x() {
        return y(true);
    }

    /* JADX WARN: Removed duplicated region for block: B:108:0x0187  */
    /* JADX WARN: Removed duplicated region for block: B:112:0x0194  */
    /* JADX WARN: Removed duplicated region for block: B:113:0x019e  */
    /* JADX WARN: Removed duplicated region for block: B:118:0x01a8  */
    /* JADX WARN: Removed duplicated region for block: B:122:0x01bb  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x00a6  */
    /* JADX WARN: Removed duplicated region for block: B:48:0x00aa  */
    /* JADX WARN: Removed duplicated region for block: B:54:0x00b8 A[ADDED_TO_REGION] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final boolean y(boolean z) {
        boolean z2;
        int i2;
        int i3;
        boolean z3;
        Object obj;
        Object obj2;
        if (this.P) {
            return false;
        }
        int i4 = this.Q;
        if (i4 == -100) {
            i4 = -100;
        }
        Object obj3 = null;
        Configuration D = D(this.i, P(this.i, i4), null);
        boolean z4 = true;
        if (!this.T && (this.f571h instanceof Activity)) {
            PackageManager packageManager = this.i.getPackageManager();
            if (packageManager != null) {
                try {
                    ActivityInfo activityInfo = packageManager.getActivityInfo(new ComponentName(this.i, this.f571h.getClass()), Build.VERSION.SDK_INT >= 29 ? 269221888 : 786432);
                    this.S = (activityInfo == null || (activityInfo.configChanges & 512) == 0) ? false : true;
                } catch (PackageManager.NameNotFoundException e2) {
                    Log.d("AppCompatDelegate", "Exception while getting ActivityInfo", e2);
                    this.S = false;
                }
            } else {
                z2 = false;
                i2 = this.i.getResources().getConfiguration().uiMode & 48;
                i3 = D.uiMode & 48;
                if (i2 != i3 && z && !z2 && this.M && (f569f || this.N)) {
                    obj2 = this.f571h;
                    if ((obj2 instanceof Activity) && !((Activity) obj2).isChild()) {
                        Activity activity = (Activity) this.f571h;
                        int i5 = b.j.b.a.f2030b;
                        if (Build.VERSION.SDK_INT < 28) {
                            activity.recreate();
                        } else if (!b.j.b.b.b(activity)) {
                            activity.recreate();
                        }
                        z3 = true;
                        if (!z3 || i2 == i3) {
                            z4 = z3;
                        } else {
                            Resources resources = this.i.getResources();
                            Configuration configuration = new Configuration(resources.getConfiguration());
                            configuration.uiMode = i3 | (resources.getConfiguration().uiMode & (-49));
                            resources.updateConfiguration(configuration, null);
                            int i6 = Build.VERSION.SDK_INT;
                            if (i6 < 26 && i6 < 28) {
                                if (!b.b.a.f540h) {
                                    try {
                                        Field declaredField = Resources.class.getDeclaredField("mResourcesImpl");
                                        b.b.a.f539g = declaredField;
                                        declaredField.setAccessible(true);
                                    } catch (NoSuchFieldException e3) {
                                        Log.e("ResourcesFlusher", "Could not retrieve Resources#mResourcesImpl field", e3);
                                    }
                                    b.b.a.f540h = true;
                                }
                                Field field = b.b.a.f539g;
                                if (field != null) {
                                    try {
                                        obj = field.get(resources);
                                    } catch (IllegalAccessException e4) {
                                        Log.e("ResourcesFlusher", "Could not retrieve value from Resources#mResourcesImpl", e4);
                                        obj = null;
                                    }
                                    if (obj != null) {
                                        if (!b.b.a.f534b) {
                                            try {
                                                Field declaredField2 = obj.getClass().getDeclaredField("mDrawableCache");
                                                b.b.a.f533a = declaredField2;
                                                declaredField2.setAccessible(true);
                                            } catch (NoSuchFieldException e5) {
                                                Log.e("ResourcesFlusher", "Could not retrieve ResourcesImpl#mDrawableCache field", e5);
                                            }
                                            b.b.a.f534b = true;
                                        }
                                        Field field2 = b.b.a.f533a;
                                        if (field2 != null) {
                                            try {
                                                obj3 = field2.get(obj);
                                            } catch (IllegalAccessException e6) {
                                                Log.e("ResourcesFlusher", "Could not retrieve value from ResourcesImpl#mDrawableCache", e6);
                                            }
                                        }
                                        if (obj3 != null) {
                                            b.b.a.g(obj3);
                                        }
                                    }
                                }
                            }
                            int i7 = this.R;
                            if (i7 != 0) {
                                this.i.setTheme(i7);
                                this.i.getTheme().applyStyle(this.R, true);
                            }
                            if (z2) {
                                Object obj4 = this.f571h;
                                if (obj4 instanceof Activity) {
                                    Activity activity2 = (Activity) obj4;
                                    if (activity2 instanceof b.t.h) {
                                        if (((b.t.i) ((b.t.h) activity2).getLifecycle()).f2579b.compareTo(e.b.STARTED) >= 0) {
                                            activity2.onConfigurationChanged(configuration);
                                        }
                                    } else if (this.O) {
                                        activity2.onConfigurationChanged(configuration);
                                    }
                                }
                            }
                        }
                        if (z4) {
                            Object obj5 = this.f571h;
                            if (obj5 instanceof b.b.c.h) {
                                ((b.b.c.h) obj5).t();
                            }
                        }
                        if (i4 != 0) {
                            K(this.i).e();
                        } else {
                            f fVar = this.U;
                            if (fVar != null) {
                                fVar.a();
                            }
                        }
                        if (i4 != 3) {
                            Context context = this.i;
                            if (this.V == null) {
                                this.V = new e(context);
                            }
                            this.V.e();
                        } else {
                            f fVar2 = this.V;
                            if (fVar2 != null) {
                                fVar2.a();
                            }
                        }
                        return z4;
                    }
                }
                z3 = false;
                if (z3) {
                }
                z4 = z3;
                if (z4) {
                }
                if (i4 != 0) {
                }
                if (i4 != 3) {
                }
                return z4;
            }
        }
        this.T = true;
        z2 = this.S;
        i2 = this.i.getResources().getConfiguration().uiMode & 48;
        i3 = D.uiMode & 48;
        if (i2 != i3) {
            obj2 = this.f571h;
            if (obj2 instanceof Activity) {
                Activity activity3 = (Activity) this.f571h;
                int i52 = b.j.b.a.f2030b;
                if (Build.VERSION.SDK_INT < 28) {
                }
                z3 = true;
                if (z3) {
                }
                z4 = z3;
                if (z4) {
                }
                if (i4 != 0) {
                }
                if (i4 != 3) {
                }
                return z4;
            }
        }
        z3 = false;
        if (z3) {
        }
        z4 = z3;
        if (z4) {
        }
        if (i4 != 0) {
        }
        if (i4 != 3) {
        }
        return z4;
    }

    public final void z(Window window) {
        if (this.j == null) {
            Window.Callback callback = window.getCallback();
            if (!(callback instanceof d)) {
                d dVar = new d(callback);
                this.k = dVar;
                window.setCallback(dVar);
                y0 q = y0.q(this.i, null, f568e);
                Drawable h2 = q.h(0);
                if (h2 != null) {
                    window.setBackgroundDrawable(h2);
                }
                q.f972b.recycle();
                this.j = window;
                return;
            }
            throw new IllegalStateException("AppCompat has already installed itself into the Window");
        }
        throw new IllegalStateException("AppCompat has already installed itself into the Window");
    }

    @Override // android.view.LayoutInflater.Factory
    public View onCreateView(String str, Context context, AttributeSet attributeSet) {
        return onCreateView(null, str, context, attributeSet);
    }
}