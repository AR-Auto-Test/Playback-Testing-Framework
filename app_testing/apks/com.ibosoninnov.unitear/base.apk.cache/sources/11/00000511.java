package b.q.b;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.content.res.Configuration;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import androidx.activity.ComponentActivity;
import androidx.activity.OnBackPressedDispatcher;
import androidx.fragment.app.Fragment;
import b.j.b.a;
import b.t.e;
import java.io.FileDescriptor;
import java.io.PrintWriter;
import java.util.Objects;

/* compiled from: FragmentActivity.java */
/* loaded from: classes.dex */
public class d extends ComponentActivity implements a.InterfaceC0031a {

    /* renamed from: g  reason: collision with root package name */
    public final l f2418g;

    /* renamed from: h  reason: collision with root package name */
    public final b.t.i f2419h;
    public boolean i;
    public boolean j;
    public boolean k;
    public boolean l;
    public boolean m;
    public boolean n;
    public int o;
    public b.f.i<String> p;

    /* compiled from: FragmentActivity.java */
    /* loaded from: classes.dex */
    public class a extends n<d> implements b.t.z, b.a.c {
        public a() {
            super(d.this);
        }

        @Override // b.q.b.j
        public View a(int i) {
            return d.this.findViewById(i);
        }

        @Override // b.a.c
        public OnBackPressedDispatcher b() {
            return d.this.f41f;
        }

        @Override // b.q.b.j
        public boolean c() {
            Window window = d.this.getWindow();
            return (window == null || window.peekDecorView() == null) ? false : true;
        }

        @Override // b.q.b.n
        public void d(Fragment fragment) {
            d.this.o();
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.q.b.n
        public d e() {
            return d.this;
        }

        @Override // b.q.b.n
        public LayoutInflater f() {
            return d.this.getLayoutInflater().cloneInContext(d.this);
        }

        @Override // b.q.b.n
        public void g(Fragment fragment, String[] strArr, int i) {
            d dVar = d.this;
            Objects.requireNonNull(dVar);
            if (i == -1) {
                b.j.b.a.c(dVar, strArr, i);
                return;
            }
            d.l(i);
            try {
                dVar.l = true;
                b.j.b.a.c(dVar, strArr, ((dVar.k(fragment) + 1) << 16) + (i & 65535));
            } finally {
                dVar.l = false;
            }
        }

        @Override // b.t.h
        public b.t.e getLifecycle() {
            return d.this.f2419h;
        }

        @Override // b.t.z
        public b.t.y getViewModelStore() {
            return d.this.getViewModelStore();
        }

        @Override // b.q.b.n
        public boolean h(Fragment fragment) {
            return !d.this.isFinishing();
        }

        @Override // b.q.b.n
        public boolean i(String str) {
            d dVar = d.this;
            int i = b.j.b.a.f2030b;
            return dVar.shouldShowRequestPermissionRationale(str);
        }

        @Override // b.q.b.n
        public void j(Fragment fragment, Intent intent, int i, Bundle bundle) {
            d dVar = d.this;
            dVar.n = true;
            try {
                if (i == -1) {
                    int i2 = b.j.b.a.f2030b;
                    dVar.startActivityForResult(intent, -1, bundle);
                } else {
                    d.l(i);
                    int k = ((dVar.k(fragment) + 1) << 16) + (i & 65535);
                    int i3 = b.j.b.a.f2030b;
                    dVar.startActivityForResult(intent, k, bundle);
                }
            } finally {
                dVar.n = false;
            }
        }

        @Override // b.q.b.n
        public void k(Fragment fragment, IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4, Bundle bundle) {
            d dVar = d.this;
            dVar.m = true;
            try {
                if (i == -1) {
                    int i5 = b.j.b.a.f2030b;
                    dVar.startIntentSenderForResult(intentSender, i, intent, i2, i3, i4, bundle);
                } else {
                    d.l(i);
                    int k = ((dVar.k(fragment) + 1) << 16) + (i & 65535);
                    int i6 = b.j.b.a.f2030b;
                    dVar.startIntentSenderForResult(intentSender, k, intent, i2, i3, i4, bundle);
                }
            } finally {
                dVar.m = false;
            }
        }

        @Override // b.q.b.n
        public void l() {
            d.this.p();
        }
    }

    public d() {
        a aVar = new a();
        b.j.b.d.h(aVar, "callbacks == null");
        this.f2418g = new l(aVar);
        this.f2419h = new b.t.i(this);
        this.k = true;
    }

    public static void l(int i) {
        if ((i & (-65536)) != 0) {
            throw new IllegalArgumentException("Can only use lower 16 bits for requestCode");
        }
    }

    public static boolean n(q qVar, e.b bVar) {
        boolean z = false;
        for (Fragment fragment : qVar.f2498c.g()) {
            if (fragment != null) {
                if (fragment.getHost() != null) {
                    z |= n(fragment.getChildFragmentManager(), bVar);
                }
                if (((b.t.i) fragment.getLifecycle()).f2579b.compareTo(e.b.STARTED) >= 0) {
                    fragment.mLifecycleRegistry.f(bVar);
                    z = true;
                }
            }
        }
        return z;
    }

    @Override // b.j.b.a.InterfaceC0031a
    public final void a(int i) {
        if (this.l || i == -1) {
            return;
        }
        l(i);
    }

    @Override // android.app.Activity
    public void dump(String str, FileDescriptor fileDescriptor, PrintWriter printWriter, String[] strArr) {
        super.dump(str, fileDescriptor, printWriter, strArr);
        printWriter.print(str);
        printWriter.print("Local FragmentActivity ");
        printWriter.print(Integer.toHexString(System.identityHashCode(this)));
        printWriter.println(" State:");
        String str2 = str + "  ";
        printWriter.print(str2);
        printWriter.print("mCreated=");
        printWriter.print(this.i);
        printWriter.print(" mResumed=");
        printWriter.print(this.j);
        printWriter.print(" mStopped=");
        printWriter.print(this.k);
        if (getApplication() != null) {
            b.u.a.a.b(this).a(str2, fileDescriptor, printWriter, strArr);
        }
        this.f2418g.f2486a.f2492e.y(str, fileDescriptor, printWriter, strArr);
    }

    public final int k(Fragment fragment) {
        if (this.p.i() >= 65534) {
            throw new IllegalStateException("Too many pending Fragment activity results.");
        }
        while (true) {
            b.f.i<String> iVar = this.p;
            int i = this.o;
            if (iVar.f1777c) {
                iVar.c();
            }
            if (b.f.d.a(iVar.f1778d, iVar.f1780f, i) >= 0) {
                this.o = (this.o + 1) % 65534;
            } else {
                int i2 = this.o;
                this.p.g(i2, fragment.mWho);
                this.o = (this.o + 1) % 65534;
                return i2;
            }
        }
    }

    public q m() {
        return this.f2418g.f2486a.f2492e;
    }

    public void o() {
    }

    @Override // android.app.Activity
    public void onActivityResult(int i, int i2, Intent intent) {
        this.f2418g.a();
        int i3 = i >> 16;
        if (i3 != 0) {
            int i4 = i3 - 1;
            String d2 = this.p.d(i4);
            this.p.h(i4);
            if (d2 == null) {
                Log.w("FragmentActivity", "Activity result delivered for unknown Fragment.");
                return;
            }
            Fragment J = this.f2418g.f2486a.f2492e.J(d2);
            if (J == null) {
                Log.w("FragmentActivity", "Activity result no fragment exists for who: " + d2);
                return;
            }
            J.onActivityResult(i & 65535, i2, intent);
            return;
        }
        int i5 = b.j.b.a.f2030b;
        super.onActivityResult(i, i2, intent);
    }

    @Override // android.app.Activity, android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        this.f2418g.a();
        this.f2418g.f2486a.f2492e.k(configuration);
    }

    @Override // androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        n<?> nVar = this.f2418g.f2486a;
        nVar.f2492e.d(nVar, nVar, null);
        if (bundle != null) {
            Parcelable parcelable = bundle.getParcelable("android:support:fragments");
            n<?> nVar2 = this.f2418g.f2486a;
            if (nVar2 instanceof b.t.z) {
                nVar2.f2492e.c0(parcelable);
                if (bundle.containsKey("android:support:next_request_index")) {
                    this.o = bundle.getInt("android:support:next_request_index");
                    int[] intArray = bundle.getIntArray("android:support:request_indicies");
                    String[] stringArray = bundle.getStringArray("android:support:request_fragment_who");
                    if (intArray != null && stringArray != null && intArray.length == stringArray.length) {
                        this.p = new b.f.i<>(intArray.length);
                        for (int i = 0; i < intArray.length; i++) {
                            this.p.g(intArray[i], stringArray[i]);
                        }
                    } else {
                        Log.w("FragmentActivity", "Invalid requestCode mapping in savedInstanceState.");
                    }
                }
            } else {
                throw new IllegalStateException("Your FragmentHostCallback must implement ViewModelStoreOwner to call restoreSaveState(). Call restoreAllState()  if you're still using retainNestedNonConfig().");
            }
        }
        if (this.p == null) {
            this.p = new b.f.i<>(10);
            this.o = 0;
        }
        super.onCreate(bundle);
        this.f2419h.d(e.a.ON_CREATE);
        this.f2418g.f2486a.f2492e.m();
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean onCreatePanelMenu(int i, Menu menu) {
        if (i == 0) {
            boolean onCreatePanelMenu = super.onCreatePanelMenu(i, menu);
            l lVar = this.f2418g;
            return onCreatePanelMenu | lVar.f2486a.f2492e.n(menu, getMenuInflater());
        }
        return super.onCreatePanelMenu(i, menu);
    }

    @Override // android.app.Activity, android.view.LayoutInflater.Factory2
    public View onCreateView(View view, String str, Context context, AttributeSet attributeSet) {
        View onCreateView = this.f2418g.f2486a.f2492e.f2501f.onCreateView(view, str, context, attributeSet);
        return onCreateView == null ? super.onCreateView(view, str, context, attributeSet) : onCreateView;
    }

    @Override // android.app.Activity
    public void onDestroy() {
        super.onDestroy();
        this.f2418g.f2486a.f2492e.o();
        this.f2419h.d(e.a.ON_DESTROY);
    }

    @Override // android.app.Activity, android.content.ComponentCallbacks
    public void onLowMemory() {
        super.onLowMemory();
        this.f2418g.f2486a.f2492e.p();
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean onMenuItemSelected(int i, MenuItem menuItem) {
        if (super.onMenuItemSelected(i, menuItem)) {
            return true;
        }
        if (i != 0) {
            if (i != 6) {
                return false;
            }
            return this.f2418g.f2486a.f2492e.l(menuItem);
        }
        return this.f2418g.f2486a.f2492e.r(menuItem);
    }

    @Override // android.app.Activity
    public void onMultiWindowModeChanged(boolean z) {
        this.f2418g.f2486a.f2492e.q(z);
    }

    @Override // android.app.Activity
    public void onNewIntent(@SuppressLint({"UnknownNullness"}) Intent intent) {
        super.onNewIntent(intent);
        this.f2418g.a();
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onPanelClosed(int i, Menu menu) {
        if (i == 0) {
            this.f2418g.f2486a.f2492e.s(menu);
        }
        super.onPanelClosed(i, menu);
    }

    @Override // android.app.Activity
    public void onPause() {
        super.onPause();
        this.j = false;
        this.f2418g.f2486a.f2492e.w(3);
        this.f2419h.d(e.a.ON_PAUSE);
    }

    @Override // android.app.Activity
    public void onPictureInPictureModeChanged(boolean z) {
        this.f2418g.f2486a.f2492e.u(z);
    }

    @Override // android.app.Activity
    public void onPostResume() {
        super.onPostResume();
        this.f2419h.d(e.a.ON_RESUME);
        q qVar = this.f2418g.f2486a.f2492e;
        qVar.t = false;
        qVar.u = false;
        qVar.w(4);
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean onPreparePanel(int i, View view, Menu menu) {
        if (i == 0) {
            return super.onPreparePanel(0, view, menu) | this.f2418g.f2486a.f2492e.v(menu);
        }
        return super.onPreparePanel(i, view, menu);
    }

    @Override // android.app.Activity
    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        this.f2418g.a();
        int i2 = (i >> 16) & 65535;
        if (i2 != 0) {
            int i3 = i2 - 1;
            String d2 = this.p.d(i3);
            this.p.h(i3);
            if (d2 == null) {
                Log.w("FragmentActivity", "Activity result delivered for unknown Fragment.");
                return;
            }
            Fragment J = this.f2418g.f2486a.f2492e.J(d2);
            if (J == null) {
                Log.w("FragmentActivity", "Activity result no fragment exists for who: " + d2);
                return;
            }
            J.onRequestPermissionsResult(i & 65535, strArr, iArr);
        }
    }

    @Override // android.app.Activity
    public void onResume() {
        super.onResume();
        this.j = true;
        this.f2418g.a();
        this.f2418g.f2486a.f2492e.C(true);
    }

    @Override // androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onSaveInstanceState(Bundle bundle) {
        super.onSaveInstanceState(bundle);
        do {
        } while (n(m(), e.b.CREATED));
        this.f2419h.d(e.a.ON_STOP);
        Parcelable d0 = this.f2418g.f2486a.f2492e.d0();
        if (d0 != null) {
            bundle.putParcelable("android:support:fragments", d0);
        }
        if (this.p.i() > 0) {
            bundle.putInt("android:support:next_request_index", this.o);
            int[] iArr = new int[this.p.i()];
            String[] strArr = new String[this.p.i()];
            for (int i = 0; i < this.p.i(); i++) {
                iArr[i] = this.p.f(i);
                strArr[i] = this.p.j(i);
            }
            bundle.putIntArray("android:support:request_indicies", iArr);
            bundle.putStringArray("android:support:request_fragment_who", strArr);
        }
    }

    @Override // android.app.Activity
    public void onStart() {
        super.onStart();
        this.k = false;
        if (!this.i) {
            this.i = true;
            q qVar = this.f2418g.f2486a.f2492e;
            qVar.t = false;
            qVar.u = false;
            qVar.w(2);
        }
        this.f2418g.a();
        this.f2418g.f2486a.f2492e.C(true);
        this.f2419h.d(e.a.ON_START);
        q qVar2 = this.f2418g.f2486a.f2492e;
        qVar2.t = false;
        qVar2.u = false;
        qVar2.w(3);
    }

    @Override // android.app.Activity
    public void onStateNotSaved() {
        this.f2418g.a();
    }

    @Override // android.app.Activity
    public void onStop() {
        super.onStop();
        this.k = true;
        do {
        } while (n(m(), e.b.CREATED));
        q qVar = this.f2418g.f2486a.f2492e;
        qVar.u = true;
        qVar.w(2);
        this.f2419h.d(e.a.ON_STOP);
    }

    @Deprecated
    public void p() {
        invalidateOptionsMenu();
    }

    @Override // android.app.Activity
    public void startActivityForResult(@SuppressLint({"UnknownNullness"}) Intent intent, int i) {
        if (!this.n && i != -1) {
            l(i);
        }
        super.startActivityForResult(intent, i);
    }

    @Override // android.app.Activity
    public void startIntentSenderForResult(@SuppressLint({"UnknownNullness"}) IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4) {
        if (!this.m && i != -1) {
            l(i);
        }
        super.startIntentSenderForResult(intentSender, i, intent, i2, i3, i4);
    }

    @Override // android.app.Activity
    public void startActivityForResult(@SuppressLint({"UnknownNullness"}) Intent intent, int i, Bundle bundle) {
        if (!this.n && i != -1) {
            l(i);
        }
        super.startActivityForResult(intent, i, bundle);
    }

    @Override // android.app.Activity
    public void startIntentSenderForResult(@SuppressLint({"UnknownNullness"}) IntentSender intentSender, int i, Intent intent, int i2, int i3, int i4, Bundle bundle) {
        if (!this.m && i != -1) {
            l(i);
        }
        super.startIntentSenderForResult(intentSender, i, intent, i2, i3, i4, bundle);
    }

    @Override // android.app.Activity, android.view.LayoutInflater.Factory
    public View onCreateView(String str, Context context, AttributeSet attributeSet) {
        View onCreateView = this.f2418g.f2486a.f2492e.f2501f.onCreateView(null, str, context, attributeSet);
        return onCreateView == null ? super.onCreateView(str, context, attributeSet) : onCreateView;
    }
}