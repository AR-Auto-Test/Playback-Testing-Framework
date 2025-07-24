package b.b.c;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import b.b.g.a;
import b.b.h.d1;
import java.lang.ref.WeakReference;
import java.util.ArrayList;

/* compiled from: AppCompatActivity.java */
/* loaded from: classes.dex */
public class h extends b.q.b.d implements i {
    public j q;

    @Override // android.app.Activity
    public void addContentView(View view, ViewGroup.LayoutParams layoutParams) {
        q().a(view, layoutParams);
    }

    @Override // android.app.Activity, android.view.ContextThemeWrapper, android.content.ContextWrapper
    public void attachBaseContext(Context context) {
        super.attachBaseContext(q().b(context));
    }

    @Override // android.app.Activity
    public void closeOptionsMenu() {
        r();
        if (getWindow().hasFeature(0)) {
            super.closeOptionsMenu();
        }
    }

    @Override // b.j.b.e, android.app.Activity, android.view.Window.Callback
    public boolean dispatchKeyEvent(KeyEvent keyEvent) {
        keyEvent.getKeyCode();
        r();
        return super.dispatchKeyEvent(keyEvent);
    }

    @Override // android.app.Activity
    public <T extends View> T findViewById(int i) {
        return (T) q().c(i);
    }

    @Override // android.app.Activity
    public MenuInflater getMenuInflater() {
        return q().e();
    }

    @Override // android.view.ContextThemeWrapper, android.content.ContextWrapper, android.content.Context
    public Resources getResources() {
        int i = d1.f822a;
        return super.getResources();
    }

    @Override // android.app.Activity
    public void invalidateOptionsMenu() {
        q().h();
    }

    @Override // b.q.b.d, android.app.Activity, android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        q().i(configuration);
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public void onContentChanged() {
    }

    @Override // b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        j q = q();
        q.g();
        q.j(bundle);
        super.onCreate(bundle);
    }

    @Override // b.q.b.d, android.app.Activity
    public void onDestroy() {
        super.onDestroy();
        q().k();
    }

    @Override // android.app.Activity, android.view.KeyEvent.Callback
    public boolean onKeyDown(int i, KeyEvent keyEvent) {
        Window window;
        if ((Build.VERSION.SDK_INT >= 26 || keyEvent.isCtrlPressed() || KeyEvent.metaStateHasNoModifiers(keyEvent.getMetaState()) || keyEvent.getRepeatCount() != 0 || KeyEvent.isModifierKey(keyEvent.getKeyCode()) || (window = getWindow()) == null || window.getDecorView() == null || !window.getDecorView().dispatchKeyShortcutEvent(keyEvent)) ? false : true) {
            return true;
        }
        return super.onKeyDown(i, keyEvent);
    }

    @Override // b.q.b.d, android.app.Activity, android.view.Window.Callback
    public final boolean onMenuItemSelected(int i, MenuItem menuItem) {
        Intent u;
        if (super.onMenuItemSelected(i, menuItem)) {
            return true;
        }
        a r = r();
        if (menuItem.getItemId() == 16908332 && r != null && (((u) r).f620g.s() & 4) != 0 && (u = b.j.b.d.u(this)) != null) {
            if (shouldUpRecreateTask(u)) {
                ArrayList arrayList = new ArrayList();
                Intent s = s();
                if (s == null) {
                    s = b.j.b.d.u(this);
                }
                if (s != null) {
                    ComponentName component = s.getComponent();
                    if (component == null) {
                        component = s.resolveActivity(getPackageManager());
                    }
                    int size = arrayList.size();
                    try {
                        Intent v = b.j.b.d.v(this, component);
                        while (v != null) {
                            arrayList.add(size, v);
                            v = b.j.b.d.v(this, v.getComponent());
                        }
                        arrayList.add(s);
                    } catch (PackageManager.NameNotFoundException e2) {
                        Log.e("TaskStackBuilder", "Bad ComponentName while traversing activity parent metadata");
                        throw new IllegalArgumentException(e2);
                    }
                }
                u();
                if (!arrayList.isEmpty()) {
                    Intent[] intentArr = (Intent[]) arrayList.toArray(new Intent[arrayList.size()]);
                    intentArr[0] = new Intent(intentArr[0]).addFlags(268484608);
                    Object obj = b.j.c.a.f2074a;
                    startActivities(intentArr, null);
                    try {
                        int i2 = b.j.b.a.f2030b;
                        finishAffinity();
                        return true;
                    } catch (IllegalStateException unused) {
                        finish();
                        return true;
                    }
                }
                throw new IllegalStateException("No intents added to TaskStackBuilder; cannot startActivities");
            }
            navigateUpTo(u);
            return true;
        }
        return false;
    }

    @Override // android.app.Activity, android.view.Window.Callback
    public boolean onMenuOpened(int i, Menu menu) {
        return super.onMenuOpened(i, menu);
    }

    @Override // b.q.b.d, android.app.Activity, android.view.Window.Callback
    public void onPanelClosed(int i, Menu menu) {
        super.onPanelClosed(i, menu);
    }

    @Override // android.app.Activity
    public void onPostCreate(Bundle bundle) {
        super.onPostCreate(bundle);
        q().l(bundle);
    }

    @Override // b.q.b.d, android.app.Activity
    public void onPostResume() {
        super.onPostResume();
        q().m();
    }

    @Override // b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onSaveInstanceState(Bundle bundle) {
        super.onSaveInstanceState(bundle);
        q().n(bundle);
    }

    @Override // b.q.b.d, android.app.Activity
    public void onStart() {
        super.onStart();
        q().o();
    }

    @Override // b.q.b.d, android.app.Activity
    public void onStop() {
        super.onStop();
        q().p();
    }

    @Override // b.b.c.i
    public void onSupportActionModeFinished(b.b.g.a aVar) {
    }

    @Override // b.b.c.i
    public void onSupportActionModeStarted(b.b.g.a aVar) {
    }

    @Override // android.app.Activity
    public void onTitleChanged(CharSequence charSequence, int i) {
        super.onTitleChanged(charSequence, i);
        q().w(charSequence);
    }

    @Override // b.b.c.i
    public b.b.g.a onWindowStartingSupportActionMode(a.InterfaceC0007a interfaceC0007a) {
        return null;
    }

    @Override // android.app.Activity
    public void openOptionsMenu() {
        r();
        if (getWindow().hasFeature(0)) {
            super.openOptionsMenu();
        }
    }

    @Override // b.q.b.d
    public void p() {
        q().h();
    }

    public j q() {
        if (this.q == null) {
            b.f.c<WeakReference<j>> cVar = j.f565b;
            this.q = new k(this, null, this, this);
        }
        return this.q;
    }

    public a r() {
        return q().f();
    }

    public Intent s() {
        return b.j.b.d.u(this);
    }

    @Override // android.app.Activity
    public void setContentView(int i) {
        q().s(i);
    }

    @Override // android.app.Activity, android.view.ContextThemeWrapper, android.content.ContextWrapper, android.content.Context
    public void setTheme(int i) {
        super.setTheme(i);
        q().v(i);
    }

    public void t() {
    }

    public void u() {
    }

    @Override // android.app.Activity
    public void setContentView(View view) {
        q().t(view);
    }

    @Override // android.app.Activity
    public void setContentView(View view, ViewGroup.LayoutParams layoutParams) {
        q().u(view, layoutParams);
    }
}