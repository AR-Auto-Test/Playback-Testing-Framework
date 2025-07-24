package b.b.c;

import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import b.b.g.a;
import b.j.j.d;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;

/* compiled from: AppCompatDialog.java */
/* loaded from: classes.dex */
public class p extends Dialog implements i {
    private j mDelegate;
    private final d.a mKeyDispatcher;

    /* compiled from: AppCompatDialog.java */
    /* loaded from: classes.dex */
    public class a implements d.a {
        public a() {
        }

        @Override // b.j.j.d.a
        public boolean i(KeyEvent keyEvent) {
            return p.this.superDispatchKeyEvent(keyEvent);
        }
    }

    public p(Context context) {
        this(context, 0);
    }

    private static int getThemeResId(Context context, int i) {
        if (i == 0) {
            TypedValue typedValue = new TypedValue();
            context.getTheme().resolveAttribute(R.attr.dialogTheme, typedValue, true);
            return typedValue.resourceId;
        }
        return i;
    }

    @Override // android.app.Dialog
    public void addContentView(View view, ViewGroup.LayoutParams layoutParams) {
        getDelegate().a(view, layoutParams);
    }

    @Override // android.app.Dialog, android.content.DialogInterface
    public void dismiss() {
        super.dismiss();
        getDelegate().k();
    }

    @Override // android.app.Dialog, android.view.Window.Callback
    public boolean dispatchKeyEvent(KeyEvent keyEvent) {
        return b.j.j.d.b(this.mKeyDispatcher, getWindow().getDecorView(), this, keyEvent);
    }

    @Override // android.app.Dialog
    public <T extends View> T findViewById(int i) {
        return (T) getDelegate().c(i);
    }

    public j getDelegate() {
        if (this.mDelegate == null) {
            b.f.c<WeakReference<j>> cVar = j.f565b;
            this.mDelegate = new k(getContext(), getWindow(), this, this);
        }
        return this.mDelegate;
    }

    public b.b.c.a getSupportActionBar() {
        return getDelegate().f();
    }

    @Override // android.app.Dialog
    public void invalidateOptionsMenu() {
        getDelegate().h();
    }

    @Override // android.app.Dialog
    public void onCreate(Bundle bundle) {
        getDelegate().g();
        super.onCreate(bundle);
        getDelegate().j(bundle);
    }

    @Override // android.app.Dialog
    public void onStop() {
        super.onStop();
        getDelegate().p();
    }

    @Override // b.b.c.i
    public void onSupportActionModeFinished(b.b.g.a aVar) {
    }

    @Override // b.b.c.i
    public void onSupportActionModeStarted(b.b.g.a aVar) {
    }

    @Override // b.b.c.i
    public b.b.g.a onWindowStartingSupportActionMode(a.InterfaceC0007a interfaceC0007a) {
        return null;
    }

    @Override // android.app.Dialog
    public void setContentView(int i) {
        getDelegate().s(i);
    }

    @Override // android.app.Dialog
    public void setTitle(CharSequence charSequence) {
        super.setTitle(charSequence);
        getDelegate().w(charSequence);
    }

    public boolean superDispatchKeyEvent(KeyEvent keyEvent) {
        return super.dispatchKeyEvent(keyEvent);
    }

    public boolean supportRequestWindowFeature(int i) {
        return getDelegate().r(i);
    }

    public p(Context context, int i) {
        super(context, getThemeResId(context, i));
        this.mKeyDispatcher = new a();
        j delegate = getDelegate();
        delegate.v(getThemeResId(context, i));
        delegate.j(null);
    }

    @Override // android.app.Dialog
    public void setContentView(View view) {
        getDelegate().t(view);
    }

    @Override // android.app.Dialog
    public void setContentView(View view, ViewGroup.LayoutParams layoutParams) {
        getDelegate().u(view, layoutParams);
    }

    @Override // android.app.Dialog
    public void setTitle(int i) {
        super.setTitle(i);
        getDelegate().w(getContext().getString(i));
    }

    public p(Context context, boolean z, DialogInterface.OnCancelListener onCancelListener) {
        super(context, z, onCancelListener);
        this.mKeyDispatcher = new a();
    }
}