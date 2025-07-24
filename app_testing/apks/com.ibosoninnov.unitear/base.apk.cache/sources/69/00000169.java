package b.b.g;

import android.content.Context;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import androidx.appcompat.widget.ActionBarContextView;
import b.b.g.a;
import b.b.g.i.g;
import java.lang.ref.WeakReference;

/* compiled from: StandaloneActionMode.java */
/* loaded from: classes.dex */
public class d extends a implements g.a {

    /* renamed from: d  reason: collision with root package name */
    public Context f641d;

    /* renamed from: e  reason: collision with root package name */
    public ActionBarContextView f642e;

    /* renamed from: f  reason: collision with root package name */
    public a.InterfaceC0007a f643f;

    /* renamed from: g  reason: collision with root package name */
    public WeakReference<View> f644g;

    /* renamed from: h  reason: collision with root package name */
    public boolean f645h;
    public b.b.g.i.g i;

    public d(Context context, ActionBarContextView actionBarContextView, a.InterfaceC0007a interfaceC0007a, boolean z) {
        this.f641d = context;
        this.f642e = actionBarContextView;
        this.f643f = interfaceC0007a;
        b.b.g.i.g defaultShowAsAction = new b.b.g.i.g(actionBarContextView.getContext()).setDefaultShowAsAction(1);
        this.i = defaultShowAsAction;
        defaultShowAsAction.setCallback(this);
    }

    @Override // b.b.g.a
    public void a() {
        if (this.f645h) {
            return;
        }
        this.f645h = true;
        this.f642e.sendAccessibilityEvent(32);
        this.f643f.a(this);
    }

    @Override // b.b.g.a
    public View b() {
        WeakReference<View> weakReference = this.f644g;
        if (weakReference != null) {
            return weakReference.get();
        }
        return null;
    }

    @Override // b.b.g.a
    public Menu c() {
        return this.i;
    }

    @Override // b.b.g.a
    public MenuInflater d() {
        return new f(this.f642e.getContext());
    }

    @Override // b.b.g.a
    public CharSequence e() {
        return this.f642e.getSubtitle();
    }

    @Override // b.b.g.a
    public CharSequence f() {
        return this.f642e.getTitle();
    }

    @Override // b.b.g.a
    public void g() {
        this.f643f.c(this, this.i);
    }

    @Override // b.b.g.a
    public boolean h() {
        return this.f642e.s;
    }

    @Override // b.b.g.a
    public void i(View view) {
        this.f642e.setCustomView(view);
        this.f644g = view != null ? new WeakReference<>(view) : null;
    }

    @Override // b.b.g.a
    public void j(int i) {
        this.f642e.setSubtitle(this.f641d.getString(i));
    }

    @Override // b.b.g.a
    public void k(CharSequence charSequence) {
        this.f642e.setSubtitle(charSequence);
    }

    @Override // b.b.g.a
    public void l(int i) {
        this.f642e.setTitle(this.f641d.getString(i));
    }

    @Override // b.b.g.a
    public void m(CharSequence charSequence) {
        this.f642e.setTitle(charSequence);
    }

    @Override // b.b.g.a
    public void n(boolean z) {
        this.f635c = z;
        this.f642e.setTitleOptional(z);
    }

    @Override // b.b.g.i.g.a
    public boolean onMenuItemSelected(b.b.g.i.g gVar, MenuItem menuItem) {
        return this.f643f.d(this, menuItem);
    }

    @Override // b.b.g.i.g.a
    public void onMenuModeChange(b.b.g.i.g gVar) {
        g();
        b.b.h.c cVar = this.f642e.f772e;
        if (cVar != null) {
            cVar.f();
        }
    }
}