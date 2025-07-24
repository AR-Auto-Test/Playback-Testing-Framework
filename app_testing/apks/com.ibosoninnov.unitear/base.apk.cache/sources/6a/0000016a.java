package b.b.g;

import android.content.Context;
import android.view.ActionMode;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import b.b.g.a;
import b.b.g.i.j;
import b.b.g.i.o;
import java.util.ArrayList;

/* compiled from: SupportActionModeWrapper.java */
/* loaded from: classes.dex */
public class e extends ActionMode {

    /* renamed from: a  reason: collision with root package name */
    public final Context f646a;

    /* renamed from: b  reason: collision with root package name */
    public final b.b.g.a f647b;

    /* compiled from: SupportActionModeWrapper.java */
    /* loaded from: classes.dex */
    public static class a implements a.InterfaceC0007a {

        /* renamed from: a  reason: collision with root package name */
        public final ActionMode.Callback f648a;

        /* renamed from: b  reason: collision with root package name */
        public final Context f649b;

        /* renamed from: c  reason: collision with root package name */
        public final ArrayList<e> f650c = new ArrayList<>();

        /* renamed from: d  reason: collision with root package name */
        public final b.f.h<Menu, Menu> f651d = new b.f.h<>();

        public a(Context context, ActionMode.Callback callback) {
            this.f649b = context;
            this.f648a = callback;
        }

        @Override // b.b.g.a.InterfaceC0007a
        public void a(b.b.g.a aVar) {
            this.f648a.onDestroyActionMode(e(aVar));
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean b(b.b.g.a aVar, Menu menu) {
            return this.f648a.onCreateActionMode(e(aVar), f(menu));
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean c(b.b.g.a aVar, Menu menu) {
            return this.f648a.onPrepareActionMode(e(aVar), f(menu));
        }

        @Override // b.b.g.a.InterfaceC0007a
        public boolean d(b.b.g.a aVar, MenuItem menuItem) {
            return this.f648a.onActionItemClicked(e(aVar), new j(this.f649b, (b.j.e.a.b) menuItem));
        }

        public ActionMode e(b.b.g.a aVar) {
            int size = this.f650c.size();
            for (int i = 0; i < size; i++) {
                e eVar = this.f650c.get(i);
                if (eVar != null && eVar.f647b == aVar) {
                    return eVar;
                }
            }
            e eVar2 = new e(this.f649b, aVar);
            this.f650c.add(eVar2);
            return eVar2;
        }

        public final Menu f(Menu menu) {
            Menu orDefault = this.f651d.getOrDefault(menu, null);
            if (orDefault == null) {
                o oVar = new o(this.f649b, (b.j.e.a.a) menu);
                this.f651d.put(menu, oVar);
                return oVar;
            }
            return orDefault;
        }
    }

    public e(Context context, b.b.g.a aVar) {
        this.f646a = context;
        this.f647b = aVar;
    }

    @Override // android.view.ActionMode
    public void finish() {
        this.f647b.a();
    }

    @Override // android.view.ActionMode
    public View getCustomView() {
        return this.f647b.b();
    }

    @Override // android.view.ActionMode
    public Menu getMenu() {
        return new o(this.f646a, (b.j.e.a.a) this.f647b.c());
    }

    @Override // android.view.ActionMode
    public MenuInflater getMenuInflater() {
        return this.f647b.d();
    }

    @Override // android.view.ActionMode
    public CharSequence getSubtitle() {
        return this.f647b.e();
    }

    @Override // android.view.ActionMode
    public Object getTag() {
        return this.f647b.f634b;
    }

    @Override // android.view.ActionMode
    public CharSequence getTitle() {
        return this.f647b.f();
    }

    @Override // android.view.ActionMode
    public boolean getTitleOptionalHint() {
        return this.f647b.f635c;
    }

    @Override // android.view.ActionMode
    public void invalidate() {
        this.f647b.g();
    }

    @Override // android.view.ActionMode
    public boolean isTitleOptional() {
        return this.f647b.h();
    }

    @Override // android.view.ActionMode
    public void setCustomView(View view) {
        this.f647b.i(view);
    }

    @Override // android.view.ActionMode
    public void setSubtitle(CharSequence charSequence) {
        this.f647b.k(charSequence);
    }

    @Override // android.view.ActionMode
    public void setTag(Object obj) {
        this.f647b.f634b = obj;
    }

    @Override // android.view.ActionMode
    public void setTitle(CharSequence charSequence) {
        this.f647b.m(charSequence);
    }

    @Override // android.view.ActionMode
    public void setTitleOptionalHint(boolean z) {
        this.f647b.n(z);
    }

    @Override // android.view.ActionMode
    public void setSubtitle(int i) {
        this.f647b.j(i);
    }

    @Override // android.view.ActionMode
    public void setTitle(int i) {
        this.f647b.l(i);
    }
}