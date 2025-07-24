package b.b.g.i;

import android.content.Context;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.view.ActionProvider;
import android.view.ContextMenu;
import android.view.KeyEvent;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.View;

/* compiled from: ActionMenuItem.java */
/* loaded from: classes.dex */
public class a implements b.j.e.a.b {

    /* renamed from: a  reason: collision with root package name */
    public CharSequence f679a;

    /* renamed from: b  reason: collision with root package name */
    public CharSequence f680b;

    /* renamed from: c  reason: collision with root package name */
    public Intent f681c;

    /* renamed from: d  reason: collision with root package name */
    public char f682d;

    /* renamed from: f  reason: collision with root package name */
    public char f684f;

    /* renamed from: h  reason: collision with root package name */
    public Drawable f686h;
    public Context i;
    public CharSequence j;
    public CharSequence k;

    /* renamed from: e  reason: collision with root package name */
    public int f683e = 4096;

    /* renamed from: g  reason: collision with root package name */
    public int f685g = 4096;
    public ColorStateList l = null;
    public PorterDuff.Mode m = null;
    public boolean n = false;
    public boolean o = false;
    public int p = 16;

    public a(Context context, int i, int i2, int i3, CharSequence charSequence) {
        this.i = context;
        this.f679a = charSequence;
    }

    @Override // b.j.e.a.b
    public b.j.e.a.b a(b.j.j.b bVar) {
        throw new UnsupportedOperationException();
    }

    @Override // b.j.e.a.b
    public b.j.j.b b() {
        return null;
    }

    public final void c() {
        Drawable drawable = this.f686h;
        if (drawable != null) {
            if (this.n || this.o) {
                this.f686h = drawable;
                Drawable mutate = drawable.mutate();
                this.f686h = mutate;
                if (this.n) {
                    mutate.setTintList(this.l);
                }
                if (this.o) {
                    this.f686h.setTintMode(this.m);
                }
            }
        }
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public boolean collapseActionView() {
        return false;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public boolean expandActionView() {
        return false;
    }

    @Override // android.view.MenuItem
    public ActionProvider getActionProvider() {
        throw new UnsupportedOperationException();
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public View getActionView() {
        return null;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public int getAlphabeticModifiers() {
        return this.f685g;
    }

    @Override // android.view.MenuItem
    public char getAlphabeticShortcut() {
        return this.f684f;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public CharSequence getContentDescription() {
        return this.j;
    }

    @Override // android.view.MenuItem
    public int getGroupId() {
        return 0;
    }

    @Override // android.view.MenuItem
    public Drawable getIcon() {
        return this.f686h;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public ColorStateList getIconTintList() {
        return this.l;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public PorterDuff.Mode getIconTintMode() {
        return this.m;
    }

    @Override // android.view.MenuItem
    public Intent getIntent() {
        return this.f681c;
    }

    @Override // android.view.MenuItem
    public int getItemId() {
        return 16908332;
    }

    @Override // android.view.MenuItem
    public ContextMenu.ContextMenuInfo getMenuInfo() {
        return null;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public int getNumericModifiers() {
        return this.f683e;
    }

    @Override // android.view.MenuItem
    public char getNumericShortcut() {
        return this.f682d;
    }

    @Override // android.view.MenuItem
    public int getOrder() {
        return 0;
    }

    @Override // android.view.MenuItem
    public SubMenu getSubMenu() {
        return null;
    }

    @Override // android.view.MenuItem
    public CharSequence getTitle() {
        return this.f679a;
    }

    @Override // android.view.MenuItem
    public CharSequence getTitleCondensed() {
        CharSequence charSequence = this.f680b;
        return charSequence != null ? charSequence : this.f679a;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public CharSequence getTooltipText() {
        return this.k;
    }

    @Override // android.view.MenuItem
    public boolean hasSubMenu() {
        return false;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public boolean isActionViewExpanded() {
        return false;
    }

    @Override // android.view.MenuItem
    public boolean isCheckable() {
        return (this.p & 1) != 0;
    }

    @Override // android.view.MenuItem
    public boolean isChecked() {
        return (this.p & 2) != 0;
    }

    @Override // android.view.MenuItem
    public boolean isEnabled() {
        return (this.p & 16) != 0;
    }

    @Override // android.view.MenuItem
    public boolean isVisible() {
        return (this.p & 8) == 0;
    }

    @Override // android.view.MenuItem
    public MenuItem setActionProvider(ActionProvider actionProvider) {
        throw new UnsupportedOperationException();
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setActionView(View view) {
        throw new UnsupportedOperationException();
    }

    @Override // android.view.MenuItem
    public MenuItem setAlphabeticShortcut(char c2) {
        this.f684f = Character.toLowerCase(c2);
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setCheckable(boolean z) {
        this.p = (z ? 1 : 0) | (this.p & (-2));
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setChecked(boolean z) {
        this.p = (z ? 2 : 0) | (this.p & (-3));
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setContentDescription(CharSequence charSequence) {
        this.j = charSequence;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setEnabled(boolean z) {
        this.p = (z ? 16 : 0) | (this.p & (-17));
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setIcon(Drawable drawable) {
        this.f686h = drawable;
        c();
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setIconTintList(ColorStateList colorStateList) {
        this.l = colorStateList;
        this.n = true;
        c();
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setIconTintMode(PorterDuff.Mode mode) {
        this.m = mode;
        this.o = true;
        c();
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setIntent(Intent intent) {
        this.f681c = intent;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setNumericShortcut(char c2) {
        this.f682d = c2;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setOnActionExpandListener(MenuItem.OnActionExpandListener onActionExpandListener) {
        throw new UnsupportedOperationException();
    }

    @Override // android.view.MenuItem
    public MenuItem setOnMenuItemClickListener(MenuItem.OnMenuItemClickListener onMenuItemClickListener) {
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setShortcut(char c2, char c3) {
        this.f682d = c2;
        this.f684f = Character.toLowerCase(c3);
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public void setShowAsAction(int i) {
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setShowAsActionFlags(int i) {
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setTitle(CharSequence charSequence) {
        this.f679a = charSequence;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setTitleCondensed(CharSequence charSequence) {
        this.f680b = charSequence;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setTooltipText(CharSequence charSequence) {
        this.k = charSequence;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setVisible(boolean z) {
        this.p = (this.p & 8) | (z ? 0 : 8);
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setActionView(int i) {
        throw new UnsupportedOperationException();
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setAlphabeticShortcut(char c2, int i) {
        this.f684f = Character.toLowerCase(c2);
        this.f685g = KeyEvent.normalizeMetaState(i);
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public b.j.e.a.b setContentDescription(CharSequence charSequence) {
        this.j = charSequence;
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setNumericShortcut(char c2, int i) {
        this.f682d = c2;
        this.f683e = KeyEvent.normalizeMetaState(i);
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setTitle(int i) {
        this.f679a = this.i.getResources().getString(i);
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public b.j.e.a.b setTooltipText(CharSequence charSequence) {
        this.k = charSequence;
        return this;
    }

    @Override // android.view.MenuItem
    public MenuItem setIcon(int i) {
        Context context = this.i;
        Object obj = b.j.c.a.f2074a;
        this.f686h = context.getDrawable(i);
        c();
        return this;
    }

    @Override // b.j.e.a.b, android.view.MenuItem
    public MenuItem setShortcut(char c2, char c3, int i, int i2) {
        this.f682d = c2;
        this.f683e = KeyEvent.normalizeMetaState(i);
        this.f684f = Character.toLowerCase(c3);
        this.f685g = KeyEvent.normalizeMetaState(i2);
        return this;
    }
}