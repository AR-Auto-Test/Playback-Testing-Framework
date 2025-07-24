package b.b.g.i;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.View;
import b.b.g.i.g;

/* compiled from: SubMenuBuilder.java */
/* loaded from: classes.dex */
public class r extends g implements SubMenu {
    private i mItem;
    private g mParentMenu;

    public r(Context context, g gVar, i iVar) {
        super(context);
        this.mParentMenu = gVar;
        this.mItem = iVar;
    }

    @Override // b.b.g.i.g
    public boolean collapseItemActionView(i iVar) {
        return this.mParentMenu.collapseItemActionView(iVar);
    }

    @Override // b.b.g.i.g
    public boolean dispatchMenuItemSelected(g gVar, MenuItem menuItem) {
        return super.dispatchMenuItemSelected(gVar, menuItem) || this.mParentMenu.dispatchMenuItemSelected(gVar, menuItem);
    }

    @Override // b.b.g.i.g
    public boolean expandItemActionView(i iVar) {
        return this.mParentMenu.expandItemActionView(iVar);
    }

    @Override // b.b.g.i.g
    public String getActionViewStatesKey() {
        i iVar = this.mItem;
        int i = iVar != null ? iVar.f730a : 0;
        if (i == 0) {
            return null;
        }
        return super.getActionViewStatesKey() + ":" + i;
    }

    @Override // android.view.SubMenu
    public MenuItem getItem() {
        return this.mItem;
    }

    public Menu getParentMenu() {
        return this.mParentMenu;
    }

    @Override // b.b.g.i.g
    public g getRootMenu() {
        return this.mParentMenu.getRootMenu();
    }

    @Override // b.b.g.i.g
    public boolean isGroupDividerEnabled() {
        return this.mParentMenu.isGroupDividerEnabled();
    }

    @Override // b.b.g.i.g
    public boolean isQwertyMode() {
        return this.mParentMenu.isQwertyMode();
    }

    @Override // b.b.g.i.g
    public boolean isShortcutsVisible() {
        return this.mParentMenu.isShortcutsVisible();
    }

    @Override // b.b.g.i.g
    public void setCallback(g.a aVar) {
        this.mParentMenu.setCallback(aVar);
    }

    @Override // b.b.g.i.g, android.view.Menu
    public void setGroupDividerEnabled(boolean z) {
        this.mParentMenu.setGroupDividerEnabled(z);
    }

    @Override // android.view.SubMenu
    public SubMenu setHeaderIcon(Drawable drawable) {
        return (SubMenu) super.setHeaderIconInt(drawable);
    }

    @Override // android.view.SubMenu
    public SubMenu setHeaderTitle(CharSequence charSequence) {
        return (SubMenu) super.setHeaderTitleInt(charSequence);
    }

    @Override // android.view.SubMenu
    public SubMenu setHeaderView(View view) {
        return (SubMenu) super.setHeaderViewInt(view);
    }

    @Override // android.view.SubMenu
    public SubMenu setIcon(Drawable drawable) {
        this.mItem.setIcon(drawable);
        return this;
    }

    @Override // b.b.g.i.g, android.view.Menu
    public void setQwertyMode(boolean z) {
        this.mParentMenu.setQwertyMode(z);
    }

    @Override // b.b.g.i.g
    public void setShortcutsVisible(boolean z) {
        this.mParentMenu.setShortcutsVisible(z);
    }

    @Override // android.view.SubMenu
    public SubMenu setHeaderIcon(int i) {
        return (SubMenu) super.setHeaderIconInt(i);
    }

    @Override // android.view.SubMenu
    public SubMenu setHeaderTitle(int i) {
        return (SubMenu) super.setHeaderTitleInt(i);
    }

    @Override // android.view.SubMenu
    public SubMenu setIcon(int i) {
        this.mItem.setIcon(i);
        return this;
    }
}