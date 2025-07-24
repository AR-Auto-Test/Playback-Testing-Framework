package com.google.android.material.navigation;

import android.content.Context;
import android.view.MenuItem;
import android.view.SubMenu;
import b.b.g.i.g;
import b.b.g.i.i;
import c.b.a.a.a;

/* loaded from: classes.dex */
public final class NavigationBarMenu extends g {
    private final int maxItemCount;
    private final Class<?> viewClass;

    public NavigationBarMenu(Context context, Class<?> cls, int i) {
        super(context);
        this.viewClass = cls;
        this.maxItemCount = i;
    }

    @Override // b.b.g.i.g
    public MenuItem addInternal(int i, int i2, int i3, CharSequence charSequence) {
        if (size() + 1 <= this.maxItemCount) {
            stopDispatchingItemsChanged();
            MenuItem addInternal = super.addInternal(i, i2, i3, charSequence);
            if (addInternal instanceof i) {
                ((i) addInternal).k(true);
            }
            startDispatchingItemsChanged();
            return addInternal;
        }
        String simpleName = this.viewClass.getSimpleName();
        StringBuilder B = a.B("Maximum number of items supported by ", simpleName, " is ");
        B.append(this.maxItemCount);
        B.append(". Limit can be checked with ");
        B.append(simpleName);
        B.append("#getMaxItemCount()");
        throw new IllegalArgumentException(B.toString());
    }

    @Override // b.b.g.i.g, android.view.Menu
    public SubMenu addSubMenu(int i, int i2, int i3, CharSequence charSequence) {
        throw new UnsupportedOperationException(this.viewClass.getSimpleName() + " does not support submenus");
    }

    public int getMaxItemCount() {
        return this.maxItemCount;
    }
}