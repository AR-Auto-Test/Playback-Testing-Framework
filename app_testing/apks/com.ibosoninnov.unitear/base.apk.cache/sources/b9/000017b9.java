package com.google.android.material.internal;

import android.content.Context;
import android.view.SubMenu;
import b.b.g.i.g;
import b.b.g.i.i;

/* loaded from: classes.dex */
public class NavigationMenu extends g {
    public NavigationMenu(Context context) {
        super(context);
    }

    @Override // b.b.g.i.g, android.view.Menu
    public SubMenu addSubMenu(int i, int i2, int i3, CharSequence charSequence) {
        i iVar = (i) addInternal(i, i2, i3, charSequence);
        NavigationSubMenu navigationSubMenu = new NavigationSubMenu(getContext(), this, iVar);
        iVar.o = navigationSubMenu;
        navigationSubMenu.setHeaderTitle(iVar.f734e);
        return navigationSubMenu;
    }
}