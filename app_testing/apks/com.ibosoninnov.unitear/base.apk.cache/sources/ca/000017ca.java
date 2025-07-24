package com.google.android.material.internal;

import android.content.Context;
import b.b.g.i.g;
import b.b.g.i.i;
import b.b.g.i.r;

/* loaded from: classes.dex */
public class NavigationSubMenu extends r {
    public NavigationSubMenu(Context context, NavigationMenu navigationMenu, i iVar) {
        super(context, navigationMenu, iVar);
    }

    @Override // b.b.g.i.g
    public void onItemsChanged(boolean z) {
        super.onItemsChanged(z);
        ((g) getParentMenu()).onItemsChanged(z);
    }
}