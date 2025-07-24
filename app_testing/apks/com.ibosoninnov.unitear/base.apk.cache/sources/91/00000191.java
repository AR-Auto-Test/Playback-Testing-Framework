package b.b.g.i;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;

/* compiled from: MenuWrapperICS.java */
/* loaded from: classes.dex */
public class o extends c implements Menu {

    /* renamed from: d  reason: collision with root package name */
    public final b.j.e.a.a f759d;

    public o(Context context, b.j.e.a.a aVar) {
        super(context);
        if (aVar != null) {
            this.f759d = aVar;
            return;
        }
        throw new IllegalArgumentException("Wrapped Object can not be null.");
    }

    @Override // android.view.Menu
    public MenuItem add(CharSequence charSequence) {
        return c(this.f759d.add(charSequence));
    }

    @Override // android.view.Menu
    public int addIntentOptions(int i, int i2, int i3, ComponentName componentName, Intent[] intentArr, Intent intent, int i4, MenuItem[] menuItemArr) {
        MenuItem[] menuItemArr2 = menuItemArr != null ? new MenuItem[menuItemArr.length] : null;
        int addIntentOptions = this.f759d.addIntentOptions(i, i2, i3, componentName, intentArr, intent, i4, menuItemArr2);
        if (menuItemArr2 != null) {
            int length = menuItemArr2.length;
            for (int i5 = 0; i5 < length; i5++) {
                menuItemArr[i5] = c(menuItemArr2[i5]);
            }
        }
        return addIntentOptions;
    }

    @Override // android.view.Menu
    public SubMenu addSubMenu(CharSequence charSequence) {
        return d(this.f759d.addSubMenu(charSequence));
    }

    @Override // android.view.Menu
    public void clear() {
        b.f.h<b.j.e.a.b, MenuItem> hVar = this.f695b;
        if (hVar != null) {
            hVar.clear();
        }
        b.f.h<b.j.e.a.c, SubMenu> hVar2 = this.f696c;
        if (hVar2 != null) {
            hVar2.clear();
        }
        this.f759d.clear();
    }

    @Override // android.view.Menu
    public void close() {
        this.f759d.close();
    }

    @Override // android.view.Menu
    public MenuItem findItem(int i) {
        return c(this.f759d.findItem(i));
    }

    @Override // android.view.Menu
    public MenuItem getItem(int i) {
        return c(this.f759d.getItem(i));
    }

    @Override // android.view.Menu
    public boolean hasVisibleItems() {
        return this.f759d.hasVisibleItems();
    }

    @Override // android.view.Menu
    public boolean isShortcutKey(int i, KeyEvent keyEvent) {
        return this.f759d.isShortcutKey(i, keyEvent);
    }

    @Override // android.view.Menu
    public boolean performIdentifierAction(int i, int i2) {
        return this.f759d.performIdentifierAction(i, i2);
    }

    @Override // android.view.Menu
    public boolean performShortcut(int i, KeyEvent keyEvent, int i2) {
        return this.f759d.performShortcut(i, keyEvent, i2);
    }

    @Override // android.view.Menu
    public void removeGroup(int i) {
        if (this.f695b != null) {
            int i2 = 0;
            while (true) {
                b.f.h<b.j.e.a.b, MenuItem> hVar = this.f695b;
                if (i2 >= hVar.f1775h) {
                    break;
                }
                if (hVar.h(i2).getGroupId() == i) {
                    this.f695b.j(i2);
                    i2--;
                }
                i2++;
            }
        }
        this.f759d.removeGroup(i);
    }

    @Override // android.view.Menu
    public void removeItem(int i) {
        if (this.f695b != null) {
            int i2 = 0;
            while (true) {
                b.f.h<b.j.e.a.b, MenuItem> hVar = this.f695b;
                if (i2 >= hVar.f1775h) {
                    break;
                } else if (hVar.h(i2).getItemId() == i) {
                    this.f695b.j(i2);
                    break;
                } else {
                    i2++;
                }
            }
        }
        this.f759d.removeItem(i);
    }

    @Override // android.view.Menu
    public void setGroupCheckable(int i, boolean z, boolean z2) {
        this.f759d.setGroupCheckable(i, z, z2);
    }

    @Override // android.view.Menu
    public void setGroupEnabled(int i, boolean z) {
        this.f759d.setGroupEnabled(i, z);
    }

    @Override // android.view.Menu
    public void setGroupVisible(int i, boolean z) {
        this.f759d.setGroupVisible(i, z);
    }

    @Override // android.view.Menu
    public void setQwertyMode(boolean z) {
        this.f759d.setQwertyMode(z);
    }

    @Override // android.view.Menu
    public int size() {
        return this.f759d.size();
    }

    @Override // android.view.Menu
    public MenuItem add(int i) {
        return c(this.f759d.add(i));
    }

    @Override // android.view.Menu
    public SubMenu addSubMenu(int i) {
        return d(this.f759d.addSubMenu(i));
    }

    @Override // android.view.Menu
    public MenuItem add(int i, int i2, int i3, CharSequence charSequence) {
        return c(this.f759d.add(i, i2, i3, charSequence));
    }

    @Override // android.view.Menu
    public SubMenu addSubMenu(int i, int i2, int i3, CharSequence charSequence) {
        return d(this.f759d.addSubMenu(i, i2, i3, charSequence));
    }

    @Override // android.view.Menu
    public MenuItem add(int i, int i2, int i3, int i4) {
        return c(this.f759d.add(i, i2, i3, i4));
    }

    @Override // android.view.Menu
    public SubMenu addSubMenu(int i, int i2, int i3, int i4) {
        return d(this.f759d.addSubMenu(i, i2, i3, i4));
    }
}