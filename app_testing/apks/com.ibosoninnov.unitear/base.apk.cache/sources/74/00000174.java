package b.b.g.i;

import android.content.Context;
import android.view.MenuItem;
import android.view.SubMenu;

/* compiled from: BaseMenuWrapper.java */
/* loaded from: classes.dex */
public abstract class c {

    /* renamed from: a  reason: collision with root package name */
    public final Context f694a;

    /* renamed from: b  reason: collision with root package name */
    public b.f.h<b.j.e.a.b, MenuItem> f695b;

    /* renamed from: c  reason: collision with root package name */
    public b.f.h<b.j.e.a.c, SubMenu> f696c;

    public c(Context context) {
        this.f694a = context;
    }

    public final MenuItem c(MenuItem menuItem) {
        if (menuItem instanceof b.j.e.a.b) {
            b.j.e.a.b bVar = (b.j.e.a.b) menuItem;
            if (this.f695b == null) {
                this.f695b = new b.f.h<>();
            }
            MenuItem orDefault = this.f695b.getOrDefault(menuItem, null);
            if (orDefault == null) {
                j jVar = new j(this.f694a, bVar);
                this.f695b.put(bVar, jVar);
                return jVar;
            }
            return orDefault;
        }
        return menuItem;
    }

    public final SubMenu d(SubMenu subMenu) {
        if (subMenu instanceof b.j.e.a.c) {
            b.j.e.a.c cVar = (b.j.e.a.c) subMenu;
            if (this.f696c == null) {
                this.f696c = new b.f.h<>();
            }
            SubMenu subMenu2 = this.f696c.get(cVar);
            if (subMenu2 == null) {
                s sVar = new s(this.f694a, cVar);
                this.f696c.put(cVar, sVar);
                return sVar;
            }
            return subMenu2;
        }
        return subMenu;
    }
}