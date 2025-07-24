package b.b.h;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.widget.HeaderViewListAdapter;
import android.widget.ListAdapter;
import android.widget.PopupWindow;
import androidx.appcompat.view.menu.ListMenuItemView;
import java.lang.reflect.Method;

/* compiled from: MenuPopupWindow.java */
/* loaded from: classes.dex */
public class m0 extends k0 implements l0 {
    public static Method D;
    public l0 E;

    /* compiled from: MenuPopupWindow.java */
    /* loaded from: classes.dex */
    public static class a extends f0 {
        public final int o;
        public final int p;
        public l0 q;
        public MenuItem r;

        public a(Context context, boolean z) {
            super(context, z);
            if (1 == context.getResources().getConfiguration().getLayoutDirection()) {
                this.o = 21;
                this.p = 22;
                return;
            }
            this.o = 22;
            this.p = 21;
        }

        @Override // b.b.h.f0, android.view.View
        public boolean onHoverEvent(MotionEvent motionEvent) {
            int i;
            b.b.g.i.f fVar;
            int pointToPosition;
            int i2;
            if (this.q != null) {
                ListAdapter adapter = getAdapter();
                if (adapter instanceof HeaderViewListAdapter) {
                    HeaderViewListAdapter headerViewListAdapter = (HeaderViewListAdapter) adapter;
                    i = headerViewListAdapter.getHeadersCount();
                    fVar = (b.b.g.i.f) headerViewListAdapter.getWrappedAdapter();
                } else {
                    i = 0;
                    fVar = (b.b.g.i.f) adapter;
                }
                b.b.g.i.i iVar = null;
                if (motionEvent.getAction() != 10 && (pointToPosition = pointToPosition((int) motionEvent.getX(), (int) motionEvent.getY())) != -1 && (i2 = pointToPosition - i) >= 0 && i2 < fVar.getCount()) {
                    iVar = fVar.getItem(i2);
                }
                MenuItem menuItem = this.r;
                if (menuItem != iVar) {
                    b.b.g.i.g gVar = fVar.f721b;
                    if (menuItem != null) {
                        this.q.f(gVar, menuItem);
                    }
                    this.r = iVar;
                    if (iVar != null) {
                        this.q.c(gVar, iVar);
                    }
                }
            }
            return super.onHoverEvent(motionEvent);
        }

        @Override // android.widget.ListView, android.widget.AbsListView, android.view.View, android.view.KeyEvent.Callback
        public boolean onKeyDown(int i, KeyEvent keyEvent) {
            ListMenuItemView listMenuItemView = (ListMenuItemView) getSelectedView();
            if (listMenuItemView != null && i == this.o) {
                if (listMenuItemView.isEnabled() && listMenuItemView.getItemData().hasSubMenu()) {
                    performItemClick(listMenuItemView, getSelectedItemPosition(), getSelectedItemId());
                }
                return true;
            } else if (listMenuItemView != null && i == this.p) {
                setSelection(-1);
                ((b.b.g.i.f) getAdapter()).f721b.close(false);
                return true;
            } else {
                return super.onKeyDown(i, keyEvent);
            }
        }

        public void setHoverListener(l0 l0Var) {
            this.q = l0Var;
        }

        @Override // b.b.h.f0, android.widget.AbsListView
        public /* bridge */ /* synthetic */ void setSelector(Drawable drawable) {
            super.setSelector(drawable);
        }
    }

    static {
        try {
            if (Build.VERSION.SDK_INT <= 28) {
                D = PopupWindow.class.getDeclaredMethod("setTouchModal", Boolean.TYPE);
            }
        } catch (NoSuchMethodException unused) {
            Log.i("MenuPopupWindow", "Could not find method setTouchModal() on PopupWindow. Oh well.");
        }
    }

    public m0(Context context, AttributeSet attributeSet, int i, int i2) {
        super(context, null, i, i2);
    }

    @Override // b.b.h.l0
    public void c(b.b.g.i.g gVar, MenuItem menuItem) {
        l0 l0Var = this.E;
        if (l0Var != null) {
            l0Var.c(gVar, menuItem);
        }
    }

    @Override // b.b.h.l0
    public void f(b.b.g.i.g gVar, MenuItem menuItem) {
        l0 l0Var = this.E;
        if (l0Var != null) {
            l0Var.f(gVar, menuItem);
        }
    }

    @Override // b.b.h.k0
    public f0 o(Context context, boolean z) {
        a aVar = new a(context, z);
        aVar.setHoverListener(this);
        return aVar;
    }
}