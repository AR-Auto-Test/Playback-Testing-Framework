package b.b.g.i;

import android.content.Context;
import android.graphics.Rect;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.FrameLayout;
import android.widget.HeaderViewListAdapter;
import android.widget.ListAdapter;
import android.widget.PopupWindow;

/* compiled from: MenuPopup.java */
/* loaded from: classes.dex */
public abstract class k implements p, m, AdapterView.OnItemClickListener {

    /* renamed from: b  reason: collision with root package name */
    public Rect f749b;

    public static int c(ListAdapter listAdapter, ViewGroup viewGroup, Context context, int i) {
        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, 0);
        int makeMeasureSpec2 = View.MeasureSpec.makeMeasureSpec(0, 0);
        f fVar = (f) listAdapter;
        int count = fVar.getCount();
        int i2 = 0;
        int i3 = 0;
        FrameLayout frameLayout = null;
        View view = null;
        for (int i4 = 0; i4 < count; i4++) {
            int itemViewType = fVar.getItemViewType(i4);
            if (itemViewType != i3) {
                view = null;
                i3 = itemViewType;
            }
            if (frameLayout == null) {
                frameLayout = new FrameLayout(context);
            }
            view = fVar.getView(i4, view, frameLayout);
            view.measure(makeMeasureSpec, makeMeasureSpec2);
            int measuredWidth = view.getMeasuredWidth();
            if (measuredWidth >= i) {
                return i;
            }
            if (measuredWidth > i2) {
                i2 = measuredWidth;
            }
        }
        return i2;
    }

    public static boolean l(g gVar) {
        int size = gVar.size();
        for (int i = 0; i < size; i++) {
            MenuItem item = gVar.getItem(i);
            if (item.isVisible() && item.getIcon() != null) {
                return true;
            }
        }
        return false;
    }

    public abstract void b(g gVar);

    @Override // b.b.g.i.m
    public boolean collapseItemActionView(g gVar, i iVar) {
        return false;
    }

    public abstract void d(View view);

    public abstract void e(boolean z);

    @Override // b.b.g.i.m
    public boolean expandItemActionView(g gVar, i iVar) {
        return false;
    }

    public abstract void f(int i);

    public abstract void g(int i);

    @Override // b.b.g.i.m
    public int getId() {
        return 0;
    }

    public abstract void i(PopupWindow.OnDismissListener onDismissListener);

    @Override // b.b.g.i.m
    public void initForMenu(Context context, g gVar) {
    }

    public abstract void j(boolean z);

    public abstract void k(int i);

    @Override // android.widget.AdapterView.OnItemClickListener
    public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
        f fVar;
        ListAdapter listAdapter = (ListAdapter) adapterView.getAdapter();
        if (listAdapter instanceof HeaderViewListAdapter) {
            fVar = (f) ((HeaderViewListAdapter) listAdapter).getWrappedAdapter();
        } else {
            fVar = (f) listAdapter;
        }
        fVar.f721b.performItemAction((MenuItem) listAdapter.getItem(i), this, (this instanceof d) ^ true ? 0 : 4);
    }
}