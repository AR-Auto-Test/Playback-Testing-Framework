package b.b.g.i;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import androidx.appcompat.view.menu.ListMenuItemView;
import b.b.g.i.n;
import java.util.ArrayList;

/* compiled from: MenuAdapter.java */
/* loaded from: classes.dex */
public class f extends BaseAdapter {

    /* renamed from: b  reason: collision with root package name */
    public g f721b;

    /* renamed from: c  reason: collision with root package name */
    public int f722c = -1;

    /* renamed from: d  reason: collision with root package name */
    public boolean f723d;

    /* renamed from: e  reason: collision with root package name */
    public final boolean f724e;

    /* renamed from: f  reason: collision with root package name */
    public final LayoutInflater f725f;

    /* renamed from: g  reason: collision with root package name */
    public final int f726g;

    public f(g gVar, LayoutInflater layoutInflater, boolean z, int i) {
        this.f724e = z;
        this.f725f = layoutInflater;
        this.f721b = gVar;
        this.f726g = i;
        a();
    }

    public void a() {
        i expandedItem = this.f721b.getExpandedItem();
        if (expandedItem != null) {
            ArrayList<i> nonActionItems = this.f721b.getNonActionItems();
            int size = nonActionItems.size();
            for (int i = 0; i < size; i++) {
                if (nonActionItems.get(i) == expandedItem) {
                    this.f722c = i;
                    return;
                }
            }
        }
        this.f722c = -1;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // android.widget.Adapter
    /* renamed from: b */
    public i getItem(int i) {
        ArrayList<i> nonActionItems = this.f724e ? this.f721b.getNonActionItems() : this.f721b.getVisibleItems();
        int i2 = this.f722c;
        if (i2 >= 0 && i >= i2) {
            i++;
        }
        return nonActionItems.get(i);
    }

    @Override // android.widget.Adapter
    public int getCount() {
        ArrayList<i> nonActionItems = this.f724e ? this.f721b.getNonActionItems() : this.f721b.getVisibleItems();
        if (this.f722c < 0) {
            return nonActionItems.size();
        }
        return nonActionItems.size() - 1;
    }

    @Override // android.widget.Adapter
    public long getItemId(int i) {
        return i;
    }

    @Override // android.widget.Adapter
    public View getView(int i, View view, ViewGroup viewGroup) {
        if (view == null) {
            view = this.f725f.inflate(this.f726g, viewGroup, false);
        }
        int i2 = getItem(i).f731b;
        int i3 = i - 1;
        ListMenuItemView listMenuItemView = (ListMenuItemView) view;
        listMenuItemView.setGroupDividerEnabled(this.f721b.isGroupDividerEnabled() && i2 != (i3 >= 0 ? getItem(i3).f731b : i2));
        n.a aVar = (n.a) view;
        if (this.f723d) {
            listMenuItemView.setForceShowIcon(true);
        }
        aVar.initialize(getItem(i), 0);
        return view;
    }

    @Override // android.widget.BaseAdapter
    public void notifyDataSetChanged() {
        a();
        super.notifyDataSetChanged();
    }
}