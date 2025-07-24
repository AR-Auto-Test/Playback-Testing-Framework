package b.b.g.i;

import android.content.Context;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.SparseArray;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.ListAdapter;
import androidx.appcompat.view.menu.ExpandedMenuView;
import b.b.c.g;
import b.b.g.i.m;
import b.b.g.i.n;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;
import java.util.Objects;

/* compiled from: ListMenuPresenter.java */
/* loaded from: classes.dex */
public class e implements m, AdapterView.OnItemClickListener {

    /* renamed from: b  reason: collision with root package name */
    public Context f713b;

    /* renamed from: c  reason: collision with root package name */
    public LayoutInflater f714c;

    /* renamed from: d  reason: collision with root package name */
    public g f715d;

    /* renamed from: e  reason: collision with root package name */
    public ExpandedMenuView f716e;

    /* renamed from: f  reason: collision with root package name */
    public m.a f717f;

    /* renamed from: g  reason: collision with root package name */
    public a f718g;

    /* compiled from: ListMenuPresenter.java */
    /* loaded from: classes.dex */
    public class a extends BaseAdapter {

        /* renamed from: b  reason: collision with root package name */
        public int f719b = -1;

        public a() {
            a();
        }

        public void a() {
            i expandedItem = e.this.f715d.getExpandedItem();
            if (expandedItem != null) {
                ArrayList<i> nonActionItems = e.this.f715d.getNonActionItems();
                int size = nonActionItems.size();
                for (int i = 0; i < size; i++) {
                    if (nonActionItems.get(i) == expandedItem) {
                        this.f719b = i;
                        return;
                    }
                }
            }
            this.f719b = -1;
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // android.widget.Adapter
        /* renamed from: b */
        public i getItem(int i) {
            ArrayList<i> nonActionItems = e.this.f715d.getNonActionItems();
            Objects.requireNonNull(e.this);
            int i2 = i + 0;
            int i3 = this.f719b;
            if (i3 >= 0 && i2 >= i3) {
                i2++;
            }
            return nonActionItems.get(i2);
        }

        @Override // android.widget.Adapter
        public int getCount() {
            int size = e.this.f715d.getNonActionItems().size();
            Objects.requireNonNull(e.this);
            int i = size + 0;
            return this.f719b < 0 ? i : i - 1;
        }

        @Override // android.widget.Adapter
        public long getItemId(int i) {
            return i;
        }

        @Override // android.widget.Adapter
        public View getView(int i, View view, ViewGroup viewGroup) {
            if (view == null) {
                e eVar = e.this;
                LayoutInflater layoutInflater = eVar.f714c;
                Objects.requireNonNull(eVar);
                view = layoutInflater.inflate(R.layout.abc_list_menu_item_layout, viewGroup, false);
            }
            ((n.a) view).initialize(getItem(i), 0);
            return view;
        }

        @Override // android.widget.BaseAdapter
        public void notifyDataSetChanged() {
            a();
            super.notifyDataSetChanged();
        }
    }

    public e(Context context, int i) {
        this.f713b = context;
        this.f714c = LayoutInflater.from(context);
    }

    public ListAdapter a() {
        if (this.f718g == null) {
            this.f718g = new a();
        }
        return this.f718g;
    }

    @Override // b.b.g.i.m
    public boolean collapseItemActionView(g gVar, i iVar) {
        return false;
    }

    @Override // b.b.g.i.m
    public boolean expandItemActionView(g gVar, i iVar) {
        return false;
    }

    @Override // b.b.g.i.m
    public boolean flagActionItems() {
        return false;
    }

    @Override // b.b.g.i.m
    public int getId() {
        return 0;
    }

    @Override // b.b.g.i.m
    public void initForMenu(Context context, g gVar) {
        if (this.f713b != null) {
            this.f713b = context;
            if (this.f714c == null) {
                this.f714c = LayoutInflater.from(context);
            }
        }
        this.f715d = gVar;
        a aVar = this.f718g;
        if (aVar != null) {
            aVar.notifyDataSetChanged();
        }
    }

    @Override // b.b.g.i.m
    public void onCloseMenu(g gVar, boolean z) {
        m.a aVar = this.f717f;
        if (aVar != null) {
            aVar.onCloseMenu(gVar, z);
        }
    }

    @Override // android.widget.AdapterView.OnItemClickListener
    public void onItemClick(AdapterView<?> adapterView, View view, int i, long j) {
        this.f715d.performItemAction(this.f718g.getItem(i), this, 0);
    }

    @Override // b.b.g.i.m
    public void onRestoreInstanceState(Parcelable parcelable) {
        SparseArray<Parcelable> sparseParcelableArray = ((Bundle) parcelable).getSparseParcelableArray("android:menu:list");
        if (sparseParcelableArray != null) {
            this.f716e.restoreHierarchyState(sparseParcelableArray);
        }
    }

    @Override // b.b.g.i.m
    public Parcelable onSaveInstanceState() {
        if (this.f716e == null) {
            return null;
        }
        Bundle bundle = new Bundle();
        SparseArray<Parcelable> sparseArray = new SparseArray<>();
        ExpandedMenuView expandedMenuView = this.f716e;
        if (expandedMenuView != null) {
            expandedMenuView.saveHierarchyState(sparseArray);
        }
        bundle.putSparseParcelableArray("android:menu:list", sparseArray);
        return bundle;
    }

    @Override // b.b.g.i.m
    public boolean onSubMenuSelected(r rVar) {
        if (rVar.hasVisibleItems()) {
            h hVar = new h(rVar);
            g.a aVar = new g.a(rVar.getContext());
            e eVar = new e(aVar.getContext(), R.layout.abc_list_menu_item_layout);
            hVar.f729d = eVar;
            eVar.f717f = hVar;
            hVar.f727b.addMenuPresenter(eVar);
            aVar.setAdapter(hVar.f729d.a(), hVar);
            View headerView = rVar.getHeaderView();
            if (headerView != null) {
                aVar.setCustomTitle(headerView);
            } else {
                aVar.setIcon(rVar.getHeaderIcon()).setTitle(rVar.getHeaderTitle());
            }
            aVar.setOnKeyListener(hVar);
            b.b.c.g create = aVar.create();
            hVar.f728c = create;
            create.setOnDismissListener(hVar);
            WindowManager.LayoutParams attributes = hVar.f728c.getWindow().getAttributes();
            attributes.type = 1003;
            attributes.flags |= 131072;
            hVar.f728c.show();
            m.a aVar2 = this.f717f;
            if (aVar2 != null) {
                aVar2.a(rVar);
                return true;
            }
            return true;
        }
        return false;
    }

    @Override // b.b.g.i.m
    public void setCallback(m.a aVar) {
        this.f717f = aVar;
    }

    @Override // b.b.g.i.m
    public void updateMenuView(boolean z) {
        a aVar = this.f718g;
        if (aVar != null) {
            aVar.notifyDataSetChanged();
        }
    }
}