package b.b.h;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.graphics.drawable.Drawable;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.SparseBooleanArray;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import androidx.appcompat.view.menu.ActionMenuItemView;
import androidx.appcompat.widget.ActionMenuView;
import b.b.g.i.m;
import b.b.g.i.n;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;

/* compiled from: ActionMenuPresenter.java */
/* loaded from: classes.dex */
public class c extends b.b.g.i.b {
    public int A;
    public d k;
    public Drawable l;
    public boolean m;
    public boolean n;
    public boolean o;
    public int p;
    public int q;
    public int r;
    public boolean s;
    public int t;
    public final SparseBooleanArray u;
    public e v;
    public a w;
    public RunnableC0010c x;
    public b y;
    public final f z;

    /* compiled from: ActionMenuPresenter.java */
    /* loaded from: classes.dex */
    public class a extends b.b.g.i.l {
        public a(Context context, b.b.g.i.r rVar, View view) {
            super(context, rVar, view, false, R.attr.actionOverflowMenuStyle, 0);
            if (!((b.b.g.i.i) rVar.getItem()).g()) {
                View view2 = c.this.k;
                this.f755f = view2 == null ? (View) c.this.i : view2;
            }
            d(c.this.z);
        }

        @Override // b.b.g.i.l
        public void c() {
            c cVar = c.this;
            cVar.w = null;
            cVar.A = 0;
            super.c();
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    /* loaded from: classes.dex */
    public class b extends ActionMenuItemView.b {
        public b() {
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    /* renamed from: b.b.h.c$c  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class RunnableC0010c implements Runnable {

        /* renamed from: b  reason: collision with root package name */
        public e f810b;

        public RunnableC0010c(e eVar) {
            this.f810b = eVar;
        }

        @Override // java.lang.Runnable
        public void run() {
            b.b.g.i.g gVar = c.this.f689d;
            if (gVar != null) {
                gVar.changeMenuMode();
            }
            View view = (View) c.this.i;
            if (view != null && view.getWindowToken() != null && this.f810b.f()) {
                c.this.v = this.f810b;
            }
            c.this.x = null;
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    /* loaded from: classes.dex */
    public class d extends n implements ActionMenuView.a {

        /* compiled from: ActionMenuPresenter.java */
        /* loaded from: classes.dex */
        public class a extends h0 {
            public a(View view, c cVar) {
                super(view);
            }

            @Override // b.b.h.h0
            public b.b.g.i.p b() {
                e eVar = c.this.v;
                if (eVar == null) {
                    return null;
                }
                return eVar.a();
            }

            @Override // b.b.h.h0
            public boolean c() {
                c.this.f();
                return true;
            }

            @Override // b.b.h.h0
            public boolean d() {
                c cVar = c.this;
                if (cVar.x != null) {
                    return false;
                }
                cVar.c();
                return true;
            }
        }

        public d(Context context) {
            super(context, null, R.attr.actionOverflowButtonStyle);
            setClickable(true);
            setFocusable(true);
            setVisibility(0);
            setEnabled(true);
            b.b.a.n(this, getContentDescription());
            setOnTouchListener(new a(this, c.this));
        }

        @Override // androidx.appcompat.widget.ActionMenuView.a
        public boolean a() {
            return false;
        }

        @Override // androidx.appcompat.widget.ActionMenuView.a
        public boolean b() {
            return false;
        }

        @Override // android.view.View
        public boolean performClick() {
            if (super.performClick()) {
                return true;
            }
            playSoundEffect(0);
            c.this.f();
            return true;
        }

        @Override // android.widget.ImageView
        public boolean setFrame(int i, int i2, int i3, int i4) {
            boolean frame = super.setFrame(i, i2, i3, i4);
            Drawable drawable = getDrawable();
            Drawable background = getBackground();
            if (drawable != null && background != null) {
                int width = getWidth();
                int height = getHeight();
                int max = Math.max(width, height) / 2;
                int paddingLeft = (width + (getPaddingLeft() - getPaddingRight())) / 2;
                int paddingTop = (height + (getPaddingTop() - getPaddingBottom())) / 2;
                background.setHotspotBounds(paddingLeft - max, paddingTop - max, paddingLeft + max, paddingTop + max);
            }
            return frame;
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    /* loaded from: classes.dex */
    public class e extends b.b.g.i.l {
        public e(Context context, b.b.g.i.g gVar, View view, boolean z) {
            super(context, gVar, view, z, R.attr.actionOverflowMenuStyle, 0);
            this.f756g = 8388613;
            d(c.this.z);
        }

        @Override // b.b.g.i.l
        public void c() {
            b.b.g.i.g gVar = c.this.f689d;
            if (gVar != null) {
                gVar.close();
            }
            c.this.v = null;
            super.c();
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    /* loaded from: classes.dex */
    public class f implements m.a {
        public f() {
        }

        @Override // b.b.g.i.m.a
        public boolean a(b.b.g.i.g gVar) {
            c cVar = c.this;
            if (gVar == cVar.f689d) {
                return false;
            }
            cVar.A = ((b.b.g.i.r) gVar).getItem().getItemId();
            m.a aVar = c.this.f691f;
            if (aVar != null) {
                return aVar.a(gVar);
            }
            return false;
        }

        @Override // b.b.g.i.m.a
        public void onCloseMenu(b.b.g.i.g gVar, boolean z) {
            if (gVar instanceof b.b.g.i.r) {
                gVar.getRootMenu().close(false);
            }
            m.a aVar = c.this.f691f;
            if (aVar != null) {
                aVar.onCloseMenu(gVar, z);
            }
        }
    }

    /* compiled from: ActionMenuPresenter.java */
    @SuppressLint({"BanParcelableUsage"})
    /* loaded from: classes.dex */
    public static class g implements Parcelable {
        public static final Parcelable.Creator<g> CREATOR = new a();

        /* renamed from: b  reason: collision with root package name */
        public int f814b;

        /* compiled from: ActionMenuPresenter.java */
        /* loaded from: classes.dex */
        public class a implements Parcelable.Creator<g> {
            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.Creator
            public g createFromParcel(Parcel parcel) {
                return new g(parcel);
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
            @Override // android.os.Parcelable.Creator
            public g[] newArray(int i) {
                return new g[i];
            }
        }

        public g() {
        }

        @Override // android.os.Parcelable
        public int describeContents() {
            return 0;
        }

        @Override // android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeInt(this.f814b);
        }

        public g(Parcel parcel) {
            this.f814b = parcel.readInt();
        }
    }

    public c(Context context) {
        super(context, R.layout.abc_action_menu_layout, R.layout.abc_action_menu_item_layout);
        this.u = new SparseBooleanArray();
        this.z = new f();
    }

    public boolean a() {
        return c() | d();
    }

    public View b(b.b.g.i.i iVar, View view, ViewGroup viewGroup) {
        n.a aVar;
        View actionView = iVar.getActionView();
        if (actionView == null || iVar.f()) {
            if (view instanceof n.a) {
                aVar = (n.a) view;
            } else {
                aVar = (n.a) this.f690e.inflate(this.f693h, viewGroup, false);
            }
            aVar.initialize(iVar, 0);
            ActionMenuItemView actionMenuItemView = (ActionMenuItemView) aVar;
            actionMenuItemView.setItemInvoker((ActionMenuView) this.i);
            if (this.y == null) {
                this.y = new b();
            }
            actionMenuItemView.setPopupCallback(this.y);
            actionView = (View) aVar;
        }
        actionView.setVisibility(iVar.C ? 8 : 0);
        ActionMenuView actionMenuView = (ActionMenuView) viewGroup;
        ViewGroup.LayoutParams layoutParams = actionView.getLayoutParams();
        if (!actionMenuView.checkLayoutParams(layoutParams)) {
            actionView.setLayoutParams(actionMenuView.generateLayoutParams(layoutParams));
        }
        return actionView;
    }

    public boolean c() {
        b.b.g.i.n nVar;
        RunnableC0010c runnableC0010c = this.x;
        if (runnableC0010c != null && (nVar = this.i) != null) {
            ((View) nVar).removeCallbacks(runnableC0010c);
            this.x = null;
            return true;
        }
        e eVar = this.v;
        if (eVar != null) {
            if (eVar.b()) {
                eVar.j.dismiss();
            }
            return true;
        }
        return false;
    }

    public boolean d() {
        a aVar = this.w;
        if (aVar != null) {
            if (aVar.b()) {
                aVar.j.dismiss();
                return true;
            }
            return true;
        }
        return false;
    }

    public boolean e() {
        e eVar = this.v;
        return eVar != null && eVar.b();
    }

    public boolean f() {
        b.b.g.i.g gVar;
        if (!this.n || e() || (gVar = this.f689d) == null || this.i == null || this.x != null || gVar.getNonActionItems().isEmpty()) {
            return false;
        }
        RunnableC0010c runnableC0010c = new RunnableC0010c(new e(this.f688c, this.f689d, this.k, true));
        this.x = runnableC0010c;
        ((View) this.i).post(runnableC0010c);
        return true;
    }

    @Override // b.b.g.i.m
    public boolean flagActionItems() {
        int i;
        ArrayList<b.b.g.i.i> arrayList;
        int i2;
        boolean z;
        b.b.g.i.g gVar = this.f689d;
        if (gVar != null) {
            arrayList = gVar.getVisibleItems();
            i = arrayList.size();
        } else {
            i = 0;
            arrayList = null;
        }
        int i3 = this.r;
        int i4 = this.q;
        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, 0);
        ViewGroup viewGroup = (ViewGroup) this.i;
        int i5 = 0;
        boolean z2 = false;
        int i6 = 0;
        int i7 = 0;
        while (true) {
            i2 = 2;
            z = true;
            if (i5 >= i) {
                break;
            }
            b.b.g.i.i iVar = arrayList.get(i5);
            int i8 = iVar.y;
            if ((i8 & 2) == 2) {
                i7++;
            } else if ((i8 & 1) == 1) {
                i6++;
            } else {
                z2 = true;
            }
            if (this.s && iVar.C) {
                i3 = 0;
            }
            i5++;
        }
        if (this.n && (z2 || i6 + i7 > i3)) {
            i3--;
        }
        int i9 = i3 - i7;
        SparseBooleanArray sparseBooleanArray = this.u;
        sparseBooleanArray.clear();
        int i10 = 0;
        int i11 = 0;
        while (i10 < i) {
            b.b.g.i.i iVar2 = arrayList.get(i10);
            int i12 = iVar2.y;
            if ((i12 & 2) == i2 ? z : false) {
                View b2 = b(iVar2, null, viewGroup);
                b2.measure(makeMeasureSpec, makeMeasureSpec);
                int measuredWidth = b2.getMeasuredWidth();
                i4 -= measuredWidth;
                if (i11 == 0) {
                    i11 = measuredWidth;
                }
                int i13 = iVar2.f731b;
                if (i13 != 0) {
                    sparseBooleanArray.put(i13, z);
                }
                iVar2.l(z);
            } else if ((i12 & 1) == z ? z : false) {
                int i14 = iVar2.f731b;
                boolean z3 = sparseBooleanArray.get(i14);
                boolean z4 = ((i9 > 0 || z3) && i4 > 0) ? z : false;
                if (z4) {
                    View b3 = b(iVar2, null, viewGroup);
                    b3.measure(makeMeasureSpec, makeMeasureSpec);
                    int measuredWidth2 = b3.getMeasuredWidth();
                    i4 -= measuredWidth2;
                    if (i11 == 0) {
                        i11 = measuredWidth2;
                    }
                    z4 &= i4 + i11 > 0;
                }
                if (z4 && i14 != 0) {
                    sparseBooleanArray.put(i14, true);
                } else if (z3) {
                    sparseBooleanArray.put(i14, false);
                    for (int i15 = 0; i15 < i10; i15++) {
                        b.b.g.i.i iVar3 = arrayList.get(i15);
                        if (iVar3.f731b == i14) {
                            if (iVar3.g()) {
                                i9++;
                            }
                            iVar3.l(false);
                        }
                    }
                }
                if (z4) {
                    i9--;
                }
                iVar2.l(z4);
            } else {
                iVar2.l(false);
                i10++;
                i2 = 2;
                z = true;
            }
            i10++;
            i2 = 2;
            z = true;
        }
        return z;
    }

    @Override // b.b.g.i.m
    public void initForMenu(Context context, b.b.g.i.g gVar) {
        this.f688c = context;
        LayoutInflater.from(context);
        this.f689d = gVar;
        Resources resources = context.getResources();
        if (!this.o) {
            this.n = true;
        }
        int i = 2;
        this.p = context.getResources().getDisplayMetrics().widthPixels / 2;
        Configuration configuration = context.getResources().getConfiguration();
        int i2 = configuration.screenWidthDp;
        int i3 = configuration.screenHeightDp;
        if (configuration.smallestScreenWidthDp > 600 || i2 > 600 || ((i2 > 960 && i3 > 720) || (i2 > 720 && i3 > 960))) {
            i = 5;
        } else if (i2 >= 500 || ((i2 > 640 && i3 > 480) || (i2 > 480 && i3 > 640))) {
            i = 4;
        } else if (i2 >= 360) {
            i = 3;
        }
        this.r = i;
        int i4 = this.p;
        if (this.n) {
            if (this.k == null) {
                d dVar = new d(this.f687b);
                this.k = dVar;
                if (this.m) {
                    dVar.setImageDrawable(this.l);
                    this.l = null;
                    this.m = false;
                }
                int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(0, 0);
                this.k.measure(makeMeasureSpec, makeMeasureSpec);
            }
            i4 -= this.k.getMeasuredWidth();
        } else {
            this.k = null;
        }
        this.q = i4;
        this.t = (int) (resources.getDisplayMetrics().density * 56.0f);
    }

    @Override // b.b.g.i.m
    public void onCloseMenu(b.b.g.i.g gVar, boolean z) {
        a();
        m.a aVar = this.f691f;
        if (aVar != null) {
            aVar.onCloseMenu(gVar, z);
        }
    }

    @Override // b.b.g.i.m
    public void onRestoreInstanceState(Parcelable parcelable) {
        int i;
        MenuItem findItem;
        if ((parcelable instanceof g) && (i = ((g) parcelable).f814b) > 0 && (findItem = this.f689d.findItem(i)) != null) {
            onSubMenuSelected((b.b.g.i.r) findItem.getSubMenu());
        }
    }

    @Override // b.b.g.i.m
    public Parcelable onSaveInstanceState() {
        g gVar = new g();
        gVar.f814b = this.A;
        return gVar;
    }

    @Override // b.b.g.i.m
    public boolean onSubMenuSelected(b.b.g.i.r rVar) {
        boolean z = false;
        if (rVar.hasVisibleItems()) {
            b.b.g.i.r rVar2 = rVar;
            while (rVar2.getParentMenu() != this.f689d) {
                rVar2 = (b.b.g.i.r) rVar2.getParentMenu();
            }
            MenuItem item = rVar2.getItem();
            ViewGroup viewGroup = (ViewGroup) this.i;
            View view = null;
            if (viewGroup != null) {
                int childCount = viewGroup.getChildCount();
                int i = 0;
                while (true) {
                    if (i >= childCount) {
                        break;
                    }
                    View childAt = viewGroup.getChildAt(i);
                    if ((childAt instanceof n.a) && ((n.a) childAt).getItemData() == item) {
                        view = childAt;
                        break;
                    }
                    i++;
                }
            }
            if (view == null) {
                return false;
            }
            this.A = rVar.getItem().getItemId();
            int size = rVar.size();
            int i2 = 0;
            while (true) {
                if (i2 >= size) {
                    break;
                }
                MenuItem item2 = rVar.getItem(i2);
                if (item2.isVisible() && item2.getIcon() != null) {
                    z = true;
                    break;
                }
                i2++;
            }
            a aVar = new a(this.f688c, rVar, view);
            this.w = aVar;
            aVar.f757h = z;
            b.b.g.i.k kVar = aVar.j;
            if (kVar != null) {
                kVar.e(z);
            }
            if (this.w.f()) {
                m.a aVar2 = this.f691f;
                if (aVar2 != null) {
                    aVar2.a(rVar);
                }
                return true;
            }
            throw new IllegalStateException("MenuPopupHelper cannot be used without an anchor");
        }
        return false;
    }

    @Override // b.b.g.i.m
    public void updateMenuView(boolean z) {
        int i;
        boolean z2;
        ViewGroup viewGroup = (ViewGroup) this.i;
        boolean z3 = false;
        if (viewGroup != null) {
            b.b.g.i.g gVar = this.f689d;
            if (gVar != null) {
                gVar.flagActionItems();
                ArrayList<b.b.g.i.i> visibleItems = this.f689d.getVisibleItems();
                int size = visibleItems.size();
                i = 0;
                for (int i2 = 0; i2 < size; i2++) {
                    b.b.g.i.i iVar = visibleItems.get(i2);
                    if (iVar.g()) {
                        View childAt = viewGroup.getChildAt(i);
                        b.b.g.i.i itemData = childAt instanceof n.a ? ((n.a) childAt).getItemData() : null;
                        View b2 = b(iVar, childAt, viewGroup);
                        if (iVar != itemData) {
                            b2.setPressed(false);
                            b2.jumpDrawablesToCurrentState();
                        }
                        if (b2 != childAt) {
                            ViewGroup viewGroup2 = (ViewGroup) b2.getParent();
                            if (viewGroup2 != null) {
                                viewGroup2.removeView(b2);
                            }
                            ((ViewGroup) this.i).addView(b2, i);
                        }
                        i++;
                    }
                }
            } else {
                i = 0;
            }
            while (i < viewGroup.getChildCount()) {
                if (viewGroup.getChildAt(i) == this.k) {
                    z2 = false;
                } else {
                    viewGroup.removeViewAt(i);
                    z2 = true;
                }
                if (!z2) {
                    i++;
                }
            }
        }
        ((View) this.i).requestLayout();
        b.b.g.i.g gVar2 = this.f689d;
        if (gVar2 != null) {
            ArrayList<b.b.g.i.i> actionItems = gVar2.getActionItems();
            int size2 = actionItems.size();
            for (int i3 = 0; i3 < size2; i3++) {
                b.j.j.b bVar = actionItems.get(i3).A;
            }
        }
        b.b.g.i.g gVar3 = this.f689d;
        ArrayList<b.b.g.i.i> nonActionItems = gVar3 != null ? gVar3.getNonActionItems() : null;
        if (this.n && nonActionItems != null) {
            int size3 = nonActionItems.size();
            if (size3 == 1) {
                z3 = !nonActionItems.get(0).C;
            } else if (size3 > 0) {
                z3 = true;
            }
        }
        if (z3) {
            if (this.k == null) {
                this.k = new d(this.f687b);
            }
            ViewGroup viewGroup3 = (ViewGroup) this.k.getParent();
            if (viewGroup3 != this.i) {
                if (viewGroup3 != null) {
                    viewGroup3.removeView(this.k);
                }
                ActionMenuView actionMenuView = (ActionMenuView) this.i;
                d dVar = this.k;
                ActionMenuView.c generateDefaultLayoutParams = actionMenuView.generateDefaultLayoutParams();
                generateDefaultLayoutParams.f120c = true;
                actionMenuView.addView(dVar, generateDefaultLayoutParams);
            }
        } else {
            d dVar2 = this.k;
            if (dVar2 != null) {
                ViewParent parent = dVar2.getParent();
                b.b.g.i.n nVar = this.i;
                if (parent == nVar) {
                    ((ViewGroup) nVar).removeView(this.k);
                }
            }
        }
        ((ActionMenuView) this.i).setOverflowReserved(this.n);
    }
}