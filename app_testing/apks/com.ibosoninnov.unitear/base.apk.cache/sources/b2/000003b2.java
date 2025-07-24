package b.d0.b;

import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import android.util.SparseArray;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityNodeInfo;
import androidx.recyclerview.widget.RecyclerView;

/* compiled from: ViewPager2.java */
/* loaded from: classes.dex */
public final class a extends ViewGroup {

    /* renamed from: b  reason: collision with root package name */
    public int f1720b;

    /* renamed from: c  reason: collision with root package name */
    public int f1721c;

    /* renamed from: d  reason: collision with root package name */
    public Parcelable f1722d;

    /* renamed from: e  reason: collision with root package name */
    public int f1723e;

    /* compiled from: ViewPager2.java */
    /* renamed from: b.d0.b.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static abstract class AbstractC0024a {
        public abstract void onPageScrollStateChanged(int i);

        public abstract void onPageScrolled(int i, float f2, int i2);

        public abstract void onPageSelected(int i);
    }

    /* compiled from: ViewPager2.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public final void a() {
        RecyclerView.g adapter;
        if (this.f1721c == -1 || (adapter = getAdapter()) == null) {
            return;
        }
        Parcelable parcelable = this.f1722d;
        if (parcelable != null) {
            if (adapter instanceof b.d0.a.a) {
                ((b.d0.a.a) adapter).a(parcelable);
            }
            this.f1722d = null;
        }
        this.f1720b = Math.max(0, Math.min(this.f1721c, adapter.getItemCount() - 1));
        this.f1721c = -1;
        throw null;
    }

    @Override // android.view.View
    public boolean canScrollHorizontally(int i) {
        throw null;
    }

    @Override // android.view.View
    public boolean canScrollVertically(int i) {
        throw null;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchRestoreInstanceState(SparseArray<Parcelable> sparseArray) {
        Parcelable parcelable = sparseArray.get(getId());
        if (!(parcelable instanceof c)) {
            super.dispatchRestoreInstanceState(sparseArray);
            a();
            return;
        }
        int i = ((c) parcelable).f1724b;
        throw null;
    }

    @Override // android.view.ViewGroup, android.view.View
    public CharSequence getAccessibilityClassName() {
        throw null;
    }

    public RecyclerView.g getAdapter() {
        throw null;
    }

    public int getCurrentItem() {
        return this.f1720b;
    }

    public int getItemDecorationCount() {
        throw null;
    }

    public int getOffscreenPageLimit() {
        return this.f1723e;
    }

    public int getOrientation() {
        throw null;
    }

    public int getPageSize() {
        if (getOrientation() == 0) {
            throw null;
        }
        throw null;
    }

    public int getScrollState() {
        throw null;
    }

    @Override // android.view.View
    public void onInitializeAccessibilityNodeInfo(AccessibilityNodeInfo accessibilityNodeInfo) {
        super.onInitializeAccessibilityNodeInfo(accessibilityNodeInfo);
        throw null;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        throw null;
    }

    @Override // android.view.View
    public void onMeasure(int i, int i2) {
        measureChild(null, i, i2);
        throw null;
    }

    @Override // android.view.View
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (!(parcelable instanceof c)) {
            super.onRestoreInstanceState(parcelable);
            return;
        }
        c cVar = (c) parcelable;
        super.onRestoreInstanceState(cVar.getSuperState());
        this.f1721c = cVar.f1725c;
        this.f1722d = cVar.f1726d;
    }

    @Override // android.view.View
    public Parcelable onSaveInstanceState() {
        new c(super.onSaveInstanceState());
        throw null;
    }

    @Override // android.view.ViewGroup
    public void onViewAdded(View view) {
        throw new IllegalStateException(a.class.getSimpleName() + " does not support direct child views");
    }

    @Override // android.view.View
    public boolean performAccessibilityAction(int i, Bundle bundle) {
        throw null;
    }

    public void setAdapter(RecyclerView.g gVar) {
        throw null;
    }

    public void setCurrentItem(int i) {
        throw null;
    }

    @Override // android.view.View
    public void setLayoutDirection(int i) {
        super.setLayoutDirection(i);
        throw null;
    }

    public void setOffscreenPageLimit(int i) {
        if (i < 1 && i != -1) {
            throw new IllegalArgumentException("Offscreen page limit must be OFFSCREEN_PAGE_LIMIT_DEFAULT or a number > 0");
        }
        this.f1723e = i;
        throw null;
    }

    public void setOrientation(int i) {
        throw null;
    }

    public void setPageTransformer(b bVar) {
        if (bVar != null) {
            throw null;
        }
        throw null;
    }

    public void setUserInputEnabled(boolean z) {
        throw null;
    }

    /* compiled from: ViewPager2.java */
    /* loaded from: classes.dex */
    public static class c extends View.BaseSavedState {
        public static final Parcelable.Creator<c> CREATOR = new C0025a();

        /* renamed from: b  reason: collision with root package name */
        public int f1724b;

        /* renamed from: c  reason: collision with root package name */
        public int f1725c;

        /* renamed from: d  reason: collision with root package name */
        public Parcelable f1726d;

        /* compiled from: ViewPager2.java */
        /* renamed from: b.d0.b.a$c$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public static class C0025a implements Parcelable.ClassLoaderCreator<c> {
            @Override // android.os.Parcelable.Creator
            public Object createFromParcel(Parcel parcel) {
                return new c(parcel, null);
            }

            @Override // android.os.Parcelable.Creator
            public Object[] newArray(int i) {
                return new c[i];
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // android.os.Parcelable.ClassLoaderCreator
            public c createFromParcel(Parcel parcel, ClassLoader classLoader) {
                return new c(parcel, classLoader);
            }
        }

        public c(Parcel parcel, ClassLoader classLoader) {
            super(parcel, classLoader);
            this.f1724b = parcel.readInt();
            this.f1725c = parcel.readInt();
            this.f1726d = parcel.readParcelable(classLoader);
        }

        @Override // android.view.View.BaseSavedState, android.view.AbsSavedState, android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            super.writeToParcel(parcel, i);
            parcel.writeInt(this.f1724b);
            parcel.writeInt(this.f1725c);
            parcel.writeParcelable(this.f1726d, i);
        }

        public c(Parcelable parcelable) {
            super(parcelable);
        }
    }
}