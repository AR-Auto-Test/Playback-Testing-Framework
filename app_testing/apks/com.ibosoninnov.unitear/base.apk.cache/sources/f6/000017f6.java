package com.google.android.material.navigation;

import android.content.Context;
import android.os.Parcel;
import android.os.Parcelable;
import android.view.ViewGroup;
import b.b.g.i.g;
import b.b.g.i.i;
import b.b.g.i.m;
import b.b.g.i.n;
import b.b.g.i.r;
import com.google.android.material.badge.BadgeUtils;
import com.google.android.material.internal.ParcelableSparseArray;

/* loaded from: classes.dex */
public class NavigationBarPresenter implements m {
    private int id;
    private g menu;
    private NavigationBarMenuView menuView;
    private boolean updateSuspended = false;

    /* loaded from: classes.dex */
    public static class SavedState implements Parcelable {
        public static final Parcelable.Creator<SavedState> CREATOR = new Parcelable.Creator<SavedState>() { // from class: com.google.android.material.navigation.NavigationBarPresenter.SavedState.1
            /* JADX DEBUG: Method merged with bridge method */
            /* JADX WARN: Can't rename method to resolve collision */
            @Override // android.os.Parcelable.Creator
            public SavedState createFromParcel(Parcel parcel) {
                return new SavedState(parcel);
            }

            /* JADX DEBUG: Method merged with bridge method */
            /* JADX WARN: Can't rename method to resolve collision */
            @Override // android.os.Parcelable.Creator
            public SavedState[] newArray(int i) {
                return new SavedState[i];
            }
        };
        public ParcelableSparseArray badgeSavedStates;
        public int selectedItemId;

        public SavedState() {
        }

        @Override // android.os.Parcelable
        public int describeContents() {
            return 0;
        }

        @Override // android.os.Parcelable
        public void writeToParcel(Parcel parcel, int i) {
            parcel.writeInt(this.selectedItemId);
            parcel.writeParcelable(this.badgeSavedStates, 0);
        }

        public SavedState(Parcel parcel) {
            this.selectedItemId = parcel.readInt();
            this.badgeSavedStates = (ParcelableSparseArray) parcel.readParcelable(getClass().getClassLoader());
        }
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
        return this.id;
    }

    public n getMenuView(ViewGroup viewGroup) {
        return this.menuView;
    }

    @Override // b.b.g.i.m
    public void initForMenu(Context context, g gVar) {
        this.menu = gVar;
        this.menuView.initialize(gVar);
    }

    @Override // b.b.g.i.m
    public void onCloseMenu(g gVar, boolean z) {
    }

    @Override // b.b.g.i.m
    public void onRestoreInstanceState(Parcelable parcelable) {
        if (parcelable instanceof SavedState) {
            SavedState savedState = (SavedState) parcelable;
            this.menuView.tryRestoreSelectedItemId(savedState.selectedItemId);
            this.menuView.setBadgeDrawables(BadgeUtils.createBadgeDrawablesFromSavedStates(this.menuView.getContext(), savedState.badgeSavedStates));
        }
    }

    @Override // b.b.g.i.m
    public Parcelable onSaveInstanceState() {
        SavedState savedState = new SavedState();
        savedState.selectedItemId = this.menuView.getSelectedItemId();
        savedState.badgeSavedStates = BadgeUtils.createParcelableBadgeStates(this.menuView.getBadgeDrawables());
        return savedState;
    }

    @Override // b.b.g.i.m
    public boolean onSubMenuSelected(r rVar) {
        return false;
    }

    @Override // b.b.g.i.m
    public void setCallback(m.a aVar) {
    }

    public void setId(int i) {
        this.id = i;
    }

    public void setMenuView(NavigationBarMenuView navigationBarMenuView) {
        this.menuView = navigationBarMenuView;
    }

    public void setUpdateSuspended(boolean z) {
        this.updateSuspended = z;
    }

    @Override // b.b.g.i.m
    public void updateMenuView(boolean z) {
        if (this.updateSuspended) {
            return;
        }
        if (z) {
            this.menuView.buildMenuView();
        } else {
            this.menuView.updateMenuView();
        }
    }
}