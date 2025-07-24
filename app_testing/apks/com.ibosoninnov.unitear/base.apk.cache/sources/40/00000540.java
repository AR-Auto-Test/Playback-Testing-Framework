package b.q.b;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import androidx.fragment.app.Fragment;

/* compiled from: FragmentState.java */
@SuppressLint({"BanParcelableUsage"})
/* loaded from: classes.dex */
public final class v implements Parcelable {
    public static final Parcelable.Creator<v> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final String f2529b;

    /* renamed from: c  reason: collision with root package name */
    public final String f2530c;

    /* renamed from: d  reason: collision with root package name */
    public final boolean f2531d;

    /* renamed from: e  reason: collision with root package name */
    public final int f2532e;

    /* renamed from: f  reason: collision with root package name */
    public final int f2533f;

    /* renamed from: g  reason: collision with root package name */
    public final String f2534g;

    /* renamed from: h  reason: collision with root package name */
    public final boolean f2535h;
    public final boolean i;
    public final boolean j;
    public final Bundle k;
    public final boolean l;
    public final int m;
    public Bundle n;

    /* compiled from: FragmentState.java */
    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<v> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public v createFromParcel(Parcel parcel) {
            return new v(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public v[] newArray(int i) {
            return new v[i];
        }
    }

    public v(Fragment fragment) {
        this.f2529b = fragment.getClass().getName();
        this.f2530c = fragment.mWho;
        this.f2531d = fragment.mFromLayout;
        this.f2532e = fragment.mFragmentId;
        this.f2533f = fragment.mContainerId;
        this.f2534g = fragment.mTag;
        this.f2535h = fragment.mRetainInstance;
        this.i = fragment.mRemoving;
        this.j = fragment.mDetached;
        this.k = fragment.mArguments;
        this.l = fragment.mHidden;
        this.m = fragment.mMaxState.ordinal();
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append("FragmentState{");
        sb.append(this.f2529b);
        sb.append(" (");
        sb.append(this.f2530c);
        sb.append(")}:");
        if (this.f2531d) {
            sb.append(" fromLayout");
        }
        if (this.f2533f != 0) {
            sb.append(" id=0x");
            sb.append(Integer.toHexString(this.f2533f));
        }
        String str = this.f2534g;
        if (str != null && !str.isEmpty()) {
            sb.append(" tag=");
            sb.append(this.f2534g);
        }
        if (this.f2535h) {
            sb.append(" retainInstance");
        }
        if (this.i) {
            sb.append(" removing");
        }
        if (this.j) {
            sb.append(" detached");
        }
        if (this.l) {
            sb.append(" hidden");
        }
        return sb.toString();
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeString(this.f2529b);
        parcel.writeString(this.f2530c);
        parcel.writeInt(this.f2531d ? 1 : 0);
        parcel.writeInt(this.f2532e);
        parcel.writeInt(this.f2533f);
        parcel.writeString(this.f2534g);
        parcel.writeInt(this.f2535h ? 1 : 0);
        parcel.writeInt(this.i ? 1 : 0);
        parcel.writeInt(this.j ? 1 : 0);
        parcel.writeBundle(this.k);
        parcel.writeInt(this.l ? 1 : 0);
        parcel.writeBundle(this.n);
        parcel.writeInt(this.m);
    }

    public v(Parcel parcel) {
        this.f2529b = parcel.readString();
        this.f2530c = parcel.readString();
        this.f2531d = parcel.readInt() != 0;
        this.f2532e = parcel.readInt();
        this.f2533f = parcel.readInt();
        this.f2534g = parcel.readString();
        this.f2535h = parcel.readInt() != 0;
        this.i = parcel.readInt() != 0;
        this.j = parcel.readInt() != 0;
        this.k = parcel.readBundle();
        this.l = parcel.readInt() != 0;
        this.n = parcel.readBundle();
        this.m = parcel.readInt();
    }
}