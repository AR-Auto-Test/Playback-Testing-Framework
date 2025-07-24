package b.q.b;

import android.annotation.SuppressLint;
import android.os.Parcel;
import android.os.Parcelable;
import java.util.ArrayList;

/* compiled from: FragmentManagerState.java */
@SuppressLint({"BanParcelableUsage"})
/* loaded from: classes.dex */
public final class t implements Parcelable {
    public static final Parcelable.Creator<t> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public ArrayList<v> f2518b;

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<String> f2519c;

    /* renamed from: d  reason: collision with root package name */
    public b[] f2520d;

    /* renamed from: e  reason: collision with root package name */
    public int f2521e;

    /* renamed from: f  reason: collision with root package name */
    public String f2522f;

    /* compiled from: FragmentManagerState.java */
    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<t> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public t createFromParcel(Parcel parcel) {
            return new t(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public t[] newArray(int i) {
            return new t[i];
        }
    }

    public t() {
        this.f2522f = null;
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeTypedList(this.f2518b);
        parcel.writeStringList(this.f2519c);
        parcel.writeTypedArray(this.f2520d, i);
        parcel.writeInt(this.f2521e);
        parcel.writeString(this.f2522f);
    }

    public t(Parcel parcel) {
        this.f2522f = null;
        this.f2518b = parcel.createTypedArrayList(v.CREATOR);
        this.f2519c = parcel.createStringArrayList();
        this.f2520d = (b[]) parcel.createTypedArray(b.CREATOR);
        this.f2521e = parcel.readInt();
        this.f2522f = parcel.readString();
    }
}