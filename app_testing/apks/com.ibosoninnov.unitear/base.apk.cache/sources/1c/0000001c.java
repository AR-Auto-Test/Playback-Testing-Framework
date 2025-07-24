package android.support.v4.media.session;

import android.os.Parcel;
import android.os.Parcelable;

/* loaded from: classes.dex */
public class ParcelableVolumeInfo implements Parcelable {
    public static final Parcelable.Creator<ParcelableVolumeInfo> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public int f22b;

    /* renamed from: c  reason: collision with root package name */
    public int f23c;

    /* renamed from: d  reason: collision with root package name */
    public int f24d;

    /* renamed from: e  reason: collision with root package name */
    public int f25e;

    /* renamed from: f  reason: collision with root package name */
    public int f26f;

    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<ParcelableVolumeInfo> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public ParcelableVolumeInfo createFromParcel(Parcel parcel) {
            return new ParcelableVolumeInfo(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public ParcelableVolumeInfo[] newArray(int i) {
            return new ParcelableVolumeInfo[i];
        }
    }

    public ParcelableVolumeInfo(Parcel parcel) {
        this.f22b = parcel.readInt();
        this.f24d = parcel.readInt();
        this.f25e = parcel.readInt();
        this.f26f = parcel.readInt();
        this.f23c = parcel.readInt();
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.f22b);
        parcel.writeInt(this.f24d);
        parcel.writeInt(this.f25e);
        parcel.writeInt(this.f26f);
        parcel.writeInt(this.f23c);
    }
}