package b.v;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.Parcel;
import android.os.Parcelable;
import java.util.UUID;

/* compiled from: NavBackStackEntryState.java */
@SuppressLint({"BanParcelableUsage"})
/* loaded from: classes.dex */
public final class f implements Parcelable {
    public static final Parcelable.Creator<f> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public final UUID f2622b;

    /* renamed from: c  reason: collision with root package name */
    public final int f2623c;

    /* renamed from: d  reason: collision with root package name */
    public final Bundle f2624d;

    /* renamed from: e  reason: collision with root package name */
    public final Bundle f2625e;

    /* compiled from: NavBackStackEntryState.java */
    /* loaded from: classes.dex */
    public class a implements Parcelable.Creator<f> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public f createFromParcel(Parcel parcel) {
            return new f(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public f[] newArray(int i) {
            return new f[i];
        }
    }

    public f(e eVar) {
        this.f2622b = eVar.f2619f;
        this.f2623c = eVar.f2615b.f2645d;
        this.f2624d = eVar.f2616c;
        Bundle bundle = new Bundle();
        this.f2625e = bundle;
        eVar.f2618e.b(bundle);
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeString(this.f2622b.toString());
        parcel.writeInt(this.f2623c);
        parcel.writeBundle(this.f2624d);
        parcel.writeBundle(this.f2625e);
    }

    public f(Parcel parcel) {
        this.f2622b = UUID.fromString(parcel.readString());
        this.f2623c = parcel.readInt();
        this.f2624d = parcel.readBundle(f.class.getClassLoader());
        this.f2625e = parcel.readBundle(f.class.getClassLoader());
    }
}