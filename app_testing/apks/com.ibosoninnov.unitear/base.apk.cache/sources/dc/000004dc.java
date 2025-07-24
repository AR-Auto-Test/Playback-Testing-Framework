package b.m;

import android.os.Parcel;
import android.os.Parcelable;
import java.io.Serializable;

/* compiled from: ObservableBoolean.java */
/* loaded from: classes.dex */
public class h extends b implements Parcelable, Serializable {
    public static final Parcelable.Creator<h> CREATOR = new a();

    /* renamed from: c  reason: collision with root package name */
    public boolean f2335c;

    /* compiled from: ObservableBoolean.java */
    /* loaded from: classes.dex */
    public static class a implements Parcelable.Creator<h> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public h createFromParcel(Parcel parcel) {
            return new h(parcel.readInt() == 1);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public h[] newArray(int i) {
            return new h[i];
        }
    }

    public h(boolean z) {
        this.f2335c = z;
    }

    public void c(boolean z) {
        if (z != this.f2335c) {
            this.f2335c = z;
            synchronized (this) {
                i iVar = this.f2328b;
                if (iVar != null) {
                    iVar.b(this, 0, null);
                }
            }
        }
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeInt(this.f2335c ? 1 : 0);
    }

    public h() {
    }
}