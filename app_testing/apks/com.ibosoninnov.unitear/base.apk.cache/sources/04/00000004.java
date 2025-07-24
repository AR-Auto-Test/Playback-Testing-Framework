package a.a.a.b;

import a.a.a.b.a;
import android.annotation.SuppressLint;
import android.os.Bundle;
import android.os.IBinder;
import android.os.IInterface;
import android.os.Parcel;
import android.os.Parcelable;

/* compiled from: ResultReceiver.java */
@SuppressLint({"BanParcelableUsage"})
/* loaded from: classes.dex */
public class b implements Parcelable {
    public static final Parcelable.Creator<b> CREATOR = new a();

    /* renamed from: b  reason: collision with root package name */
    public a.a.a.b.a f2b;

    /* compiled from: ResultReceiver.java */
    /* loaded from: classes.dex */
    public class a implements Parcelable.Creator<b> {
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.os.Parcelable.Creator
        public b createFromParcel(Parcel parcel) {
            return new b(parcel);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object[]' to match base method */
        @Override // android.os.Parcelable.Creator
        public b[] newArray(int i) {
            return new b[i];
        }
    }

    /* compiled from: ResultReceiver.java */
    /* renamed from: a.a.a.b.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class BinderC0002b extends a.AbstractBinderC0000a {
        public BinderC0002b() {
        }
    }

    public b(Parcel parcel) {
        a.a.a.b.a c0001a;
        IBinder readStrongBinder = parcel.readStrongBinder();
        int i = a.AbstractBinderC0000a.f0a;
        if (readStrongBinder == null) {
            c0001a = null;
        } else {
            IInterface queryLocalInterface = readStrongBinder.queryLocalInterface("android.support.v4.os.IResultReceiver");
            if (queryLocalInterface != null && (queryLocalInterface instanceof a.a.a.b.a)) {
                c0001a = (a.a.a.b.a) queryLocalInterface;
            } else {
                c0001a = new a.AbstractBinderC0000a.C0001a(readStrongBinder);
            }
        }
        this.f2b = c0001a;
    }

    public void a(int i, Bundle bundle) {
    }

    @Override // android.os.Parcelable
    public int describeContents() {
        return 0;
    }

    @Override // android.os.Parcelable
    public void writeToParcel(Parcel parcel, int i) {
        synchronized (this) {
            if (this.f2b == null) {
                this.f2b = new BinderC0002b();
            }
            parcel.writeStrongBinder(this.f2b.asBinder());
        }
    }
}