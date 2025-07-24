package com.google.ar.core.dependencies;

import android.os.IInterface;
import android.os.Parcel;
import android.os.Parcelable;

/* compiled from: Codecs.java */
/* loaded from: classes.dex */
public final class f {
    static {
        f.class.getClassLoader();
    }

    private f() {
    }

    public static <T extends Parcelable> T a(Parcel parcel, Parcelable.Creator<T> creator) {
        if (parcel.readInt() == 0) {
            return null;
        }
        return creator.createFromParcel(parcel);
    }

    public static void b(Parcel parcel, Parcelable parcelable) {
        parcel.writeInt(1);
        parcelable.writeToParcel(parcel, 0);
    }

    public static void c(Parcel parcel, IInterface iInterface) {
        if (iInterface == null) {
            parcel.writeStrongBinder(null);
        } else {
            parcel.writeStrongBinder(iInterface.asBinder());
        }
    }
}