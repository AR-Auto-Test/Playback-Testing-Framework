package com.google.vr.dynamite.client;

import android.os.IBinder;
import android.os.Parcel;

/* compiled from: INativeLibraryLoader.java */
/* loaded from: classes2.dex */
public final class b extends com.google.ar.core.dependencies.d implements INativeLibraryLoader {
    public b(IBinder iBinder) {
        super(iBinder, "com.google.vr.dynamite.client.INativeLibraryLoader");
    }

    @Override // com.google.vr.dynamite.client.INativeLibraryLoader
    public final int checkVersion(String str) {
        Parcel a2 = a();
        a2.writeString(str);
        Parcel b2 = b(2, a2);
        int readInt = b2.readInt();
        b2.recycle();
        return readInt;
    }

    @Override // com.google.vr.dynamite.client.INativeLibraryLoader
    public final long initializeAndLoadNativeLibrary(String str) {
        Parcel a2 = a();
        a2.writeString(str);
        Parcel b2 = b(1, a2);
        long readLong = b2.readLong();
        b2.recycle();
        return readLong;
    }
}