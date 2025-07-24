package com.google.vr.dynamite.client;

import android.os.IBinder;
import android.os.IInterface;
import android.os.Parcel;

/* compiled from: ILoadedInstanceCreator.java */
/* loaded from: classes2.dex */
public final class a extends com.google.ar.core.dependencies.d implements ILoadedInstanceCreator {
    public a(IBinder iBinder) {
        super(iBinder, "com.google.vr.dynamite.client.ILoadedInstanceCreator");
    }

    @Override // com.google.vr.dynamite.client.ILoadedInstanceCreator
    public final INativeLibraryLoader newNativeLibraryLoader(IObjectWrapper iObjectWrapper, IObjectWrapper iObjectWrapper2) {
        INativeLibraryLoader bVar;
        Parcel a2 = a();
        com.google.ar.core.dependencies.f.c(a2, iObjectWrapper);
        com.google.ar.core.dependencies.f.c(a2, iObjectWrapper2);
        Parcel b2 = b(1, a2);
        IBinder readStrongBinder = b2.readStrongBinder();
        if (readStrongBinder == null) {
            bVar = null;
        } else {
            IInterface queryLocalInterface = readStrongBinder.queryLocalInterface("com.google.vr.dynamite.client.INativeLibraryLoader");
            bVar = queryLocalInterface instanceof INativeLibraryLoader ? (INativeLibraryLoader) queryLocalInterface : new b(readStrongBinder);
        }
        b2.recycle();
        return bVar;
    }
}