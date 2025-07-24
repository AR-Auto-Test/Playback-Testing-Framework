package com.google.ar.core.dependencies;

import android.os.IBinder;
import android.os.IInterface;

/* compiled from: IInstallService.java */
/* loaded from: classes.dex */
public abstract class h extends e implements i {
    public static i b(IBinder iBinder) {
        if (iBinder == null) {
            return null;
        }
        IInterface queryLocalInterface = iBinder.queryLocalInterface("com.google.android.play.core.install.protocol.IInstallService");
        return queryLocalInterface instanceof i ? (i) queryLocalInterface : new g(iBinder);
    }
}