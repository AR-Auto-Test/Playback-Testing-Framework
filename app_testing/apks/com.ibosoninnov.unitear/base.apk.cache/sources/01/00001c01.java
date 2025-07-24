package com.google.ar.core.dependencies;

import android.os.Bundle;
import android.os.IBinder;
import android.os.Parcel;
import java.util.List;

/* compiled from: IInstallService.java */
/* loaded from: classes.dex */
public final class g extends d implements i {
    public g(IBinder iBinder) {
        super(iBinder, "com.google.android.play.core.install.protocol.IInstallService");
    }

    @Override // com.google.ar.core.dependencies.i
    public final void d(String str, Bundle bundle, k kVar) {
        Parcel a2 = a();
        a2.writeString(str);
        f.b(a2, bundle);
        f.c(a2, kVar);
        c(2, a2);
    }

    @Override // com.google.ar.core.dependencies.i
    public final void e(String str, List<Bundle> list, Bundle bundle, k kVar) {
        Parcel a2 = a();
        a2.writeString(str);
        a2.writeTypedList(list);
        f.b(a2, bundle);
        f.c(a2, kVar);
        c(1, a2);
    }
}