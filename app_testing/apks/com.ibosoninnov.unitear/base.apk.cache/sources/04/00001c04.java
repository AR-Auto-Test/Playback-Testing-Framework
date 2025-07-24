package com.google.ar.core.dependencies;

import android.os.Bundle;
import android.os.Parcel;

/* compiled from: IInstallServiceCallback.java */
/* loaded from: classes.dex */
public abstract class j extends e implements k {
    public j() {
        super("com.google.android.play.core.install.protocol.IInstallServiceCallback");
    }

    @Override // com.google.ar.core.dependencies.e
    public final boolean a(int i, Parcel parcel) {
        if (i == 1) {
            c((Bundle) f.a(parcel, Bundle.CREATOR));
        } else if (i == 2) {
            b((Bundle) f.a(parcel, Bundle.CREATOR));
        } else if (i != 3) {
            return false;
        } else {
            Bundle bundle = (Bundle) f.a(parcel, Bundle.CREATOR);
        }
        return true;
    }
}