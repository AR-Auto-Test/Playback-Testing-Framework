package com.google.android.gms.signin.internal;

import android.os.IBinder;
import android.os.Parcel;
import com.google.android.gms.common.internal.IAccountAccessor;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zah extends com.google.android.gms.internal.base.zab implements zae {
    public zah(IBinder iBinder) {
        super(iBinder, "com.google.android.gms.signin.internal.ISignInService");
    }

    @Override // com.google.android.gms.signin.internal.zae
    public final void zaa(int i) {
        Parcel zaa = zaa();
        zaa.writeInt(i);
        zab(7, zaa);
    }

    @Override // com.google.android.gms.signin.internal.zae
    public final void zaa(IAccountAccessor iAccountAccessor, int i, boolean z) {
        Parcel zaa = zaa();
        com.google.android.gms.internal.base.zad.zaa(zaa, iAccountAccessor);
        zaa.writeInt(i);
        com.google.android.gms.internal.base.zad.zaa(zaa, z);
        zab(9, zaa);
    }

    @Override // com.google.android.gms.signin.internal.zae
    public final void zaa(zak zakVar, zac zacVar) {
        Parcel zaa = zaa();
        com.google.android.gms.internal.base.zad.zaa(zaa, zakVar);
        com.google.android.gms.internal.base.zad.zaa(zaa, zacVar);
        zab(12, zaa);
    }
}