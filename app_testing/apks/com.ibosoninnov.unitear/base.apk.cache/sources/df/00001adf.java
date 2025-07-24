package com.google.android.play.core.internal;

import android.os.Bundle;
import android.os.IBinder;
import android.os.IInterface;
import android.os.Parcel;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzz extends zzk implements IInterface {
    public zzz(IBinder iBinder) {
        super(iBinder, "com.google.android.play.core.assetpacks.protocol.IAssetPackExtractionServiceCallback");
    }

    public final void zzc(Bundle bundle) {
        Parcel zza = zza();
        zzm.zzb(zza, bundle);
        zzb(4, zza);
    }

    public final void zzd(Bundle bundle) {
        Parcel zza = zza();
        zzm.zzb(zza, bundle);
        zzb(3, zza);
    }

    public final void zze(Bundle bundle, Bundle bundle2) {
        Parcel zza = zza();
        zzm.zzb(zza, bundle);
        zzm.zzb(zza, bundle2);
        zzb(2, zza);
    }
}