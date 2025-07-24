package com.google.android.play.core.internal;

import android.os.Bundle;
import android.os.IBinder;
import android.os.Parcel;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzn extends zzk implements zzp {
    public zzn(IBinder iBinder) {
        super(iBinder, "com.google.android.play.core.appupdate.protocol.IAppUpdateService");
    }

    @Override // com.google.android.play.core.internal.zzp
    public final void zzc(String str, Bundle bundle, zzr zzrVar) {
        Parcel zza = zza();
        zza.writeString(str);
        zzm.zzb(zza, bundle);
        zzm.zzc(zza, zzrVar);
        zzb(3, zza);
    }

    @Override // com.google.android.play.core.internal.zzp
    public final void zzd(String str, Bundle bundle, zzr zzrVar) {
        Parcel zza = zza();
        zza.writeString(str);
        zzm.zzb(zza, bundle);
        zzm.zzc(zza, zzrVar);
        zzb(2, zza);
    }
}