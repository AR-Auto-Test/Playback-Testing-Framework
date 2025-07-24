package com.google.android.play.core.internal;

import android.os.IBinder;
import android.os.IInterface;
import java.util.List;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzao extends zzah {
    public final /* synthetic */ IBinder zza;
    public final /* synthetic */ zzar zzb;

    public zzao(zzar zzarVar, IBinder iBinder) {
        this.zzb = zzarVar;
        this.zza = iBinder;
    }

    @Override // com.google.android.play.core.internal.zzah
    public final void zza() {
        zzan zzanVar;
        List<Runnable> list;
        List list2;
        zzas zzasVar = this.zzb.zza;
        zzanVar = zzasVar.zzj;
        zzasVar.zzo = (IInterface) zzanVar.zza(this.zza);
        zzas.zzo(this.zzb.zza);
        this.zzb.zza.zzh = false;
        list = this.zzb.zza.zze;
        for (Runnable runnable : list) {
            runnable.run();
        }
        list2 = this.zzb.zza.zze;
        list2.clear();
    }
}