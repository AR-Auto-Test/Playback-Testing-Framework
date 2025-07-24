package com.google.android.play.core.splitinstall;

import android.os.Bundle;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzba extends zzbb {
    public zzba(zzbc zzbcVar, com.google.android.play.core.tasks.zzi zziVar) {
        super(zzbcVar, zziVar);
    }

    @Override // com.google.android.play.core.splitinstall.zzbb, com.google.android.play.core.internal.zzcc
    public final void zzi(int i, Bundle bundle) {
        super.zzi(i, bundle);
        this.zza.zze(Integer.valueOf(i));
    }
}