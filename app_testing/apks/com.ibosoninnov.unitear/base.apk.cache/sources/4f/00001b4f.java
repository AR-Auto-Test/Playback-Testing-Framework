package com.google.android.play.core.splitinstall;

import android.os.Bundle;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzay extends zzbb {
    public zzay(zzbc zzbcVar, com.google.android.play.core.tasks.zzi zziVar) {
        super(zzbcVar, zziVar);
    }

    @Override // com.google.android.play.core.splitinstall.zzbb, com.google.android.play.core.internal.zzcc
    public final void zzg(int i, Bundle bundle) {
        super.zzg(i, bundle);
        this.zza.zze(SplitInstallSessionState.zzd(bundle));
    }
}