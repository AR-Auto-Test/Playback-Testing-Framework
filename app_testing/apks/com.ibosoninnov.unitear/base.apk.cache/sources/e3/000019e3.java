package com.google.android.play.core.assetpacks;

import android.os.Bundle;
import android.os.ParcelFileDescriptor;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzan extends zzal {
    public zzan(zzaw zzawVar, com.google.android.play.core.tasks.zzi zziVar) {
        super(zzawVar, zziVar);
    }

    @Override // com.google.android.play.core.assetpacks.zzal, com.google.android.play.core.internal.zzw
    public final void zze(Bundle bundle, Bundle bundle2) {
        super.zze(bundle, bundle2);
        this.zza.zze((ParcelFileDescriptor) bundle.getParcelable("chunk_file_descriptor"));
    }
}