package com.google.android.play.core.splitcompat;

import android.util.Log;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzp implements Runnable {
    public final /* synthetic */ SplitCompat zza;

    public zzp(SplitCompat splitCompat) {
        this.zza = splitCompat;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zze zzeVar;
        try {
            zzeVar = this.zza.zzc;
            zzeVar.zzk();
        } catch (Exception e2) {
            Log.e("SplitCompat", "Failed to cleanup splitcompat storage", e2);
        }
    }
}