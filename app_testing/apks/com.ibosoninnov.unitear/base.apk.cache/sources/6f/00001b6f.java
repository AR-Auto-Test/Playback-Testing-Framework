package com.google.android.play.core.splitinstall;

import android.os.Bundle;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzy implements Runnable {
    public final /* synthetic */ SplitInstallRequest zza;
    public final /* synthetic */ zzaa zzb;

    public zzy(zzaa zzaaVar, SplitInstallRequest splitInstallRequest) {
        this.zzb = zzaaVar;
        this.zza = splitInstallRequest;
    }

    @Override // java.lang.Runnable
    public final void run() {
        zzx zzxVar;
        List zze;
        zzxVar = this.zzb.zzb;
        List<String> moduleNames = this.zza.getModuleNames();
        zze = zzaa.zze(this.zza.getLanguages());
        Bundle bundle = new Bundle();
        bundle.putInt("session_id", 0);
        bundle.putInt(SettingsJsonConstants.APP_STATUS_KEY, 5);
        bundle.putInt("error_code", 0);
        if (!moduleNames.isEmpty()) {
            bundle.putStringArrayList("module_names", new ArrayList<>(moduleNames));
        }
        if (!zze.isEmpty()) {
            bundle.putStringArrayList("languages", new ArrayList<>(zze));
        }
        bundle.putLong("total_bytes_to_download", 0L);
        bundle.putLong("bytes_downloaded", 0L);
        zzxVar.zzm(SplitInstallSessionState.zzd(bundle));
    }
}