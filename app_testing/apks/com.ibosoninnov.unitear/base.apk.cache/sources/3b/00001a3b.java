package com.google.android.play.core.assetpacks;

import java.io.File;
import java.io.IOException;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzdu {
    private static final com.google.android.play.core.internal.zzag zza = new com.google.android.play.core.internal.zzag("MergeSliceTaskHandler");
    private final zzbh zzb;

    public zzdu(zzbh zzbhVar) {
        this.zzb = zzbhVar;
    }

    private static void zzb(File file, File file2) {
        File[] listFiles;
        if (file.isDirectory()) {
            file2.mkdirs();
            for (File file3 : file.listFiles()) {
                zzb(file3, new File(file2, file3.getName()));
            }
            if (file.delete()) {
                return;
            }
            String valueOf = String.valueOf(file);
            valueOf.length();
            throw new zzck("Unable to delete directory: ".concat(valueOf));
        } else if (!file2.exists()) {
            if (file.renameTo(file2)) {
                return;
            }
            String valueOf2 = String.valueOf(file);
            valueOf2.length();
            throw new zzck("Unable to move file: ".concat(valueOf2));
        } else {
            throw new zzck("File clashing with existing file from other slice: ".concat(file2.toString()));
        }
    }

    public final void zza(zzdt zzdtVar) {
        File zzq = this.zzb.zzq(zzdtVar.zzl, zzdtVar.zza, zzdtVar.zzb, zzdtVar.zzc);
        if (zzq.exists()) {
            File zzj = this.zzb.zzj(zzdtVar.zzl, zzdtVar.zza, zzdtVar.zzb);
            if (!zzj.exists()) {
                zzj.mkdirs();
            }
            zzb(zzq, zzj);
            try {
                this.zzb.zzA(zzdtVar.zzl, zzdtVar.zza, zzdtVar.zzb, this.zzb.zzb(zzdtVar.zzl, zzdtVar.zza, zzdtVar.zzb) + 1);
                return;
            } catch (IOException e2) {
                zza.zzb("Writing merge checkpoint failed with %s.", e2.getMessage());
                throw new zzck("Writing merge checkpoint failed.", e2, zzdtVar.zzk);
            }
        }
        throw new zzck(String.format("Cannot find verified files for slice %s.", zzdtVar.zzc), zzdtVar.zzk);
    }
}