package com.google.android.play.core.splitcompat;

import java.io.File;
import java.util.Objects;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzb extends zzs {
    private final File zza;
    private final String zzb;

    public zzb(File file, String str) {
        Objects.requireNonNull(file, "Null splitFile");
        this.zza = file;
        Objects.requireNonNull(str, "Null splitId");
        this.zzb = str;
    }

    public final boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof zzs) {
            zzs zzsVar = (zzs) obj;
            if (this.zza.equals(zzsVar.zza()) && this.zzb.equals(zzsVar.zzb())) {
                return true;
            }
        }
        return false;
    }

    public final int hashCode() {
        return ((this.zza.hashCode() ^ 1000003) * 1000003) ^ this.zzb.hashCode();
    }

    public final String toString() {
        String obj = this.zza.toString();
        String str = this.zzb;
        StringBuilder sb = new StringBuilder(str.length() + obj.length() + 35);
        sb.append("SplitFileInfo{splitFile=");
        sb.append(obj);
        sb.append(", splitId=");
        sb.append(str);
        sb.append("}");
        return sb.toString();
    }

    @Override // com.google.android.play.core.splitcompat.zzs
    public final File zza() {
        return this.zza;
    }

    @Override // com.google.android.play.core.splitcompat.zzs
    public final String zzb() {
        return this.zzb;
    }
}