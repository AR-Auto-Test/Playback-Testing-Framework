package com.google.android.play.core.internal;

import java.io.InputStream;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzcn extends zzcm {
    private final zzcm zza;
    private final long zzb;
    private final long zzc;

    public zzcn(zzcm zzcmVar, long j, long j2) {
        this.zza = zzcmVar;
        long zzd = zzd(j);
        this.zzb = zzd;
        this.zzc = zzd(zzd + j2);
    }

    private final long zzd(long j) {
        if (j < 0) {
            return 0L;
        }
        return j > this.zza.zza() ? this.zza.zza() : j;
    }

    @Override // java.io.Closeable, java.lang.AutoCloseable
    public final void close() {
    }

    @Override // com.google.android.play.core.internal.zzcm
    public final long zza() {
        return this.zzc - this.zzb;
    }

    @Override // com.google.android.play.core.internal.zzcm
    public final InputStream zzb(long j, long j2) {
        long zzd = zzd(this.zzb);
        return this.zza.zzb(zzd, zzd(j2 + zzd) - zzd);
    }
}