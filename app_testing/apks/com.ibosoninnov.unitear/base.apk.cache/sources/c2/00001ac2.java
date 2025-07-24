package com.google.android.play.core.internal;

import java.io.Closeable;
import java.io.InputStream;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public abstract class zzcm implements Closeable {
    public abstract long zza();

    public abstract InputStream zzb(long j, long j2);

    public final synchronized InputStream zzc() {
        return zzb(0L, zza());
    }
}