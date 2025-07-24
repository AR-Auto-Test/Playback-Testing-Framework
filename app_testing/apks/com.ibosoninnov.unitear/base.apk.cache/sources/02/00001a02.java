package com.google.android.play.core.assetpacks;

import com.google.common.primitives.UnsignedBytes;
import com.google.common.primitives.UnsignedInts;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzbr {
    public static int zza(byte[] bArr, int i) {
        return ((bArr[i + 1] & UnsignedBytes.MAX_VALUE) << 8) | (bArr[i] & UnsignedBytes.MAX_VALUE);
    }

    public static int zzb(byte[] bArr, int i) {
        return (bArr[i + 3] & UnsignedBytes.MAX_VALUE) | ((bArr[i] & UnsignedBytes.MAX_VALUE) << 24) | ((bArr[i + 1] & UnsignedBytes.MAX_VALUE) << 16) | ((bArr[i + 2] & UnsignedBytes.MAX_VALUE) << 8);
    }

    public static long zzc(byte[] bArr, int i) {
        return ((zza(bArr, i + 2) << 16) | zza(bArr, i)) & UnsignedInts.INT_MASK;
    }
}