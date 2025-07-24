package com.google.android.gms.internal.common;

import c.b.a.a.a;

/* compiled from: com.google.android.gms:play-services-basement@@18.1.0 */
/* loaded from: classes.dex */
public final class zzl extends zzk {
    private final char zza;

    public zzl(char c2) {
        this.zza = c2;
    }

    public final String toString() {
        StringBuilder x = a.x("CharMatcher.is('");
        int i = this.zza;
        char[] cArr = {'\\', 'u', 0, 0, 0, 0};
        for (int i2 = 0; i2 < 4; i2++) {
            cArr[5 - i2] = "0123456789ABCDEF".charAt(i & 15);
            i >>= 4;
        }
        x.append(String.copyValueOf(cArr));
        x.append("')");
        return x.toString();
    }

    @Override // com.google.android.gms.internal.common.zzo
    public final boolean zza(char c2) {
        return c2 == this.zza;
    }
}