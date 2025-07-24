package com.google.android.gms.internal.vision;

import android.net.Uri;
import b.f.a;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzbj {
    private static final a<String, Uri> zza = new a<>();

    public static synchronized Uri zza(String str) {
        Uri orDefault;
        synchronized (zzbj.class) {
            a<String, Uri> aVar = zza;
            orDefault = aVar.getOrDefault(str, null);
            if (orDefault == null) {
                String valueOf = String.valueOf(Uri.encode(str));
                orDefault = Uri.parse(valueOf.length() != 0 ? "content://com.google.android.gms.phenotype/".concat(valueOf) : new String("content://com.google.android.gms.phenotype/"));
                aVar.put(str, orDefault);
            }
        }
        return orDefault;
    }
}