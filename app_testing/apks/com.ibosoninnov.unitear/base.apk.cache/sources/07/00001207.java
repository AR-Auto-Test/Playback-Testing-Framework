package com.google.android.gms.internal.measurement;

import android.net.Uri;
import b.f.a;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhq {
    private static final a zza = new a();

    public static synchronized Uri zza(String str) {
        synchronized (zzhq.class) {
            a aVar = zza;
            Uri uri = (Uri) aVar.getOrDefault("com.google.android.gms.measurement", null);
            if (uri == null) {
                Uri parse = Uri.parse("content://com.google.android.gms.phenotype/".concat(String.valueOf(Uri.encode("com.google.android.gms.measurement"))));
                aVar.put("com.google.android.gms.measurement", parse);
                return parse;
            }
            return uri;
        }
    }
}