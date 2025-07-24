package com.google.android.gms.internal.measurement;

import android.net.Uri;
import b.f.h;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzhh {
    private final h zza;

    public zzhh(h hVar) {
        this.zza = hVar;
    }

    public final String zza(Uri uri, String str, String str2, String str3) {
        if (uri != null) {
            h hVar = (h) this.zza.get(uri.toString());
            if (hVar == null) {
                return null;
            }
            return (String) hVar.get("".concat(String.valueOf(str3)));
        }
        return null;
    }
}