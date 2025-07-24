package com.google.android.gms.internal.vision;

import android.content.Context;
import android.database.ContentObserver;
import android.util.Log;
import b.j.b.d;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public final class zzbd implements zzay {
    private static zzbd zza;
    private final Context zzb;
    private final ContentObserver zzc;

    private zzbd(Context context) {
        this.zzb = context;
        zzbf zzbfVar = new zzbf(this, null);
        this.zzc = zzbfVar;
        context.getContentResolver().registerContentObserver(zzaq.zza, true, zzbfVar);
    }

    public static zzbd zza(Context context) {
        zzbd zzbdVar;
        synchronized (zzbd.class) {
            if (zza == null) {
                zza = d.j(context, "com.google.android.providers.gsf.permission.READ_GSERVICES") == 0 ? new zzbd(context) : new zzbd();
            }
            zzbdVar = zza;
        }
        return zzbdVar;
    }

    /* JADX DEBUG: Method merged with bridge method */
    /* JADX INFO: Access modifiers changed from: private */
    @Override // com.google.android.gms.internal.vision.zzay
    /* renamed from: zzc */
    public final String zza(String str) {
        if (this.zzb == null) {
            return null;
        }
        try {
            return (String) zzbb.zza(new zzba(this, str) { // from class: com.google.android.gms.internal.vision.zzbc
                private final zzbd zza;
                private final String zzb;

                {
                    this.zza = this;
                    this.zzb = str;
                }

                @Override // com.google.android.gms.internal.vision.zzba
                public final Object zza() {
                    return this.zza.zzb(this.zzb);
                }
            });
        } catch (IllegalStateException | SecurityException e2) {
            String valueOf = String.valueOf(str);
            Log.e("GservicesLoader", valueOf.length() != 0 ? "Unable to read GServices for: ".concat(valueOf) : new String("Unable to read GServices for: "), e2);
            return null;
        }
    }

    public final /* synthetic */ String zzb(String str) {
        return zzaq.zza(this.zzb.getContentResolver(), str, (String) null);
    }

    private zzbd() {
        this.zzb = null;
        this.zzc = null;
    }

    public static synchronized void zza() {
        Context context;
        synchronized (zzbd.class) {
            zzbd zzbdVar = zza;
            if (zzbdVar != null && (context = zzbdVar.zzb) != null && zzbdVar.zzc != null) {
                context.getContentResolver().unregisterContentObserver(zza.zzc);
            }
            zza = null;
        }
    }
}