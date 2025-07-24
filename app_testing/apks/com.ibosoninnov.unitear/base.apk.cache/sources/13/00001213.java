package com.google.android.gms.internal.measurement;

import android.content.Context;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public abstract class zzib {
    public static final /* synthetic */ int zzc = 0;
    private static volatile zzhz zze = null;
    private static volatile boolean zzf = false;
    public final zzhy zza;
    public final String zzb;
    private final Object zzj;
    private volatile int zzk = -1;
    private volatile Object zzl;
    private final boolean zzm;
    private static final Object zzd = new Object();
    private static final AtomicReference zzg = new AtomicReference();
    private static final zzid zzh = new zzid(zzht.zza, null);
    private static final AtomicInteger zzi = new AtomicInteger();

    public /* synthetic */ zzib(zzhy zzhyVar, String str, Object obj, boolean z, zzia zziaVar) {
        if (zzhyVar.zzb == null) {
            throw new IllegalArgumentException("Must pass a valid SharedPreferences file name or ContentProvider URI");
        }
        this.zza = zzhyVar;
        this.zzb = str;
        this.zzj = obj;
        this.zzm = true;
    }

    public static void zzd() {
        zzi.incrementAndGet();
    }

    public static void zze(final Context context) {
        if (zze == null) {
            Object obj = zzd;
            synchronized (obj) {
                if (zze == null) {
                    synchronized (obj) {
                        zzhz zzhzVar = zze;
                        Context applicationContext = context.getApplicationContext();
                        if (applicationContext != null) {
                            context = applicationContext;
                        }
                        if (zzhzVar == null || zzhzVar.zza() != context) {
                            zzhf.zze();
                            zzic.zzc();
                            zzhn.zze();
                            zze = new zzhc(context, zzim.zza(new zzii() { // from class: com.google.android.gms.internal.measurement.zzhs
                                @Override // com.google.android.gms.internal.measurement.zzii
                                public final Object zza() {
                                    Context context2 = context;
                                    int i = zzib.zzc;
                                    return zzho.zza(context2);
                                }
                            }));
                            zzi.incrementAndGet();
                        }
                    }
                }
            }
        }
    }

    public abstract Object zza(Object obj);

    /* JADX WARN: Removed duplicated region for block: B:37:0x0099 A[Catch: all -> 0x00d3, TryCatch #0 {, blocks: (B:8:0x0016, B:10:0x001a, B:12:0x0020, B:14:0x0029, B:16:0x0037, B:20:0x0060, B:22:0x006a, B:38:0x009b, B:40:0x00ab, B:42:0x00bf, B:43:0x00c2, B:44:0x00c6, B:26:0x0073, B:28:0x0079, B:32:0x008b, B:34:0x0091, B:37:0x0099, B:31:0x0089, B:18:0x0050, B:45:0x00cb, B:46:0x00d0, B:47:0x00d1), top: B:54:0x0016 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Object zzb() {
        zzhk zza;
        Object zzb;
        if (!this.zzm) {
            Objects.requireNonNull(this.zzb, "flagName must not be null");
        }
        int i = zzi.get();
        if (this.zzk < i) {
            synchronized (this) {
                if (this.zzk < i) {
                    zzhz zzhzVar = zze;
                    if (zzhzVar != null) {
                        zzhy zzhyVar = this.zza;
                        boolean z = zzhyVar.zzf;
                        if (zzhyVar.zzb != null) {
                            if (zzhp.zza(zzhzVar.zza(), this.zza.zzb)) {
                                boolean z2 = this.zza.zzh;
                                zza = zzhf.zza(zzhzVar.zza().getContentResolver(), this.zza.zzb, zzhr.zza);
                            } else {
                                zza = null;
                            }
                        } else {
                            Context zza2 = zzhzVar.zza();
                            String str = this.zza.zza;
                            zza = zzic.zza(zza2, null, zzhr.zza);
                        }
                        Object zza3 = (zza == null || (zzb = zza.zzb(zzc())) == null) ? null : zza(zzb);
                        if (zza3 == null) {
                            if (!this.zza.zze) {
                                String zzb2 = zzhn.zza(zzhzVar.zza()).zzb(this.zza.zze ? null : this.zzb);
                                if (zzb2 != null) {
                                    zza3 = zza(zzb2);
                                    if (zza3 == null) {
                                        zza3 = this.zzj;
                                    }
                                }
                            }
                            zza3 = null;
                            if (zza3 == null) {
                            }
                        }
                        zzig zzigVar = (zzig) zzhzVar.zzb().zza();
                        if (zzigVar.zzb()) {
                            zzhy zzhyVar2 = this.zza;
                            String zza4 = ((zzhh) zzigVar.zza()).zza(zzhyVar2.zzb, null, zzhyVar2.zzd, this.zzb);
                            zza3 = zza4 == null ? this.zzj : zza(zza4);
                        }
                        this.zzl = zza3;
                        this.zzk = i;
                    } else {
                        throw new IllegalStateException("Must call PhenotypeFlag.init() first");
                    }
                }
            }
        }
        return this.zzl;
    }

    public final String zzc() {
        String str = this.zza.zzd;
        return this.zzb;
    }
}