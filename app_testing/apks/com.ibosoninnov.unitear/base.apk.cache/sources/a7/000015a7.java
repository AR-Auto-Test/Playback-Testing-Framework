package com.google.android.gms.measurement.internal;

import android.database.Cursor;
import android.database.sqlite.SQLiteException;
import android.text.TextUtils;
import android.util.Pair;
import com.google.android.gms.common.internal.Preconditions;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzw {
    public final /* synthetic */ zzaa zza;
    private com.google.android.gms.internal.measurement.zzft zzb;
    private Long zzc;
    private long zzd;

    /* JADX WARN: Code restructure failed: missing block: B:36:0x00fa, code lost:
        if (r14 == null) goto L64;
     */
    /* JADX WARN: Not initialized variable reg: 14, insn: 0x01df: MOVE  (r5 I:??[OBJECT, ARRAY]) = (r14 I:??[OBJECT, ARRAY]), block:B:66:0x01df */
    /* JADX WARN: Removed duplicated region for block: B:68:0x01e2  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final com.google.android.gms.internal.measurement.zzft zza(String str, com.google.android.gms.internal.measurement.zzft zzftVar) {
        Cursor cursor;
        Cursor cursor2;
        Cursor cursor3;
        Pair pair;
        Object obj;
        String zzh = zzftVar.zzh();
        List zzi = zzftVar.zzi();
        this.zza.zzf.zzu();
        Long l = (Long) zzkv.zzC(zzftVar, "_eid");
        if (l != null) {
            if (zzh.equals("_ep")) {
                Preconditions.checkNotNull(l);
                this.zza.zzf.zzu();
                String str2 = (String) zzkv.zzC(zzftVar, "_en");
                if (TextUtils.isEmpty(str2)) {
                    this.zza.zzt.zzay().zzh().zzb("Extra parameter without an event name. eventId", l);
                    return null;
                }
                if (this.zzb == null || this.zzc == null || l.longValue() != this.zzc.longValue()) {
                    zzam zzi2 = this.zza.zzf.zzi();
                    zzi2.zzg();
                    zzi2.zzW();
                    try {
                        try {
                            cursor2 = zzi2.zzh().rawQuery("select main_event, children_to_process from main_event_params where app_id=? and event_id=?", new String[]{str, l.toString()});
                        } catch (SQLiteException e2) {
                            e = e2;
                            cursor2 = null;
                        } catch (Throwable th) {
                            th = th;
                            cursor = null;
                            if (cursor != null) {
                            }
                            throw th;
                        }
                        try {
                            if (!cursor2.moveToFirst()) {
                                zzi2.zzt.zzay().zzj().zza("Main event not found");
                                cursor2.close();
                                pair = null;
                            } else {
                                try {
                                    pair = Pair.create((com.google.android.gms.internal.measurement.zzft) ((com.google.android.gms.internal.measurement.zzfs) zzkv.zzl(com.google.android.gms.internal.measurement.zzft.zze(), cursor2.getBlob(0))).zzaC(), Long.valueOf(cursor2.getLong(1)));
                                    cursor2.close();
                                } catch (IOException e3) {
                                    zzi2.zzt.zzay().zzd().zzd("Failed to merge main event. appId, eventId", zzeh.zzn(str), l, e3);
                                    cursor2.close();
                                    pair = null;
                                    if (pair == null) {
                                    }
                                    this.zza.zzt.zzay().zzh().zzc("Extra parameter without existing main event. eventName, eventId", str2, l);
                                    return null;
                                }
                            }
                        } catch (SQLiteException e4) {
                            e = e4;
                            zzi2.zzt.zzay().zzd().zzb("Error selecting main event", e);
                        }
                        if (pair == null && (obj = pair.first) != null) {
                            this.zzb = (com.google.android.gms.internal.measurement.zzft) obj;
                            this.zzd = ((Long) pair.second).longValue();
                            this.zza.zzf.zzu();
                            this.zzc = (Long) zzkv.zzC(this.zzb, "_eid");
                        } else {
                            this.zza.zzt.zzay().zzh().zzc("Extra parameter without existing main event. eventName, eventId", str2, l);
                            return null;
                        }
                    } catch (Throwable th2) {
                        th = th2;
                        cursor = cursor3;
                        if (cursor != null) {
                            cursor.close();
                        }
                        throw th;
                    }
                }
                long j = this.zzd - 1;
                this.zzd = j;
                if (j <= 0) {
                    zzam zzi3 = this.zza.zzf.zzi();
                    zzi3.zzg();
                    zzi3.zzt.zzay().zzj().zzb("Clearing complex main event info. appId", str);
                    try {
                        zzi3.zzh().execSQL("delete from main_event_params where app_id=?", new String[]{str});
                    } catch (SQLiteException e5) {
                        zzi3.zzt.zzay().zzd().zzb("Error clearing complex main event", e5);
                    }
                } else {
                    this.zza.zzf.zzi().zzJ(str, l, this.zzd, this.zzb);
                }
                ArrayList arrayList = new ArrayList();
                for (com.google.android.gms.internal.measurement.zzfx zzfxVar : this.zzb.zzi()) {
                    this.zza.zzf.zzu();
                    if (zzkv.zzB(zzftVar, zzfxVar.zzg()) == null) {
                        arrayList.add(zzfxVar);
                    }
                }
                if (!arrayList.isEmpty()) {
                    arrayList.addAll(zzi);
                    zzi = arrayList;
                } else {
                    this.zza.zzt.zzay().zzh().zzb("No unique parameters in main event. eventName", str2);
                }
                zzh = str2;
            } else {
                this.zzc = l;
                this.zzb = zzftVar;
                this.zza.zzf.zzu();
                Object zzC = zzkv.zzC(zzftVar, "_epc");
                long longValue = ((Long) (zzC != null ? zzC : 0L)).longValue();
                this.zzd = longValue;
                if (longValue <= 0) {
                    this.zza.zzt.zzay().zzh().zzb("Complex event with zero extra param count. eventName", zzh);
                } else {
                    this.zza.zzf.zzi().zzJ(str, (Long) Preconditions.checkNotNull(l), this.zzd, zzftVar);
                }
            }
        }
        com.google.android.gms.internal.measurement.zzfs zzfsVar = (com.google.android.gms.internal.measurement.zzfs) zzftVar.zzby();
        zzfsVar.zzi(zzh);
        zzfsVar.zzg();
        zzfsVar.zzd(zzi);
        return (com.google.android.gms.internal.measurement.zzft) zzfsVar.zzaC();
    }
}