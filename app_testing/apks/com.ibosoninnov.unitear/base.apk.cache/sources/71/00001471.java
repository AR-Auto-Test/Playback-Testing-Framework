package com.google.android.gms.measurement.internal;

import android.content.ContentValues;
import android.database.Cursor;
import android.database.sqlite.SQLiteException;
import android.util.Log;
import b.f.a;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.internal.measurement.zznz;
import com.google.android.gms.internal.measurement.zzoc;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzaa extends zzkh {
    private String zza;
    private Set zzb;
    private Map zzc;
    private Long zzd;
    private Long zze;

    public zzaa(zzkt zzktVar) {
        super(zzktVar);
    }

    private final zzu zzd(Integer num) {
        if (this.zzc.containsKey(num)) {
            return (zzu) this.zzc.get(num);
        }
        zzu zzuVar = new zzu(this, this.zza, null);
        this.zzc.put(num, zzuVar);
        return zzuVar;
    }

    private final boolean zzf(int i, int i2) {
        zzu zzuVar = (zzu) this.zzc.get(Integer.valueOf(i));
        if (zzuVar == null) {
            return false;
        }
        return zzu.zzb(zzuVar).get(i2);
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:108:0x02cd */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:41:0x0153 */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:435:0x008b */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:501:0x02c4 */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:533:0x030f */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:535:0x030f */
    /* JADX DEBUG: Type inference failed for r0v84. Raw type applied. Possible types: V */
    /* JADX DEBUG: Type inference failed for r16v11. Raw type applied. Possible types: V */
    /* JADX DEBUG: Type inference failed for r16v15. Raw type applied. Possible types: V */
    /* JADX DEBUG: Type inference failed for r16v7. Raw type applied. Possible types: V */
    /* JADX WARN: Can't wrap try/catch for region: R(11:1|(2:2|(2:4|(2:6|7))(2:526|527))|8|(3:10|11|12)|16|(16:(6:19|20|21|22|23|(20:(7:25|26|27|28|(1:30)(3:503|(1:505)(1:507)|506)|31|(1:34)(1:33))|35|36|37|38|39|40|(2:42|43)(3:463|(6:464|465|466|467|468|(1:471)(1:470))|472)|44|(1:46)(6:274|(11:276|277|278|279|280|(2:(3:282|(1:284)|285)|287)(1:449)|288|289|(3:384|(6:387|(1:446)(2:391|(4:397|398|(7:400|(4:403|(2:405|406)(1:408)|407|401)|409|410|(4:413|(3:415|416|417)(1:419)|418|411)|420|421)(6:425|(4:428|(2:430|431)(1:433)|432|426)|434|435|(4:438|(2:440|441)(1:443)|442|436)|444)|422)(4:393|394|395|396))|423|424|396|385)|448)|291|292)(1:462)|293|(10:296|(3:300|(4:303|(5:305|306|(1:308)(1:312)|309|310)(1:313)|311|301)|314)|315|(3:319|(4:322|(3:327|328|329)|330|320)|333)|334|(3:336|(6:339|(2:341|(3:343|344|345))(1:348)|346|347|345|337)|349)|350|(3:359|(8:362|(1:364)|365|(1:367)|368|(3:370|371|372)(1:374)|373|360)|375)|376|294)|382|383)|47|(3:175|(4:178|(10:180|181|(1:183)(1:271)|184|(9:186|187|188|189|190|191|193|194|(4:196|(11:197|198|199|200|201|202|203|(3:205|206|207)(1:248)|208|209|(1:212)(1:211))|213|214)(4:254|255|247|214))(1:270)|215|(4:218|(3:236|237|238)(4:220|221|(2:222|(2:224|(1:226)(2:227|228))(1:235))|(3:230|231|232)(1:234))|233|216)|239|240|241)(1:272)|242|176)|273)|49|50|(3:77|(6:80|(6:82|83|84|85|86|(3:(9:88|89|90|91|92|(1:94)(1:151)|95|96|(1:99)(1:98))|100|101)(4:158|159|150|101))(1:173)|102|(2:103|(2:105|(3:141|142|143)(6:107|(2:108|(4:110|(3:112|(1:114)(1:137)|115)(1:138)|116|(1:1)(2:120|(1:122)(2:123|124)))(2:139|140))|(2:129|128)|126|127|128))(0))|144|78)|174)|52|53|(9:56|57|58|59|60|61|(2:63|64)(1:66)|65|54)|74|75)(2:511|512))|39|40|(0)(0)|44|(0)(0)|47|(0)|49|50|(0)|52|53|(1:54)|74|75)|525|36|37|38|(5:(0)|(0)|(0)|(0)|(0))) */
    /* JADX WARN: Can't wrap try/catch for region: R(26:1|(2:2|(2:4|(2:6|7))(2:526|527))|8|(3:10|11|12)|16|(6:19|20|21|22|23|(20:(7:25|26|27|28|(1:30)(3:503|(1:505)(1:507)|506)|31|(1:34)(1:33))|35|36|37|38|39|40|(2:42|43)(3:463|(6:464|465|466|467|468|(1:471)(1:470))|472)|44|(1:46)(6:274|(11:276|277|278|279|280|(2:(3:282|(1:284)|285)|287)(1:449)|288|289|(3:384|(6:387|(1:446)(2:391|(4:397|398|(7:400|(4:403|(2:405|406)(1:408)|407|401)|409|410|(4:413|(3:415|416|417)(1:419)|418|411)|420|421)(6:425|(4:428|(2:430|431)(1:433)|432|426)|434|435|(4:438|(2:440|441)(1:443)|442|436)|444)|422)(4:393|394|395|396))|423|424|396|385)|448)|291|292)(1:462)|293|(10:296|(3:300|(4:303|(5:305|306|(1:308)(1:312)|309|310)(1:313)|311|301)|314)|315|(3:319|(4:322|(3:327|328|329)|330|320)|333)|334|(3:336|(6:339|(2:341|(3:343|344|345))(1:348)|346|347|345|337)|349)|350|(3:359|(8:362|(1:364)|365|(1:367)|368|(3:370|371|372)(1:374)|373|360)|375)|376|294)|382|383)|47|(3:175|(4:178|(10:180|181|(1:183)(1:271)|184|(9:186|187|188|189|190|191|193|194|(4:196|(11:197|198|199|200|201|202|203|(3:205|206|207)(1:248)|208|209|(1:212)(1:211))|213|214)(4:254|255|247|214))(1:270)|215|(4:218|(3:236|237|238)(4:220|221|(2:222|(2:224|(1:226)(2:227|228))(1:235))|(3:230|231|232)(1:234))|233|216)|239|240|241)(1:272)|242|176)|273)|49|50|(3:77|(6:80|(6:82|83|84|85|86|(3:(9:88|89|90|91|92|(1:94)(1:151)|95|96|(1:99)(1:98))|100|101)(4:158|159|150|101))(1:173)|102|(2:103|(2:105|(3:141|142|143)(6:107|(2:108|(4:110|(3:112|(1:114)(1:137)|115)(1:138)|116|(1:1)(2:120|(1:122)(2:123|124)))(2:139|140))|(2:129|128)|126|127|128))(0))|144|78)|174)|52|53|(9:56|57|58|59|60|61|(2:63|64)(1:66)|65|54)|74|75)(2:511|512))|525|36|37|38|39|40|(0)(0)|44|(0)(0)|47|(0)|49|50|(0)|52|53|(1:54)|74|75|(5:(0)|(0)|(0)|(0)|(0))) */
    /* JADX WARN: Code restructure failed: missing block: B:117:0x02ef, code lost:
        if (r5 == null) goto L289;
     */
    /* JADX WARN: Code restructure failed: missing block: B:295:0x07b6, code lost:
        if (r5 != null) goto L247;
     */
    /* JADX WARN: Code restructure failed: missing block: B:358:0x0959, code lost:
        if (r13 == null) goto L101;
     */
    /* JADX WARN: Code restructure failed: missing block: B:393:0x0a59, code lost:
        r0 = r64.zzt.zzay().zzk();
        r6 = com.google.android.gms.measurement.internal.zzeh.zzn(r64.zza);
     */
    /* JADX WARN: Code restructure failed: missing block: B:394:0x0a6d, code lost:
        if (r7.zzj() == false) goto L135;
     */
    /* JADX WARN: Code restructure failed: missing block: B:395:0x0a6f, code lost:
        r7 = java.lang.Integer.valueOf(r7.zza());
     */
    /* JADX WARN: Code restructure failed: missing block: B:396:0x0a78, code lost:
        r7 = null;
     */
    /* JADX WARN: Code restructure failed: missing block: B:397:0x0a79, code lost:
        r0.zzc("Invalid property filter ID. appId, id", r6, java.lang.String.valueOf(r7));
     */
    /* JADX WARN: Code restructure failed: missing block: B:50:0x0171, code lost:
        if (r5 == null) goto L525;
     */
    /* JADX WARN: Code restructure failed: missing block: B:80:0x0220, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:81:0x0221, code lost:
        r18 = "audience_id";
     */
    /* JADX WARN: Code restructure failed: missing block: B:83:0x0226, code lost:
        r0 = th;
     */
    /* JADX WARN: Code restructure failed: missing block: B:84:0x0227, code lost:
        r5 = null;
     */
    /* JADX WARN: Code restructure failed: missing block: B:85:0x022a, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:86:0x022b, code lost:
        r18 = "audience_id";
        r19 = "data";
        r4 = null;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:168:0x0454  */
    /* JADX WARN: Removed duplicated region for block: B:242:0x060b  */
    /* JADX WARN: Removed duplicated region for block: B:321:0x086c  */
    /* JADX WARN: Removed duplicated region for block: B:406:0x0ab6  */
    /* JADX WARN: Removed duplicated region for block: B:53:0x0176  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x01b0 A[Catch: SQLiteException -> 0x0220, all -> 0x0b46, TRY_LEAVE, TryCatch #7 {SQLiteException -> 0x0220, blocks: (B:58:0x01aa, B:60:0x01b0, B:62:0x01be, B:63:0x01c3, B:64:0x01cd, B:65:0x01dd, B:67:0x01ec), top: B:432:0x01aa }] */
    /* JADX WARN: Removed duplicated region for block: B:62:0x01be A[Catch: SQLiteException -> 0x0220, all -> 0x0b46, TRY_ENTER, TryCatch #7 {SQLiteException -> 0x0220, blocks: (B:58:0x01aa, B:60:0x01b0, B:62:0x01be, B:63:0x01c3, B:64:0x01cd, B:65:0x01dd, B:67:0x01ec), top: B:432:0x01aa }] */
    /* JADX WARN: Removed duplicated region for block: B:89:0x0249  */
    /* JADX WARN: Removed duplicated region for block: B:93:0x0254  */
    /* JADX WARN: Removed duplicated region for block: B:94:0x025c  */
    /* JADX WARN: Type inference failed for: r0v39, types: [b.f.a, b.f.h] */
    /* JADX WARN: Type inference failed for: r0v45, types: [java.util.Map] */
    /* JADX WARN: Type inference failed for: r0v69 */
    /* JADX WARN: Type inference failed for: r0v71, types: [java.util.Map] */
    /* JADX WARN: Type inference failed for: r5v5, types: [android.database.sqlite.SQLiteDatabase] */
    /* JADX WARN: Type inference failed for: r5v59, types: [java.lang.String] */
    /* JADX WARN: Type inference failed for: r5v6 */
    /* JADX WARN: Type inference failed for: r5v60 */
    /* JADX WARN: Type inference failed for: r5v61, types: [java.lang.String[]] */
    /* JADX WARN: Type inference failed for: r5v8, types: [android.database.Cursor] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final List zza(String str, List list, List list2, Long l, Long l2) {
        int i;
        int i2;
        boolean z;
        Cursor cursor;
        a aVar;
        String str2;
        String str3;
        Cursor cursor2;
        a aVar2;
        String str4;
        a aVar3;
        String str5;
        String str6;
        String str7;
        String str8;
        List<com.google.android.gms.internal.measurement.zzek> list3;
        String str9;
        Cursor cursor3;
        Map map;
        Map map2;
        Iterator it;
        String str10;
        zzas zzasVar;
        zzw zzwVar;
        Iterator it2;
        zzas zzasVar2;
        String str11;
        Cursor cursor4;
        List list4;
        Iterator it3;
        String str12;
        String str13;
        Map map3;
        Cursor cursor5;
        Cursor cursor6;
        List list5;
        a aVar4;
        Cursor cursor7;
        List list6;
        String str14 = "current_results";
        Preconditions.checkNotEmpty(str);
        Preconditions.checkNotNull(list);
        Preconditions.checkNotNull(list2);
        this.zza = str;
        this.zzb = new HashSet();
        this.zzc = new a();
        this.zzd = l;
        this.zze = l2;
        Iterator it4 = list.iterator();
        while (true) {
            i = 0;
            i2 = 1;
            if (it4.hasNext()) {
                if ("_s".equals(((com.google.android.gms.internal.measurement.zzft) it4.next()).zzh())) {
                    z = true;
                    break;
                }
            } else {
                z = false;
                break;
            }
        }
        zznz.zzc();
        boolean zzs = this.zzt.zzf().zzs(this.zza, zzdu.zzW);
        zznz.zzc();
        boolean zzs2 = this.zzt.zzf().zzs(this.zza, zzdu.zzV);
        if (z) {
            zzam zzi = this.zzf.zzi();
            String str15 = this.zza;
            zzi.zzW();
            zzi.zzg();
            Preconditions.checkNotEmpty(str15);
            ContentValues contentValues = new ContentValues();
            ?? r5 = "current_session_count";
            contentValues.put("current_session_count", (Integer) 0);
            try {
                r5 = new String[]{str15};
                zzi.zzh().update("events", contentValues, "app_id = ?", r5);
                cursor = r5;
            } catch (SQLiteException e2) {
                zzi.zzt.zzay().zzd().zzc("Error resetting session-scoped event counts. appId", zzeh.zzn(str15), e2);
                cursor = r5;
            }
        }
        Map emptyMap = Collections.emptyMap();
        String str16 = "Failed to merge filter. appId";
        String str17 = "Database error querying filters. appId";
        String str18 = "data";
        String str19 = "audience_id";
        try {
            if (zzs2 && zzs) {
                zzam zzi2 = this.zzf.zzi();
                String str20 = this.zza;
                Preconditions.checkNotEmpty(str20);
                a aVar5 = new a();
                try {
                    try {
                        cursor7 = zzi2.zzh().query("event_filters", new String[]{"audience_id", "data"}, "app_id=?", new String[]{str20}, null, null, null);
                    } catch (SQLiteException e3) {
                        e = e3;
                        cursor7 = null;
                    } catch (Throwable th) {
                        th = th;
                        cursor = null;
                        if (cursor != null) {
                        }
                        throw th;
                    }
                    try {
                    } catch (SQLiteException e4) {
                        e = e4;
                        zzi2.zzt.zzay().zzd().zzc("Database error querying filters. appId", zzeh.zzn(str20), e);
                        emptyMap = Collections.emptyMap();
                    }
                    if (cursor7.moveToFirst()) {
                        while (true) {
                            try {
                                com.google.android.gms.internal.measurement.zzek zzekVar = (com.google.android.gms.internal.measurement.zzek) ((com.google.android.gms.internal.measurement.zzej) zzkv.zzl(com.google.android.gms.internal.measurement.zzek.zzc(), cursor7.getBlob(i2))).zzaC();
                                if (zzekVar.zzo()) {
                                    Integer valueOf = Integer.valueOf(cursor7.getInt(i));
                                    List list7 = (List) aVar5.get(valueOf);
                                    if (list7 == null) {
                                        list6 = new ArrayList();
                                        aVar5.put(valueOf, list6);
                                    } else {
                                        list6 = list7;
                                    }
                                    list6.add(zzekVar);
                                }
                            } catch (IOException e5) {
                                zzi2.zzt.zzay().zzd().zzc("Failed to merge filter. appId", zzeh.zzn(str20), e5);
                            }
                            if (!cursor7.moveToNext()) {
                                break;
                            }
                            i = 0;
                            i2 = 1;
                        }
                        cursor7.close();
                        aVar = aVar5;
                        zzam zzi3 = this.zzf.zzi();
                        String str21 = this.zza;
                        zzi3.zzW();
                        zzi3.zzg();
                        Preconditions.checkNotEmpty(str21);
                        cursor2 = zzi3.zzh().query("audience_filter_values", new String[]{"audience_id", "current_results"}, "app_id=?", new String[]{str21}, null, null, null);
                        if (cursor2.moveToFirst()) {
                            Map emptyMap2 = Collections.emptyMap();
                            cursor2.close();
                            aVar2 = emptyMap2;
                            str2 = "audience_id";
                            str3 = "data";
                        } else {
                            a aVar6 = new a();
                            while (true) {
                                int i3 = cursor2.getInt(0);
                                try {
                                    aVar6.put(Integer.valueOf(i3), (com.google.android.gms.internal.measurement.zzgi) ((com.google.android.gms.internal.measurement.zzgh) zzkv.zzl(com.google.android.gms.internal.measurement.zzgi.zzf(), cursor2.getBlob(1))).zzaC());
                                    aVar4 = aVar6;
                                    str2 = str19;
                                    str3 = str18;
                                } catch (IOException e6) {
                                    aVar4 = aVar6;
                                    str2 = str19;
                                    try {
                                        str3 = str18;
                                    } catch (SQLiteException e7) {
                                        e = e7;
                                        str3 = str18;
                                        zzi3.zzt.zzay().zzd().zzc("Database error querying filter results. appId", zzeh.zzn(str21), e);
                                        Map emptyMap3 = Collections.emptyMap();
                                        if (cursor2 != null) {
                                        }
                                        aVar2 = emptyMap3;
                                        if (aVar2.isEmpty()) {
                                        }
                                        if (!list.isEmpty()) {
                                        }
                                        String str22 = str14;
                                        if (!list2.isEmpty()) {
                                        }
                                        String str23 = str6;
                                        ArrayList arrayList = new ArrayList();
                                        Set<Integer> keySet = this.zzc.keySet();
                                        keySet.removeAll(this.zzb);
                                        while (r2.hasNext()) {
                                        }
                                        return arrayList;
                                    }
                                    try {
                                        zzi3.zzt.zzay().zzd().zzd("Failed to merge filter results. appId, audienceId, error", zzeh.zzn(str21), Integer.valueOf(i3), e6);
                                    } catch (SQLiteException e8) {
                                        e = e8;
                                        zzi3.zzt.zzay().zzd().zzc("Database error querying filter results. appId", zzeh.zzn(str21), e);
                                        Map emptyMap32 = Collections.emptyMap();
                                        if (cursor2 != null) {
                                            cursor2.close();
                                        }
                                        aVar2 = emptyMap32;
                                        if (aVar2.isEmpty()) {
                                        }
                                        if (!list.isEmpty()) {
                                        }
                                        String str222 = str14;
                                        if (!list2.isEmpty()) {
                                        }
                                        String str232 = str6;
                                        ArrayList arrayList2 = new ArrayList();
                                        Set<Integer> keySet2 = this.zzc.keySet();
                                        keySet2.removeAll(this.zzb);
                                        while (r2.hasNext()) {
                                        }
                                        return arrayList2;
                                    }
                                }
                                if (!cursor2.moveToNext()) {
                                    break;
                                }
                                aVar6 = aVar4;
                                str19 = str2;
                                str18 = str3;
                            }
                            cursor2.close();
                            aVar2 = aVar4;
                        }
                        if (aVar2.isEmpty()) {
                            HashSet hashSet = new HashSet(aVar2.keySet());
                            if (z) {
                                String str24 = this.zza;
                                zzam zzi4 = this.zzf.zzi();
                                String str25 = this.zza;
                                zzi4.zzW();
                                zzi4.zzg();
                                Preconditions.checkNotEmpty(str25);
                                ?? aVar7 = new a();
                                ?? zzh = zzi4.zzh();
                                try {
                                    try {
                                        cursor3 = zzh.rawQuery("select audience_id, filter_id from event_filters where app_id = ? and session_scoped = 1 UNION select audience_id, filter_id from property_filters where app_id = ? and session_scoped = 1;", new String[]{str25, str25});
                                    } catch (SQLiteException e9) {
                                        e = e9;
                                        cursor3 = null;
                                    } catch (Throwable th2) {
                                        th = th2;
                                        zzh = 0;
                                        if (zzh != 0) {
                                        }
                                        throw th;
                                    }
                                    try {
                                        if (cursor3.moveToFirst()) {
                                            do {
                                                Integer valueOf2 = Integer.valueOf(cursor3.getInt(0));
                                                List list8 = (List) aVar7.get(valueOf2);
                                                if (list8 == null) {
                                                    list8 = new ArrayList();
                                                    aVar7.put(valueOf2, list8);
                                                }
                                                list8.add(Integer.valueOf(cursor3.getInt(1)));
                                            } while (cursor3.moveToNext());
                                        } else {
                                            aVar7 = Collections.emptyMap();
                                        }
                                    } catch (SQLiteException e10) {
                                        e = e10;
                                        zzi4.zzt.zzay().zzd().zzc("Database error querying scoped filters. appId", zzeh.zzn(str25), e);
                                        aVar7 = Collections.emptyMap();
                                        map = aVar7;
                                    }
                                    cursor3.close();
                                    map = aVar7;
                                    Preconditions.checkNotEmpty(str24);
                                    Preconditions.checkNotNull(aVar2);
                                    a aVar8 = new a();
                                    if (!aVar2.isEmpty()) {
                                        Iterator it5 = aVar2.keySet().iterator();
                                        while (it5.hasNext()) {
                                            int intValue = ((Integer) it5.next()).intValue();
                                            Integer valueOf3 = Integer.valueOf(intValue);
                                            com.google.android.gms.internal.measurement.zzgi zzgiVar = (com.google.android.gms.internal.measurement.zzgi) aVar2.get(valueOf3);
                                            List list9 = (List) map.get(valueOf3);
                                            if (list9 != null && !list9.isEmpty()) {
                                                map2 = map;
                                                List zzq = this.zzf.zzu().zzq(zzgiVar.zzk(), list9);
                                                if (zzq.isEmpty()) {
                                                    map = map2;
                                                } else {
                                                    com.google.android.gms.internal.measurement.zzgh zzghVar = (com.google.android.gms.internal.measurement.zzgh) zzgiVar.zzby();
                                                    zzghVar.zzf();
                                                    zzghVar.zzb(zzq);
                                                    it = it5;
                                                    List zzq2 = this.zzf.zzu().zzq(zzgiVar.zzn(), list9);
                                                    zzghVar.zzh();
                                                    zzghVar.zzd(zzq2);
                                                    zzoc.zzc();
                                                    str10 = str17;
                                                    if (this.zzt.zzf().zzs(null, zzdu.zzas)) {
                                                        ArrayList arrayList3 = new ArrayList();
                                                        Iterator it6 = zzgiVar.zzj().iterator();
                                                        while (it6.hasNext()) {
                                                            com.google.android.gms.internal.measurement.zzfr zzfrVar = (com.google.android.gms.internal.measurement.zzfr) it6.next();
                                                            Iterator it7 = it6;
                                                            if (!list9.contains(Integer.valueOf(zzfrVar.zza()))) {
                                                                arrayList3.add(zzfrVar);
                                                            }
                                                            it6 = it7;
                                                        }
                                                        zzghVar.zze();
                                                        zzghVar.zza(arrayList3);
                                                        ArrayList arrayList4 = new ArrayList();
                                                        for (com.google.android.gms.internal.measurement.zzgk zzgkVar : zzgiVar.zzm()) {
                                                            if (!list9.contains(Integer.valueOf(zzgkVar.zzb()))) {
                                                                arrayList4.add(zzgkVar);
                                                            }
                                                        }
                                                        zzghVar.zzg();
                                                        zzghVar.zzc(arrayList4);
                                                    } else {
                                                        for (int i4 = 0; i4 < zzgiVar.zza(); i4++) {
                                                            if (list9.contains(Integer.valueOf(zzgiVar.zze(i4).zza()))) {
                                                                zzghVar.zzi(i4);
                                                            }
                                                        }
                                                        for (int i5 = 0; i5 < zzgiVar.zzc(); i5++) {
                                                            if (list9.contains(Integer.valueOf(zzgiVar.zzi(i5).zzb()))) {
                                                                zzghVar.zzj(i5);
                                                            }
                                                        }
                                                    }
                                                    aVar8.put(Integer.valueOf(intValue), (com.google.android.gms.internal.measurement.zzgi) zzghVar.zzaC());
                                                }
                                            } else {
                                                map2 = map;
                                                it = it5;
                                                str10 = str17;
                                                aVar8.put(valueOf3, zzgiVar);
                                            }
                                            map = map2;
                                            it5 = it;
                                            str17 = str10;
                                        }
                                    }
                                    str4 = str17;
                                    aVar3 = aVar8;
                                } catch (Throwable th3) {
                                    th = th3;
                                    if (zzh != 0) {
                                        zzh.close();
                                    }
                                    throw th;
                                }
                            } else {
                                str4 = "Database error querying filters. appId";
                                aVar3 = aVar2;
                            }
                            Iterator it8 = hashSet.iterator();
                            while (it8.hasNext()) {
                                int intValue2 = ((Integer) it8.next()).intValue();
                                com.google.android.gms.internal.measurement.zzgi zzgiVar2 = (com.google.android.gms.internal.measurement.zzgi) aVar3.get(Integer.valueOf(intValue2));
                                BitSet bitSet = new BitSet();
                                BitSet bitSet2 = new BitSet();
                                a aVar9 = new a();
                                if (zzgiVar2 != null && zzgiVar2.zza() != 0) {
                                    for (com.google.android.gms.internal.measurement.zzfr zzfrVar2 : zzgiVar2.zzj()) {
                                        if (zzfrVar2.zzh()) {
                                            aVar9.put(Integer.valueOf(zzfrVar2.zza()), zzfrVar2.zzg() ? Long.valueOf(zzfrVar2.zzb()) : null);
                                        }
                                    }
                                }
                                a aVar10 = new a();
                                if (zzgiVar2 != null && zzgiVar2.zzc() != 0) {
                                    for (com.google.android.gms.internal.measurement.zzgk zzgkVar2 : zzgiVar2.zzm()) {
                                        if (zzgkVar2.zzi() && zzgkVar2.zza() > 0) {
                                            aVar10.put(Integer.valueOf(zzgkVar2.zzb()), Long.valueOf(zzgkVar2.zzc(zzgkVar2.zza() - 1)));
                                            aVar3 = aVar3;
                                        }
                                    }
                                }
                                Map map4 = aVar3;
                                if (zzgiVar2 != null) {
                                    int i6 = 0;
                                    while (i6 < zzgiVar2.zzd() * 64) {
                                        if (zzkv.zzv(zzgiVar2.zzn(), i6)) {
                                            str9 = str16;
                                            this.zzt.zzay().zzj().zzc("Filter already evaluated. audience ID, filter ID", Integer.valueOf(intValue2), Integer.valueOf(i6));
                                            bitSet2.set(i6);
                                            if (zzkv.zzv(zzgiVar2.zzk(), i6)) {
                                                bitSet.set(i6);
                                                i6++;
                                                str16 = str9;
                                            }
                                        } else {
                                            str9 = str16;
                                        }
                                        aVar9.remove(Integer.valueOf(i6));
                                        i6++;
                                        str16 = str9;
                                    }
                                }
                                String str26 = str16;
                                Integer valueOf4 = Integer.valueOf(intValue2);
                                com.google.android.gms.internal.measurement.zzgi zzgiVar3 = (com.google.android.gms.internal.measurement.zzgi) aVar2.get(valueOf4);
                                if (zzs2 && zzs && (list3 = (List) aVar.get(valueOf4)) != null && this.zze != null && this.zzd != null) {
                                    for (com.google.android.gms.internal.measurement.zzek zzekVar2 : list3) {
                                        int zzb = zzekVar2.zzb();
                                        long longValue = this.zze.longValue() / 1000;
                                        if (zzekVar2.zzm()) {
                                            longValue = this.zzd.longValue() / 1000;
                                        }
                                        Integer valueOf5 = Integer.valueOf(zzb);
                                        if (aVar9.containsKey(valueOf5)) {
                                            aVar9.put(valueOf5, Long.valueOf(longValue));
                                        }
                                        if (aVar10.containsKey(valueOf5)) {
                                            aVar10.put(valueOf5, Long.valueOf(longValue));
                                        }
                                    }
                                }
                                this.zzc.put(Integer.valueOf(intValue2), new zzu(this, this.zza, zzgiVar3, bitSet, bitSet2, aVar9, aVar10, null));
                                str16 = str26;
                                aVar = aVar;
                                aVar3 = map4;
                                aVar2 = aVar2;
                            }
                            str5 = str16;
                            str6 = str2;
                            str7 = str3;
                            str8 = str4;
                        } else {
                            str8 = "Database error querying filters. appId";
                            str5 = "Failed to merge filter. appId";
                            str6 = str2;
                            str7 = str3;
                        }
                        if (!list.isEmpty()) {
                            zzw zzwVar2 = new zzw(this, null);
                            a aVar11 = new a();
                            Iterator it9 = list.iterator();
                            while (it9.hasNext()) {
                                com.google.android.gms.internal.measurement.zzft zzftVar = (com.google.android.gms.internal.measurement.zzft) it9.next();
                                com.google.android.gms.internal.measurement.zzft zza = zzwVar2.zza(this.zza, zzftVar);
                                if (zza != null) {
                                    zzam zzi5 = this.zzf.zzi();
                                    String str27 = this.zza;
                                    String zzh2 = zza.zzh();
                                    zzas zzn = zzi5.zzn(str27, zzftVar.zzh());
                                    if (zzn == null) {
                                        zzi5.zzt.zzay().zzk().zzc("Event aggregate wasn't created during raw event logging. appId, event", zzeh.zzn(str27), zzi5.zzt.zzj().zzd(zzh2));
                                        zzasVar = new zzas(str27, zzftVar.zzh(), 1L, 1L, 1L, zzftVar.zzd(), 0L, null, null, null, null);
                                    } else {
                                        zzasVar = new zzas(zzn.zza, zzn.zzb, zzn.zzc + 1, zzn.zzd + 1, zzn.zze + 1, zzn.zzf, zzn.zzg, zzn.zzh, zzn.zzi, zzn.zzj, zzn.zzk);
                                    }
                                    this.zzf.zzi().zzE(zzasVar);
                                    long j = zzasVar.zzc;
                                    String zzh3 = zza.zzh();
                                    Map map5 = (Map) aVar11.get(zzh3);
                                    if (map5 == null) {
                                        zzam zzi6 = this.zzf.zzi();
                                        String str28 = this.zza;
                                        zzi6.zzW();
                                        zzi6.zzg();
                                        Preconditions.checkNotEmpty(str28);
                                        Preconditions.checkNotEmpty(zzh3);
                                        zzwVar = zzwVar2;
                                        a aVar12 = new a();
                                        it2 = it9;
                                        str11 = str14;
                                        String str29 = str6;
                                        String str30 = str7;
                                        try {
                                            try {
                                                str7 = str30;
                                                try {
                                                    cursor4 = zzi6.zzh().query("event_filters", new String[]{str29, str30}, "app_id=? AND event_name=?", new String[]{str28, zzh3}, null, null, null);
                                                    try {
                                                        try {
                                                        } catch (SQLiteException e11) {
                                                            e = e11;
                                                            zzasVar2 = zzasVar;
                                                            str6 = str29;
                                                        }
                                                    } catch (Throwable th4) {
                                                        th = th4;
                                                        if (cursor4 != null) {
                                                            cursor4.close();
                                                        }
                                                        throw th;
                                                    }
                                                } catch (SQLiteException e12) {
                                                    e = e12;
                                                    zzasVar2 = zzasVar;
                                                    str6 = str29;
                                                    cursor4 = null;
                                                    zzi6.zzt.zzay().zzd().zzc(str8, zzeh.zzn(str28), e);
                                                    map5 = Collections.emptyMap();
                                                }
                                            } catch (Throwable th5) {
                                                th = th5;
                                                cursor4 = null;
                                            }
                                        } catch (SQLiteException e13) {
                                            e = e13;
                                            str7 = str30;
                                        }
                                        if (cursor4.moveToFirst()) {
                                            str6 = str29;
                                            while (true) {
                                                try {
                                                    try {
                                                        com.google.android.gms.internal.measurement.zzek zzekVar3 = (com.google.android.gms.internal.measurement.zzek) ((com.google.android.gms.internal.measurement.zzej) zzkv.zzl(com.google.android.gms.internal.measurement.zzek.zzc(), cursor4.getBlob(1))).zzaC();
                                                        Integer valueOf6 = Integer.valueOf(cursor4.getInt(0));
                                                        List list10 = (List) aVar12.get(valueOf6);
                                                        if (list10 == null) {
                                                            zzasVar2 = zzasVar;
                                                            try {
                                                                list4 = new ArrayList();
                                                                aVar12.put(valueOf6, list4);
                                                            } catch (SQLiteException e14) {
                                                                e = e14;
                                                                zzi6.zzt.zzay().zzd().zzc(str8, zzeh.zzn(str28), e);
                                                                map5 = Collections.emptyMap();
                                                            }
                                                        } else {
                                                            zzasVar2 = zzasVar;
                                                            list4 = list10;
                                                        }
                                                        list4.add(zzekVar3);
                                                    } catch (IOException e15) {
                                                        zzasVar2 = zzasVar;
                                                        zzi6.zzt.zzay().zzd().zzc(str5, zzeh.zzn(str28), e15);
                                                    }
                                                    if (!cursor4.moveToNext()) {
                                                        break;
                                                    }
                                                    zzasVar = zzasVar2;
                                                } catch (SQLiteException e16) {
                                                    e = e16;
                                                    zzasVar2 = zzasVar;
                                                }
                                            }
                                            cursor4.close();
                                            map5 = aVar12;
                                            aVar11.put(zzh3, map5);
                                        } else {
                                            zzasVar2 = zzasVar;
                                            str6 = str29;
                                            map5 = Collections.emptyMap();
                                            cursor4.close();
                                            aVar11.put(zzh3, map5);
                                        }
                                    } else {
                                        zzwVar = zzwVar2;
                                        it2 = it9;
                                        zzasVar2 = zzasVar;
                                        str11 = str14;
                                    }
                                    for (Integer num : map5.keySet()) {
                                        int intValue3 = num.intValue();
                                        Set set = this.zzb;
                                        Integer valueOf7 = Integer.valueOf(intValue3);
                                        if (set.contains(valueOf7)) {
                                            this.zzt.zzay().zzj().zzb("Skipping failed audience ID", valueOf7);
                                        } else {
                                            Iterator it10 = ((List) map5.get(valueOf7)).iterator();
                                            boolean z2 = true;
                                            while (true) {
                                                if (!it10.hasNext()) {
                                                    break;
                                                }
                                                com.google.android.gms.internal.measurement.zzek zzekVar4 = (com.google.android.gms.internal.measurement.zzek) it10.next();
                                                zzx zzxVar = new zzx(this, this.zza, intValue3, zzekVar4);
                                                z2 = zzxVar.zzd(this.zzd, this.zze, zza, j, zzasVar2, zzf(intValue3, zzekVar4.zzb()));
                                                if (z2) {
                                                    zzd(Integer.valueOf(intValue3)).zzc(zzxVar);
                                                } else {
                                                    this.zzb.add(Integer.valueOf(intValue3));
                                                    break;
                                                }
                                            }
                                            if (!z2) {
                                                this.zzb.add(Integer.valueOf(intValue3));
                                            }
                                        }
                                    }
                                    zzwVar2 = zzwVar;
                                    it9 = it2;
                                    str14 = str11;
                                }
                            }
                        }
                        String str2222 = str14;
                        if (!list2.isEmpty()) {
                            a aVar13 = new a();
                            Iterator it11 = list2.iterator();
                            while (it11.hasNext()) {
                                com.google.android.gms.internal.measurement.zzgm zzgmVar = (com.google.android.gms.internal.measurement.zzgm) it11.next();
                                String zzf = zzgmVar.zzf();
                                Map map6 = (Map) aVar13.get(zzf);
                                if (map6 == null) {
                                    zzam zzi7 = this.zzf.zzi();
                                    String str31 = this.zza;
                                    zzi7.zzW();
                                    zzi7.zzg();
                                    Preconditions.checkNotEmpty(str31);
                                    Preconditions.checkNotEmpty(zzf);
                                    a aVar14 = new a();
                                    str12 = str6;
                                    str13 = str7;
                                    try {
                                        cursor6 = zzi7.zzh().query("property_filters", new String[]{str12, str13}, "app_id=? AND property_name=?", new String[]{str31, zzf}, null, null, null);
                                    } catch (SQLiteException e17) {
                                        e = e17;
                                        it3 = it11;
                                        cursor6 = null;
                                    } catch (Throwable th6) {
                                        th = th6;
                                        cursor5 = null;
                                    }
                                    try {
                                        try {
                                        } catch (SQLiteException e18) {
                                            e = e18;
                                            it3 = it11;
                                        }
                                        if (cursor6.moveToFirst()) {
                                            while (true) {
                                                try {
                                                    com.google.android.gms.internal.measurement.zzet zzetVar = (com.google.android.gms.internal.measurement.zzet) ((com.google.android.gms.internal.measurement.zzes) zzkv.zzl(com.google.android.gms.internal.measurement.zzet.zzc(), cursor6.getBlob(1))).zzaC();
                                                    Integer valueOf8 = Integer.valueOf(cursor6.getInt(0));
                                                    List list11 = (List) aVar14.get(valueOf8);
                                                    if (list11 == null) {
                                                        list5 = new ArrayList();
                                                        aVar14.put(valueOf8, list5);
                                                    } else {
                                                        list5 = list11;
                                                    }
                                                    list5.add(zzetVar);
                                                    it3 = it11;
                                                } catch (IOException e19) {
                                                    it3 = it11;
                                                    try {
                                                        zzi7.zzt.zzay().zzd().zzc("Failed to merge filter", zzeh.zzn(str31), e19);
                                                    } catch (SQLiteException e20) {
                                                        e = e20;
                                                        zzi7.zzt.zzay().zzd().zzc(str8, zzeh.zzn(str31), e);
                                                        map6 = Collections.emptyMap();
                                                    }
                                                }
                                                if (!cursor6.moveToNext()) {
                                                    break;
                                                }
                                                it11 = it3;
                                            }
                                            cursor6.close();
                                            map6 = aVar14;
                                            aVar13.put(zzf, map6);
                                        } else {
                                            it3 = it11;
                                            map6 = Collections.emptyMap();
                                            cursor6.close();
                                            aVar13.put(zzf, map6);
                                        }
                                    } catch (Throwable th7) {
                                        th = th7;
                                        cursor5 = cursor6;
                                        if (cursor5 != null) {
                                            cursor5.close();
                                        }
                                        throw th;
                                    }
                                } else {
                                    it3 = it11;
                                    str12 = str6;
                                    str13 = str7;
                                }
                                Iterator it12 = map6.keySet().iterator();
                                while (true) {
                                    if (it12.hasNext()) {
                                        int intValue4 = ((Integer) it12.next()).intValue();
                                        Set set2 = this.zzb;
                                        Integer valueOf9 = Integer.valueOf(intValue4);
                                        if (set2.contains(valueOf9)) {
                                            this.zzt.zzay().zzj().zzb("Skipping failed audience ID", valueOf9);
                                            break;
                                        }
                                        Iterator it13 = ((List) map6.get(valueOf9)).iterator();
                                        boolean z3 = true;
                                        while (true) {
                                            if (!it13.hasNext()) {
                                                map3 = map6;
                                                break;
                                            }
                                            com.google.android.gms.internal.measurement.zzet zzetVar2 = (com.google.android.gms.internal.measurement.zzet) it13.next();
                                            if (Log.isLoggable(this.zzt.zzay().zzq(), 2)) {
                                                map3 = map6;
                                                this.zzt.zzay().zzj().zzd("Evaluating filter. audience, filter, property", Integer.valueOf(intValue4), zzetVar2.zzj() ? Integer.valueOf(zzetVar2.zza()) : null, this.zzt.zzj().zzf(zzetVar2.zze()));
                                                this.zzt.zzay().zzj().zzb("Filter definition", this.zzf.zzu().zzp(zzetVar2));
                                            } else {
                                                map3 = map6;
                                            }
                                            if (!zzetVar2.zzj() || zzetVar2.zza() > 256) {
                                                break;
                                            }
                                            zzz zzzVar = new zzz(this, this.zza, intValue4, zzetVar2);
                                            z3 = zzzVar.zzd(this.zzd, this.zze, zzgmVar, zzf(intValue4, zzetVar2.zza()));
                                            if (z3) {
                                                zzd(Integer.valueOf(intValue4)).zzc(zzzVar);
                                                map6 = map3;
                                            } else {
                                                this.zzb.add(Integer.valueOf(intValue4));
                                                break;
                                            }
                                        }
                                        if (z3) {
                                            map6 = map3;
                                        }
                                        this.zzb.add(Integer.valueOf(intValue4));
                                        map6 = map3;
                                    }
                                }
                                it11 = it3;
                                str7 = str13;
                                str6 = str12;
                            }
                        }
                        String str2322 = str6;
                        ArrayList arrayList22 = new ArrayList();
                        Set<Integer> keySet22 = this.zzc.keySet();
                        keySet22.removeAll(this.zzb);
                        for (Integer num2 : keySet22) {
                            int intValue5 = num2.intValue();
                            Map map7 = this.zzc;
                            Integer valueOf10 = Integer.valueOf(intValue5);
                            zzu zzuVar = (zzu) map7.get(valueOf10);
                            Preconditions.checkNotNull(zzuVar);
                            com.google.android.gms.internal.measurement.zzfp zza2 = zzuVar.zza(intValue5);
                            arrayList22.add(zza2);
                            zzam zzi8 = this.zzf.zzi();
                            String str32 = this.zza;
                            com.google.android.gms.internal.measurement.zzgi zzd = zza2.zzd();
                            zzi8.zzW();
                            zzi8.zzg();
                            Preconditions.checkNotEmpty(str32);
                            Preconditions.checkNotNull(zzd);
                            byte[] zzbu = zzd.zzbu();
                            ContentValues contentValues2 = new ContentValues();
                            contentValues2.put("app_id", str32);
                            contentValues2.put(str2322, valueOf10);
                            String str33 = str2222;
                            contentValues2.put(str33, zzbu);
                            try {
                            } catch (SQLiteException e21) {
                                e = e21;
                            }
                            try {
                                if (zzi8.zzh().insertWithOnConflict("audience_filter_values", null, contentValues2, 5) == -1) {
                                    zzi8.zzt.zzay().zzd().zzb("Failed to insert filter results (got -1). appId", zzeh.zzn(str32));
                                }
                            } catch (SQLiteException e22) {
                                e = e22;
                                zzi8.zzt.zzay().zzd().zzc("Error storing filter results. appId", zzeh.zzn(str32), e);
                                str2222 = str33;
                            }
                            str2222 = str33;
                        }
                        return arrayList22;
                    }
                    emptyMap = Collections.emptyMap();
                    cursor7.close();
                } catch (Throwable th8) {
                    th = th8;
                    if (cursor != null) {
                        cursor.close();
                    }
                    throw th;
                }
            }
            if (cursor2.moveToFirst()) {
            }
            if (aVar2.isEmpty()) {
            }
            if (!list.isEmpty()) {
            }
            String str22222 = str14;
            if (!list2.isEmpty()) {
            }
            String str23222 = str6;
            ArrayList arrayList222 = new ArrayList();
            Set<Integer> keySet222 = this.zzc.keySet();
            keySet222.removeAll(this.zzb);
            while (r2.hasNext()) {
            }
            return arrayList222;
        } catch (Throwable th9) {
            th = th9;
            Cursor cursor8 = cursor2;
            if (cursor8 != null) {
                cursor8.close();
            }
            throw th;
        }
        aVar = emptyMap;
        zzam zzi32 = this.zzf.zzi();
        String str212 = this.zza;
        zzi32.zzW();
        zzi32.zzg();
        Preconditions.checkNotEmpty(str212);
        cursor2 = zzi32.zzh().query("audience_filter_values", new String[]{"audience_id", "current_results"}, "app_id=?", new String[]{str212}, null, null, null);
    }

    @Override // com.google.android.gms.measurement.internal.zzkh
    public final boolean zzb() {
        return false;
    }
}