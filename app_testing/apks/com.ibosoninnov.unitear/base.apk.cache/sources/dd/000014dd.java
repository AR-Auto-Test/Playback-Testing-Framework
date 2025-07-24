package com.google.android.gms.measurement.internal;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDatabaseLockedException;
import android.database.sqlite.SQLiteException;
import android.database.sqlite.SQLiteFullException;
import android.os.Parcel;
import android.os.SystemClock;
import c.b.a.a.a;
import com.google.android.gms.common.internal.safeparcel.SafeParcelReader;
import com.google.android.gms.common.util.VisibleForTesting;
import java.util.ArrayList;
import java.util.List;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzea extends zzf {
    private final zzdz zza;
    private boolean zzb;

    public zzea(zzfr zzfrVar) {
        super(zzfrVar);
        Context zzau = this.zzt.zzau();
        this.zzt.zzf();
        this.zza = new zzdz(this, zzau, "google_app_measurement_local.db");
    }

    /* JADX WARN: Removed duplicated region for block: B:81:0x012a  */
    /* JADX WARN: Removed duplicated region for block: B:83:0x012f  */
    /* JADX WARN: Type inference failed for: r2v1 */
    /* JADX WARN: Type inference failed for: r2v10 */
    /* JADX WARN: Type inference failed for: r2v2, types: [int, boolean] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final boolean zzq(int i, byte[] bArr) {
        SQLiteDatabase sQLiteDatabase;
        Cursor cursor;
        zzg();
        int i2 = 0;
        if (this.zzb) {
            return false;
        }
        ContentValues contentValues = new ContentValues();
        contentValues.put("type", Integer.valueOf(i));
        contentValues.put("entry", bArr);
        this.zzt.zzf();
        int i3 = 5;
        ?? r2 = 0;
        for (int i4 = 5; i2 < i4; i4 = 5) {
            Cursor cursor2 = null;
            cursor2 = null;
            r3 = null;
            Cursor cursor3 = null;
            cursor2 = null;
            Cursor cursor4 = null;
            SQLiteDatabase sQLiteDatabase2 = null;
            try {
                sQLiteDatabase = zzh();
                try {
                    if (sQLiteDatabase == null) {
                        this.zzb = true;
                        return r2;
                    }
                    sQLiteDatabase.beginTransaction();
                    Cursor rawQuery = sQLiteDatabase.rawQuery("select count(1) from messages", null);
                    long j = 0;
                    if (rawQuery != null) {
                        try {
                            if (rawQuery.moveToFirst()) {
                                j = rawQuery.getLong(r2);
                            }
                        } catch (SQLiteDatabaseLockedException unused) {
                            cursor2 = rawQuery;
                            try {
                                SystemClock.sleep(i3);
                                i3 += 20;
                                if (cursor2 != null) {
                                    cursor2.close();
                                }
                                if (sQLiteDatabase != null) {
                                    sQLiteDatabase.close();
                                }
                                i2++;
                                r2 = 0;
                            } catch (Throwable th) {
                                th = th;
                                if (cursor2 != null) {
                                }
                                if (sQLiteDatabase != null) {
                                }
                                throw th;
                            }
                        } catch (SQLiteFullException e2) {
                            e = e2;
                            cursor4 = rawQuery;
                            try {
                                this.zzt.zzay().zzd().zzb("Error writing entry; local database full", e);
                                this.zzb = true;
                                if (cursor4 != null) {
                                    cursor4.close();
                                }
                                if (sQLiteDatabase != null) {
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    sQLiteDatabase2.close();
                                    i2++;
                                    r2 = 0;
                                } else {
                                    i2++;
                                    r2 = 0;
                                }
                            } catch (Throwable th2) {
                                th = th2;
                                cursor = cursor4;
                                sQLiteDatabase2 = sQLiteDatabase;
                                sQLiteDatabase = sQLiteDatabase2;
                                cursor2 = cursor;
                                if (cursor2 != null) {
                                    cursor2.close();
                                }
                                if (sQLiteDatabase != null) {
                                    sQLiteDatabase.close();
                                }
                                throw th;
                            }
                        } catch (SQLiteException e3) {
                            e = e3;
                            cursor3 = rawQuery;
                            cursor = cursor3;
                            sQLiteDatabase2 = sQLiteDatabase;
                            if (sQLiteDatabase2 != null) {
                                try {
                                    if (sQLiteDatabase2.inTransaction()) {
                                        sQLiteDatabase2.endTransaction();
                                    }
                                } catch (Throwable th3) {
                                    th = th3;
                                    sQLiteDatabase = sQLiteDatabase2;
                                    cursor2 = cursor;
                                    if (cursor2 != null) {
                                    }
                                    if (sQLiteDatabase != null) {
                                    }
                                    throw th;
                                }
                            }
                            this.zzt.zzay().zzd().zzb("Error writing entry to local database", e);
                            this.zzb = true;
                            if (cursor != null) {
                                cursor.close();
                            }
                            if (sQLiteDatabase2 == null) {
                                i2++;
                                r2 = 0;
                            }
                            sQLiteDatabase2.close();
                            i2++;
                            r2 = 0;
                        } catch (Throwable th4) {
                            th = th4;
                            cursor2 = rawQuery;
                            if (cursor2 != null) {
                            }
                            if (sQLiteDatabase != null) {
                            }
                            throw th;
                        }
                    }
                    if (j >= 100000) {
                        this.zzt.zzay().zzd().zza("Data loss, local db full");
                        long j2 = (100000 - j) + 1;
                        String[] strArr = new String[1];
                        strArr[r2] = Long.toString(j2);
                        long delete = sQLiteDatabase.delete("messages", "rowid in (select rowid from messages order by rowid asc limit ?)", strArr);
                        if (delete != j2) {
                            this.zzt.zzay().zzd().zzd("Different delete count than expected in local db. expected, received, difference", Long.valueOf(j2), Long.valueOf(delete), Long.valueOf(j2 - delete));
                        }
                    }
                    sQLiteDatabase.insertOrThrow("messages", null, contentValues);
                    sQLiteDatabase.setTransactionSuccessful();
                    sQLiteDatabase.endTransaction();
                    if (rawQuery != 0) {
                        rawQuery.close();
                    }
                    sQLiteDatabase.close();
                    return true;
                } catch (SQLiteDatabaseLockedException unused2) {
                } catch (SQLiteFullException e4) {
                    e = e4;
                } catch (SQLiteException e5) {
                    e = e5;
                }
            } catch (SQLiteDatabaseLockedException unused3) {
                sQLiteDatabase = null;
            } catch (SQLiteFullException e6) {
                e = e6;
                sQLiteDatabase = null;
            } catch (SQLiteException e7) {
                e = e7;
                cursor = null;
            } catch (Throwable th5) {
                th = th5;
                sQLiteDatabase = null;
            }
        }
        a.F(this.zzt, "Failed to write entry to local database");
        return false;
    }

    @Override // com.google.android.gms.measurement.internal.zzf
    public final boolean zzf() {
        return false;
    }

    @VisibleForTesting
    public final SQLiteDatabase zzh() {
        if (this.zzb) {
            return null;
        }
        SQLiteDatabase writableDatabase = this.zza.getWritableDatabase();
        if (writableDatabase == null) {
            this.zzb = true;
            return null;
        }
        return writableDatabase;
    }

    /* JADX WARN: Removed duplicated region for block: B:142:0x0216  */
    /* JADX WARN: Removed duplicated region for block: B:150:0x0226  */
    /* JADX WARN: Removed duplicated region for block: B:157:0x0240  */
    /* JADX WARN: Removed duplicated region for block: B:165:0x0251  */
    /* JADX WARN: Removed duplicated region for block: B:167:0x0256  */
    /* JADX WARN: Removed duplicated region for block: B:178:0x01fc A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:190:0x01d8 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:205:0x0248 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:206:0x0248 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:207:0x0248 A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final List zzi(int i) {
        int i2;
        SQLiteDatabase sQLiteDatabase;
        Cursor cursor;
        SQLiteDatabase sQLiteDatabase2;
        Cursor cursor2;
        long j;
        String str;
        String[] strArr;
        zzkw zzkwVar;
        zzac zzacVar;
        zzg();
        Cursor cursor3 = null;
        if (this.zzb) {
            return null;
        }
        ArrayList arrayList = new ArrayList();
        if (zzl()) {
            int i3 = 5;
            for (i2 = 0; i2 < 5; i2 = i2 + 1) {
                try {
                    sQLiteDatabase2 = zzh();
                    if (sQLiteDatabase2 == null) {
                        this.zzb = true;
                        return null;
                    }
                    try {
                        sQLiteDatabase2.beginTransaction();
                        try {
                        } catch (Throwable th) {
                            th = th;
                            sQLiteDatabase = sQLiteDatabase2;
                        }
                        try {
                            cursor2 = sQLiteDatabase2.query("messages", new String[]{"rowid"}, "type=?", new String[]{"3"}, null, null, "rowid desc", "1");
                        } catch (Throwable th2) {
                            th = th2;
                            sQLiteDatabase = sQLiteDatabase2;
                            cursor2 = null;
                            if (cursor2 != null) {
                            }
                            throw th;
                            break;
                        }
                        try {
                            long j2 = -1;
                            if (cursor2.moveToFirst()) {
                                j = cursor2.getLong(0);
                                try {
                                    cursor2.close();
                                } catch (SQLiteDatabaseLockedException unused) {
                                    sQLiteDatabase = sQLiteDatabase2;
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    SystemClock.sleep(i3);
                                    i3 += 20;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (SQLiteFullException e2) {
                                    e = e2;
                                    sQLiteDatabase = sQLiteDatabase2;
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                    this.zzb = true;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (SQLiteException e3) {
                                    e = e3;
                                    sQLiteDatabase = sQLiteDatabase2;
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    if (sQLiteDatabase2 != null) {
                                    }
                                    this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                    this.zzb = true;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (Throwable th3) {
                                    th = th3;
                                    sQLiteDatabase = sQLiteDatabase2;
                                    if (cursor3 != null) {
                                    }
                                    if (sQLiteDatabase != null) {
                                    }
                                    throw th;
                                }
                            } else {
                                cursor2.close();
                                j = -1;
                            }
                            if (j != -1) {
                                str = "rowid<?";
                                strArr = new String[]{String.valueOf(j)};
                            } else {
                                str = null;
                                strArr = null;
                            }
                            cursor = sQLiteDatabase2.query("messages", new String[]{"rowid", "type", "entry"}, str, strArr, null, null, "rowid asc", Integer.toString(100));
                            while (cursor.moveToNext()) {
                                try {
                                    j2 = cursor.getLong(0);
                                    int i4 = cursor.getInt(1);
                                    byte[] blob = cursor.getBlob(2);
                                    if (i4 == 0) {
                                        Parcel obtain = Parcel.obtain();
                                        try {
                                            obtain.unmarshall(blob, 0, blob.length);
                                            obtain.setDataPosition(0);
                                            zzaw createFromParcel = zzaw.CREATOR.createFromParcel(obtain);
                                            obtain.recycle();
                                            if (createFromParcel != null) {
                                                arrayList.add(createFromParcel);
                                            }
                                        } catch (SafeParcelReader.ParseException unused2) {
                                            this.zzt.zzay().zzd().zza("Failed to load event from local database");
                                            obtain.recycle();
                                        }
                                    } else if (i4 == 1) {
                                        Parcel obtain2 = Parcel.obtain();
                                        try {
                                            obtain2.unmarshall(blob, 0, blob.length);
                                            obtain2.setDataPosition(0);
                                            zzkwVar = zzkw.CREATOR.createFromParcel(obtain2);
                                            obtain2.recycle();
                                        } catch (SafeParcelReader.ParseException unused3) {
                                            this.zzt.zzay().zzd().zza("Failed to load user property from local database");
                                            obtain2.recycle();
                                            zzkwVar = null;
                                        }
                                        if (zzkwVar != null) {
                                            arrayList.add(zzkwVar);
                                        }
                                    } else if (i4 == 2) {
                                        Parcel obtain3 = Parcel.obtain();
                                        try {
                                            obtain3.unmarshall(blob, 0, blob.length);
                                            obtain3.setDataPosition(0);
                                            zzacVar = zzac.CREATOR.createFromParcel(obtain3);
                                            obtain3.recycle();
                                        } catch (SafeParcelReader.ParseException unused4) {
                                            this.zzt.zzay().zzd().zza("Failed to load conditional user property from local database");
                                            obtain3.recycle();
                                            zzacVar = null;
                                        }
                                        if (zzacVar != null) {
                                            arrayList.add(zzacVar);
                                        }
                                    } else if (i4 == 3) {
                                        this.zzt.zzay().zzk().zza("Skipping app launch break");
                                    } else {
                                        this.zzt.zzay().zzd().zza("Unknown record type in local database");
                                    }
                                } catch (SQLiteDatabaseLockedException unused5) {
                                    sQLiteDatabase = sQLiteDatabase2;
                                } catch (SQLiteFullException e4) {
                                    e = e4;
                                    sQLiteDatabase = sQLiteDatabase2;
                                } catch (SQLiteException e5) {
                                    e = e5;
                                    sQLiteDatabase = sQLiteDatabase2;
                                } catch (Throwable th4) {
                                    th = th4;
                                    sQLiteDatabase = sQLiteDatabase2;
                                }
                            }
                            sQLiteDatabase = sQLiteDatabase2;
                            try {
                                if (sQLiteDatabase.delete("messages", "rowid <= ?", new String[]{Long.toString(j2)}) < arrayList.size()) {
                                    this.zzt.zzay().zzd().zza("Fewer entries removed from local database than expected");
                                }
                                sQLiteDatabase.setTransactionSuccessful();
                                sQLiteDatabase.endTransaction();
                                cursor.close();
                                sQLiteDatabase.close();
                                return arrayList;
                            } catch (SQLiteDatabaseLockedException unused6) {
                                sQLiteDatabase2 = sQLiteDatabase;
                                SystemClock.sleep(i3);
                                i3 += 20;
                                if (cursor != null) {
                                    cursor.close();
                                }
                                i2 = sQLiteDatabase2 == null ? i2 + 1 : 0;
                                sQLiteDatabase2.close();
                            } catch (SQLiteFullException e6) {
                                e = e6;
                                sQLiteDatabase2 = sQLiteDatabase;
                                this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                this.zzb = true;
                                if (cursor != null) {
                                    cursor.close();
                                }
                                if (sQLiteDatabase2 == null) {
                                }
                                sQLiteDatabase2.close();
                            } catch (SQLiteException e7) {
                                e = e7;
                                sQLiteDatabase2 = sQLiteDatabase;
                                if (sQLiteDatabase2 != null) {
                                    try {
                                        if (sQLiteDatabase2.inTransaction()) {
                                            sQLiteDatabase2.endTransaction();
                                        }
                                    } catch (Throwable th5) {
                                        th = th5;
                                        cursor3 = cursor;
                                        sQLiteDatabase = sQLiteDatabase2;
                                        if (cursor3 != null) {
                                            cursor3.close();
                                        }
                                        if (sQLiteDatabase != null) {
                                            sQLiteDatabase.close();
                                        }
                                        throw th;
                                    }
                                }
                                this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                this.zzb = true;
                                if (cursor != null) {
                                    cursor.close();
                                }
                                if (sQLiteDatabase2 == null) {
                                }
                                sQLiteDatabase2.close();
                            } catch (Throwable th6) {
                                th = th6;
                                cursor3 = cursor;
                                if (cursor3 != null) {
                                }
                                if (sQLiteDatabase != null) {
                                }
                                throw th;
                            }
                        } catch (Throwable th7) {
                            th = th7;
                            sQLiteDatabase = sQLiteDatabase2;
                            if (cursor2 != null) {
                                try {
                                    cursor2.close();
                                } catch (SQLiteDatabaseLockedException unused7) {
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    SystemClock.sleep(i3);
                                    i3 += 20;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (SQLiteFullException e8) {
                                    e = e8;
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                    this.zzb = true;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (SQLiteException e9) {
                                    e = e9;
                                    cursor = null;
                                    sQLiteDatabase2 = sQLiteDatabase;
                                    if (sQLiteDatabase2 != null) {
                                    }
                                    this.zzt.zzay().zzd().zzb("Error reading entries from local database", e);
                                    this.zzb = true;
                                    if (cursor != null) {
                                    }
                                    if (sQLiteDatabase2 == null) {
                                    }
                                    sQLiteDatabase2.close();
                                } catch (Throwable th8) {
                                    th = th8;
                                    if (cursor3 != null) {
                                    }
                                    if (sQLiteDatabase != null) {
                                    }
                                    throw th;
                                }
                            }
                            throw th;
                            break;
                            break;
                        }
                    } catch (SQLiteDatabaseLockedException unused8) {
                        sQLiteDatabase = sQLiteDatabase2;
                    } catch (SQLiteFullException e10) {
                        e = e10;
                        sQLiteDatabase = sQLiteDatabase2;
                    } catch (SQLiteException e11) {
                        e = e11;
                        sQLiteDatabase = sQLiteDatabase2;
                    } catch (Throwable th9) {
                        th = th9;
                        sQLiteDatabase = sQLiteDatabase2;
                        if (cursor3 != null) {
                        }
                        if (sQLiteDatabase != null) {
                        }
                        throw th;
                    }
                } catch (SQLiteDatabaseLockedException unused9) {
                    cursor = null;
                    sQLiteDatabase2 = null;
                } catch (SQLiteFullException e12) {
                    e = e12;
                    cursor = null;
                    sQLiteDatabase2 = null;
                } catch (SQLiteException e13) {
                    e = e13;
                    cursor = null;
                    sQLiteDatabase2 = null;
                } catch (Throwable th10) {
                    th = th10;
                    sQLiteDatabase = null;
                }
            }
            a.G(this.zzt, "Failed to read events from database in reasonable time");
            return null;
        }
        return arrayList;
    }

    public final void zzj() {
        int delete;
        zzg();
        try {
            SQLiteDatabase zzh = zzh();
            if (zzh == null || (delete = zzh.delete("messages", null, null)) <= 0) {
                return;
            }
            this.zzt.zzay().zzj().zzb("Reset local analytics data. records", Integer.valueOf(delete));
        } catch (SQLiteException e2) {
            this.zzt.zzay().zzd().zzb("Error resetting local analytics data. error", e2);
        }
    }

    public final boolean zzk() {
        return zzq(3, new byte[0]);
    }

    @VisibleForTesting
    public final boolean zzl() {
        Context zzau = this.zzt.zzau();
        this.zzt.zzf();
        return zzau.getDatabasePath("google_app_measurement_local.db").exists();
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[IF]}, finally: {[IF, INVOKE] complete} */
    public final boolean zzm() {
        int i;
        zzg();
        if (!this.zzb && zzl()) {
            int i2 = 5;
            for (i = 0; i < 5; i = i + 1) {
                SQLiteDatabase sQLiteDatabase = null;
                try {
                    try {
                        SQLiteDatabase zzh = zzh();
                        if (zzh == null) {
                            this.zzb = true;
                            return false;
                        }
                        zzh.beginTransaction();
                        zzh.delete("messages", "type == ?", new String[]{Integer.toString(3)});
                        zzh.setTransactionSuccessful();
                        zzh.endTransaction();
                        zzh.close();
                        return true;
                    } catch (SQLiteFullException e2) {
                        this.zzt.zzay().zzd().zzb("Error deleting app launch break from local database", e2);
                        this.zzb = true;
                        i = 0 == 0 ? i + 1 : 0;
                        sQLiteDatabase.close();
                    } catch (SQLiteException e3) {
                        if (0 != 0) {
                            try {
                                if (sQLiteDatabase.inTransaction()) {
                                    sQLiteDatabase.endTransaction();
                                }
                            } catch (Throwable th) {
                                if (0 != 0) {
                                    sQLiteDatabase.close();
                                }
                                throw th;
                            }
                        }
                        this.zzt.zzay().zzd().zzb("Error deleting app launch break from local database", e3);
                        this.zzb = true;
                        if (0 != 0) {
                            sQLiteDatabase.close();
                        }
                    }
                } catch (SQLiteDatabaseLockedException unused) {
                    SystemClock.sleep(i2);
                    i2 += 20;
                    if (0 != 0) {
                        sQLiteDatabase.close();
                    }
                }
            }
            a.G(this.zzt, "Error deleting app launch break from local database in reasonable time");
        }
        return false;
    }

    public final boolean zzn(zzac zzacVar) {
        byte[] zzan = this.zzt.zzv().zzan(zzacVar);
        if (zzan.length > 131072) {
            this.zzt.zzay().zzh().zza("Conditional user property too long for local database. Sending directly to service");
            return false;
        }
        return zzq(2, zzan);
    }

    public final boolean zzo(zzaw zzawVar) {
        Parcel obtain = Parcel.obtain();
        zzax.zza(zzawVar, obtain, 0);
        byte[] marshall = obtain.marshall();
        obtain.recycle();
        if (marshall.length > 131072) {
            this.zzt.zzay().zzh().zza("Event is too long for local database. Sending event directly to service");
            return false;
        }
        return zzq(0, marshall);
    }

    public final boolean zzp(zzkw zzkwVar) {
        Parcel obtain = Parcel.obtain();
        zzkx.zza(zzkwVar, obtain, 0);
        byte[] marshall = obtain.marshall();
        obtain.recycle();
        if (marshall.length > 131072) {
            this.zzt.zzay().zzh().zza("User property too long for local database. Sending directly to service");
            return false;
        }
        return zzq(1, marshall);
    }
}