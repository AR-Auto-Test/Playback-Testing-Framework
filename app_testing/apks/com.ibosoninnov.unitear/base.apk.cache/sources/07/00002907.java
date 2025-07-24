package com.google.mediapipe.framework;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.provider.BaseColumns;
import com.google.common.flogger.FluentLogger;
import java.io.File;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AssetCacheDbHelper.class */
public class AssetCacheDbHelper extends SQLiteOpenHelper {
    private static final FluentLogger logger = FluentLogger.forEnclosingClass();
    public static final int DATABASE_VERSION = 2;
    public static final String DATABASE_NAME = "mediapipe.db";
    private static final String INT_TYPE = " INTEGER";
    private static final String TEXT_TYPE = " TEXT";
    private static final String TEXT_UNIQUE_TYPE = " TEXT NOT NULL UNIQUE";
    private static final String COMMA_SEP = ",";
    private static final String SQL_CREATE_TABLE = "CREATE TABLE AssetVersion (_id INTEGER PRIMARY KEY,asset TEXT NOT NULL UNIQUE,cache_path TEXT,version INTEGER )";
    private static final String SQL_DELETE_TABLE = "DROP TABLE IF EXISTS AssetVersion";

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AssetCacheDbHelper$AssetCacheEntry.class */
    public static abstract class AssetCacheEntry implements BaseColumns {
        public static final String TABLE_NAME = "AssetVersion";
        public static final String COLUMN_NAME_ASSET = "asset";
        public static final String COLUMN_NAME_CACHE_PATH = "cache_path";
        public static final String COLUMN_NAME_VERSION = "version";
    }

    public AssetCacheDbHelper(Context context) {
        super(context, "mediapipe.db", (SQLiteDatabase.CursorFactory) null, 2);
    }

    public boolean checkVersion(String assetPath, int currentAppVersion) {
        SQLiteDatabase db = getReadableDatabase();
        String[] projection = {"version"};
        String[] selectionArgs = {assetPath};
        Cursor cursor = queryAssetCacheTable(db, projection, "asset = ?", selectionArgs);
        if (cursor.getCount() == 0) {
            return false;
        }
        cursor.moveToFirst();
        int cachedVersion = cursor.getInt(cursor.getColumnIndexOrThrow("version"));
        cursor.close();
        return cachedVersion == currentAppVersion;
    }

    public void invalidateCache(int currentAppVersion) {
        SQLiteDatabase db = getWritableDatabase();
        String[] selectionArgs = {Integer.toString(currentAppVersion)};
        removeCachedFiles(db, "version != ?", selectionArgs);
        db.delete("AssetVersion", "version != ?", selectionArgs);
    }

    public void insertAsset(String asset, String cachePath, int appVersion) {
        SQLiteDatabase db = getWritableDatabase();
        String[] selectionArgs = {asset, cachePath};
        removeCachedFiles(db, "asset = ? and cache_path != ?", selectionArgs);
        ContentValues values = new ContentValues();
        values.put("asset", asset);
        values.put("cache_path", cachePath);
        values.put("version", Integer.valueOf(appVersion));
        long newRowId = db.insertWithOnConflict("AssetVersion", null, values, 5);
        if (newRowId == -1) {
            throw new RuntimeException("Can't insert entry into the mediapipe db.");
        }
    }

    @Override // android.database.sqlite.SQLiteOpenHelper
    public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE AssetVersion (_id INTEGER PRIMARY KEY,asset TEXT NOT NULL UNIQUE,cache_path TEXT,version INTEGER )");
    }

    @Override // android.database.sqlite.SQLiteOpenHelper
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS AssetVersion");
        onCreate(db);
    }

    @Override // android.database.sqlite.SQLiteOpenHelper
    public void onDowngrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        onUpgrade(db, oldVersion, newVersion);
    }

    private Cursor queryAssetCacheTable(SQLiteDatabase db, String[] projection, String selection, String[] selectionArgs) {
        return db.query("AssetVersion", projection, selection, selectionArgs, null, null, null);
    }

    private void removeCachedFiles(SQLiteDatabase db, String selection, String[] selectionArgs) {
        String[] projection = {"cache_path"};
        Cursor cursor = queryAssetCacheTable(db, projection, selection, selectionArgs);
        if (cursor.moveToFirst()) {
            do {
                String cachedPath = cursor.getString(cursor.getColumnIndexOrThrow("cache_path"));
                File file = new File(cachedPath);
                if (file.exists() && !file.delete()) {
                    logger.atWarning().log("Stale cached file: %s can't be deleted.", cachedPath);
                }
            } while (cursor.moveToNext());
            cursor.close();
        }
        cursor.close();
    }
}