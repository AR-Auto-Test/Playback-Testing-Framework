package com.google.mediapipe.framework;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.text.TextUtils;
import androidx.annotation.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.flogger.FluentLogger;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/AssetCache.class */
public class AssetCache {
    private static final FluentLogger logger = FluentLogger.forEnclosingClass();
    @VisibleForTesting
    static final String MEDIAPIPE_ASSET_CACHE_DIR = "mediapipe_asset_cache";
    private static AssetCache assetCache;
    private int appVersionCode;
    private AssetCacheDbHelper versionDatabase;
    private Context context;

    public static synchronized AssetCache create(Context context) {
        Preconditions.checkNotNull(context);
        if (assetCache == null) {
            assetCache = new AssetCache(context);
        }
        return assetCache;
    }

    public static synchronized void purgeCache(Context context) {
        AssetCacheDbHelper dbHelper = new AssetCacheDbHelper(context);
        dbHelper.invalidateCache(-1);
        dbHelper.close();
    }

    @Nullable
    public static synchronized AssetCache getAssetCache() {
        return assetCache;
    }

    public synchronized void loadAllAssets(String assetsPath) {
        String[] strArr;
        Preconditions.checkNotNull(assetsPath);
        AssetManager assetManager = this.context.getAssets();
        String[] assetFiles = null;
        try {
            assetFiles = assetManager.list(assetsPath);
        } catch (IOException e2) {
            logger.atSevere().withCause(e2).log("Unable to get files in assets path: %s", assetsPath);
        }
        if (assetFiles == null || assetFiles.length == 0) {
            logger.atWarning().log("No files to load");
            return;
        }
        for (String file : assetFiles) {
            String path = TextUtils.isEmpty(assetsPath) ? file : assetsPath + "/" + file;
            getAbsolutePathFromAsset(path);
        }
    }

    public synchronized String getAbsolutePathFromAsset(String assetPath) {
        AssetManager assetManager = this.context.getAssets();
        File destinationDir = getDefaultMediaPipeCacheDir();
        destinationDir.mkdir();
        File assetFile = new File(assetPath);
        String assetName = assetFile.getName();
        File destinationFile = new File(destinationDir.getPath(), assetName);
        if (destinationFile.exists() && this.appVersionCode != 0 && this.versionDatabase.checkVersion(assetPath, this.appVersionCode)) {
            return destinationFile.getAbsolutePath();
        }
        InputStream inStream = null;
        try {
            inStream = assetManager.open(assetPath);
            writeStreamToFile(inStream, destinationFile);
            if (this.appVersionCode != 0) {
                this.versionDatabase.insertAsset(assetPath, destinationFile.getAbsolutePath(), this.appVersionCode);
            }
            return destinationFile.getAbsolutePath();
        } catch (IOException e2) {
            logger.atSevere().log("Unable to unpack: %s", assetPath);
            if (inStream != null) {
                try {
                    inStream.close();
                } catch (IOException e3) {
                    return null;
                }
            }
            return null;
        }
    }

    public synchronized String[] getAvailableAssets() {
        File assetsDir = getDefaultMediaPipeCacheDir();
        if (assetsDir.exists()) {
            return assetsDir.list();
        }
        return new String[0];
    }

    public File getDefaultMediaPipeCacheDir() {
        return new File(this.context.getCacheDir(), "mediapipe_asset_cache");
    }

    private AssetCache(Context context) {
        this.context = context;
        this.versionDatabase = new AssetCacheDbHelper(context);
        try {
            this.appVersionCode = context.getPackageManager().getPackageInfo(context.getPackageName(), 0).versionCode;
            logger.atInfo().log("Current app version code: %d", this.appVersionCode);
            this.versionDatabase.invalidateCache(this.appVersionCode);
        } catch (PackageManager.NameNotFoundException e2) {
            throw new RuntimeException("Can't get app version code.", e2);
        }
    }

    private static void writeStreamToFile(InputStream inStream, File destinationFile) throws IOException {
        FileOutputStream outStream = null;
        try {
            outStream = new FileOutputStream(destinationFile);
            byte[] buffer = new byte[1000];
            while (true) {
                int n = inStream.read(buffer);
                if (n == -1) {
                    break;
                }
                outStream.write(buffer, 0, n);
            }
            if (outStream != null) {
                outStream.close();
            }
        } catch (Throwable th) {
            if (outStream != null) {
                outStream.close();
            }
            throw th;
        }
    }
}