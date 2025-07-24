package com.google.android.play.core.assetpacks;

import com.google.android.play.core.assetpacks.model.AssetPackStorageMethod;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public abstract class AssetPackLocation {
    private static final AssetPackLocation zza = new zzbm(1, null, null);

    public static AssetPackLocation zza() {
        return zza;
    }

    public abstract String assetsPath();

    @AssetPackStorageMethod
    public abstract int packStorageMethod();

    public abstract String path();
}