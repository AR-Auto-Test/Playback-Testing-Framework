package com.google.android.play.core.appupdate;

import com.google.android.play.core.install.model.AppUpdateType;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public abstract class AppUpdateOptions {

    /* compiled from: com.google.android.play:core@@1.10.3 */
    /* loaded from: classes.dex */
    public static abstract class Builder {
        public abstract AppUpdateOptions build();

        public abstract Builder setAllowAssetPackDeletion(boolean z);

        public abstract Builder setAppUpdateType(@AppUpdateType int i);
    }

    public static AppUpdateOptions defaultOptions(@AppUpdateType int i) {
        return newBuilder(i).build();
    }

    public static Builder newBuilder(@AppUpdateType int i) {
        zzu zzuVar = new zzu();
        zzuVar.setAppUpdateType(i);
        zzuVar.setAllowAssetPackDeletion(false);
        return zzuVar;
    }

    public abstract boolean allowAssetPackDeletion();

    @AppUpdateType
    public abstract int appUpdateType();
}