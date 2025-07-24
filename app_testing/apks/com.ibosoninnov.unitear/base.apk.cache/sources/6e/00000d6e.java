package com.google.android.gms.common;

import android.app.Activity;
import android.app.Dialog;
import android.app.PendingIntent;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.Resources;
import androidx.annotation.RecentlyNonNull;
import androidx.annotation.RecentlyNullable;
import androidx.fragment.app.Fragment;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.android.gms.common.internal.HideFirstParty;
import com.google.android.gms.common.util.VisibleForTesting;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class GooglePlayServicesUtil extends GooglePlayServicesUtilLight {
    @RecentlyNonNull
    public static final String GMS_ERROR_DIALOG = "GooglePlayServicesErrorDialog";
    @RecentlyNonNull
    @Deprecated
    public static final String GOOGLE_PLAY_SERVICES_PACKAGE = "com.google.android.gms";
    @RecentlyNonNull
    @Deprecated
    public static final int GOOGLE_PLAY_SERVICES_VERSION_CODE = GooglePlayServicesUtilLight.GOOGLE_PLAY_SERVICES_VERSION_CODE;
    @RecentlyNonNull
    public static final String GOOGLE_PLAY_STORE_PACKAGE = "com.android.vending";

    private GooglePlayServicesUtil() {
    }

    @RecentlyNullable
    @Deprecated
    public static Dialog getErrorDialog(@RecentlyNonNull int i, @RecentlyNonNull Activity activity, @RecentlyNonNull int i2) {
        return getErrorDialog(i, activity, i2, null);
    }

    @RecentlyNonNull
    @Deprecated
    public static PendingIntent getErrorPendingIntent(@RecentlyNonNull int i, @RecentlyNonNull Context context, @RecentlyNonNull int i2) {
        return GooglePlayServicesUtilLight.getErrorPendingIntent(i, context, i2);
    }

    @RecentlyNonNull
    @VisibleForTesting
    @Deprecated
    public static String getErrorString(@RecentlyNonNull int i) {
        return GooglePlayServicesUtilLight.getErrorString(i);
    }

    @RecentlyNonNull
    public static Context getRemoteContext(@RecentlyNonNull Context context) {
        return GooglePlayServicesUtilLight.getRemoteContext(context);
    }

    @RecentlyNonNull
    public static Resources getRemoteResource(@RecentlyNonNull Context context) {
        return GooglePlayServicesUtilLight.getRemoteResource(context);
    }

    @RecentlyNonNull
    @HideFirstParty
    @Deprecated
    public static int isGooglePlayServicesAvailable(@RecentlyNonNull Context context) {
        return GooglePlayServicesUtilLight.isGooglePlayServicesAvailable(context);
    }

    @RecentlyNonNull
    @Deprecated
    public static boolean isUserRecoverableError(@RecentlyNonNull int i) {
        return GooglePlayServicesUtilLight.isUserRecoverableError(i);
    }

    @RecentlyNonNull
    @Deprecated
    public static boolean showErrorDialogFragment(@RecentlyNonNull int i, @RecentlyNonNull Activity activity, @RecentlyNonNull int i2, DialogInterface.OnCancelListener onCancelListener) {
        return showErrorDialogFragment(i, activity, null, i2, onCancelListener);
    }

    @Deprecated
    public static void showErrorNotification(@RecentlyNonNull int i, @RecentlyNonNull Context context) {
        GoogleApiAvailability googleApiAvailability = GoogleApiAvailability.getInstance();
        if (!GooglePlayServicesUtilLight.isPlayServicesPossiblyUpdating(context, i) && !GooglePlayServicesUtilLight.isPlayStorePossiblyUpdating(context, i)) {
            googleApiAvailability.showErrorNotification(context, i);
        } else {
            googleApiAvailability.zaa(context);
        }
    }

    @RecentlyNullable
    @Deprecated
    public static Dialog getErrorDialog(@RecentlyNonNull int i, @RecentlyNonNull Activity activity, @RecentlyNonNull int i2, DialogInterface.OnCancelListener onCancelListener) {
        if (GooglePlayServicesUtilLight.isPlayServicesPossiblyUpdating(activity, i)) {
            i = 18;
        }
        return GoogleApiAvailability.getInstance().getErrorDialog(activity, i, i2, onCancelListener);
    }

    @RecentlyNonNull
    @KeepForSdk
    @Deprecated
    public static int isGooglePlayServicesAvailable(@RecentlyNonNull Context context, @RecentlyNonNull int i) {
        return GooglePlayServicesUtilLight.isGooglePlayServicesAvailable(context, i);
    }

    @RecentlyNonNull
    public static boolean showErrorDialogFragment(@RecentlyNonNull int i, @RecentlyNonNull Activity activity, Fragment fragment, @RecentlyNonNull int i2, DialogInterface.OnCancelListener onCancelListener) {
        if (GooglePlayServicesUtilLight.isPlayServicesPossiblyUpdating(activity, i)) {
            i = 18;
        }
        GoogleApiAvailability googleApiAvailability = GoogleApiAvailability.getInstance();
        if (fragment == null) {
            return googleApiAvailability.showErrorDialogFragment(activity, i, i2, onCancelListener);
        }
        Dialog zaa = GoogleApiAvailability.zaa(activity, i, com.google.android.gms.common.internal.zab.zaa(fragment, GoogleApiAvailability.getInstance().getErrorResolutionIntent(activity, i, GoogleApiAvailabilityLight.TRACKING_SOURCE_DIALOG), i2), onCancelListener);
        if (zaa == null) {
            return false;
        }
        GoogleApiAvailability.zaa(activity, zaa, GMS_ERROR_DIALOG, onCancelListener);
        return true;
    }

    @RecentlyNonNull
    @Deprecated
    public static boolean showErrorDialogFragment(@RecentlyNonNull int i, @RecentlyNonNull Activity activity, @RecentlyNonNull int i2) {
        return showErrorDialogFragment(i, activity, i2, null);
    }
}