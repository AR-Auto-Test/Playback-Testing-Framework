package com.google.firebase.crashlytics.internal.common;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import com.google.firebase.analytics.FirebaseAnalytics;
import com.google.firebase.crashlytics.internal.Logger;
import com.google.firebase.crashlytics.internal.settings.SettingsJsonConstants;

/* loaded from: classes2.dex */
public class BatteryState {
    public static final int VELOCITY_CHARGING = 2;
    public static final int VELOCITY_FULL = 3;
    public static final int VELOCITY_UNPLUGGED = 1;
    private final Float level;
    private final boolean powerConnected;

    private BatteryState(Float f2, boolean z) {
        this.powerConnected = z;
        this.level = f2;
    }

    public static BatteryState get(Context context) {
        Float f2 = null;
        boolean z = false;
        try {
            Intent registerReceiver = context.registerReceiver(null, new IntentFilter("android.intent.action.BATTERY_CHANGED"));
            if (registerReceiver != null) {
                z = isPowerConnected(registerReceiver);
                f2 = getLevel(registerReceiver);
            }
        } catch (IllegalStateException e2) {
            Logger.getLogger().e("An error occurred getting battery state.", e2);
        }
        return new BatteryState(f2, z);
    }

    private static Float getLevel(Intent intent) {
        int intExtra = intent.getIntExtra(FirebaseAnalytics.Param.LEVEL, -1);
        int intExtra2 = intent.getIntExtra("scale", -1);
        if (intExtra == -1 || intExtra2 == -1) {
            return null;
        }
        return Float.valueOf(intExtra / intExtra2);
    }

    public Float getBatteryLevel() {
        return this.level;
    }

    public int getBatteryVelocity() {
        Float f2;
        if (!this.powerConnected || (f2 = this.level) == null) {
            return 1;
        }
        return ((double) f2.floatValue()) < 0.99d ? 2 : 3;
    }

    public boolean isPowerConnected() {
        return this.powerConnected;
    }

    private static boolean isPowerConnected(Intent intent) {
        int intExtra = intent.getIntExtra(SettingsJsonConstants.APP_STATUS_KEY, -1);
        if (intExtra == -1) {
            return false;
        }
        return intExtra == 2 || intExtra == 5;
    }
}