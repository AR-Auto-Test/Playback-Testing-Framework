package com.google.firebase.crashlytics.internal.settings;

import com.google.firebase.crashlytics.internal.common.CurrentTimeProvider;
import com.google.firebase.crashlytics.internal.settings.Settings;
import org.json.JSONObject;

/* loaded from: classes2.dex */
public class SettingsV3JsonTransform implements SettingsJsonTransform {
    private static Settings.FeatureFlagData buildFeatureFlagDataFrom(JSONObject jSONObject) {
        return new Settings.FeatureFlagData(jSONObject.optBoolean(SettingsJsonConstants.FEATURES_COLLECT_REPORTS_KEY, true), jSONObject.optBoolean(SettingsJsonConstants.FEATURES_COLLECT_ANRS_KEY, false));
    }

    private static Settings.SessionData buildSessionDataFrom(JSONObject jSONObject) {
        return new Settings.SessionData(jSONObject.optInt(SettingsJsonConstants.SETTINGS_MAX_CUSTOM_EXCEPTION_EVENTS_KEY, 8), 4);
    }

    private static long getExpiresAtFrom(CurrentTimeProvider currentTimeProvider, long j, JSONObject jSONObject) {
        if (jSONObject.has(SettingsJsonConstants.EXPIRES_AT_KEY)) {
            return jSONObject.optLong(SettingsJsonConstants.EXPIRES_AT_KEY);
        }
        return (j * 1000) + currentTimeProvider.getCurrentTimeMillis();
    }

    @Override // com.google.firebase.crashlytics.internal.settings.SettingsJsonTransform
    public Settings buildFromJson(CurrentTimeProvider currentTimeProvider, JSONObject jSONObject) {
        Settings.SessionData buildSessionDataFrom;
        int optInt = jSONObject.optInt(SettingsJsonConstants.SETTINGS_VERSION, 0);
        int optInt2 = jSONObject.optInt(SettingsJsonConstants.CACHE_DURATION_KEY, SettingsJsonConstants.SETTINGS_CACHE_DURATION_DEFAULT);
        double optDouble = jSONObject.optDouble(SettingsJsonConstants.ON_DEMAND_UPLOAD_RATE_PER_MINUTE_KEY, 10.0d);
        double optDouble2 = jSONObject.optDouble(SettingsJsonConstants.ON_DEMAND_BACKOFF_BASE_KEY, 1.2d);
        int optInt3 = jSONObject.optInt(SettingsJsonConstants.ON_DEMAND_BACKOFF_STEP_DURATION_SECONDS_KEY, 60);
        if (jSONObject.has(SettingsJsonConstants.SESSION_KEY)) {
            buildSessionDataFrom = buildSessionDataFrom(jSONObject.getJSONObject(SettingsJsonConstants.SESSION_KEY));
        } else {
            buildSessionDataFrom = buildSessionDataFrom(new JSONObject());
        }
        return new Settings(getExpiresAtFrom(currentTimeProvider, optInt2, jSONObject), buildSessionDataFrom, buildFeatureFlagDataFrom(jSONObject.getJSONObject(SettingsJsonConstants.FEATURES_KEY)), optInt, optInt2, optDouble, optDouble2, optInt3);
    }
}