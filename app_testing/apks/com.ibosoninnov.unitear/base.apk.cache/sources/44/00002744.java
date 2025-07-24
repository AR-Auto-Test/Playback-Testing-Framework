package com.google.firebase.crashlytics.internal.settings;

import com.google.firebase.crashlytics.internal.Logger;
import com.google.firebase.crashlytics.internal.common.CurrentTimeProvider;
import org.json.JSONObject;

/* loaded from: classes2.dex */
public class SettingsJsonParser {
    private final CurrentTimeProvider currentTimeProvider;

    public SettingsJsonParser(CurrentTimeProvider currentTimeProvider) {
        this.currentTimeProvider = currentTimeProvider;
    }

    private static SettingsJsonTransform getJsonTransformForVersion(int i) {
        if (i != 3) {
            Logger logger = Logger.getLogger();
            logger.e("Could not determine SettingsJsonTransform for settings version " + i + ". Using default settings values.");
            return new DefaultSettingsJsonTransform();
        }
        return new SettingsV3JsonTransform();
    }

    public Settings parseSettingsJson(JSONObject jSONObject) {
        return getJsonTransformForVersion(jSONObject.getInt(SettingsJsonConstants.SETTINGS_VERSION)).buildFromJson(this.currentTimeProvider, jSONObject);
    }
}