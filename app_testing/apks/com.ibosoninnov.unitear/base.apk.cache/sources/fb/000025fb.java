package com.google.firebase.analytics.connector;

import android.os.Bundle;
import com.google.android.gms.common.annotation.KeepForSdk;
import com.google.firebase.annotations.DeferredApi;
import java.util.List;
import java.util.Map;
import java.util.Set;

/* compiled from: com.google.firebase:firebase-measurement-connector@@19.0.0 */
/* loaded from: classes.dex */
public interface AnalyticsConnector {

    /* compiled from: com.google.firebase:firebase-measurement-connector@@19.0.0 */
    @KeepForSdk
    /* loaded from: classes.dex */
    public interface AnalyticsConnectorHandle {
        @KeepForSdk
        void registerEventNames(Set<String> set);

        @KeepForSdk
        void unregister();

        @KeepForSdk
        void unregisterEventNames();
    }

    /* compiled from: com.google.firebase:firebase-measurement-connector@@19.0.0 */
    @KeepForSdk
    /* loaded from: classes.dex */
    public interface AnalyticsConnectorListener {
        @KeepForSdk
        void onMessageTriggered(int i, Bundle bundle);
    }

    /* compiled from: com.google.firebase:firebase-measurement-connector@@19.0.0 */
    @KeepForSdk
    /* loaded from: classes.dex */
    public static class ConditionalUserProperty {
        @KeepForSdk
        public boolean active;
        @KeepForSdk
        public long creationTimestamp;
        @KeepForSdk
        public String expiredEventName;
        @KeepForSdk
        public Bundle expiredEventParams;
        @KeepForSdk
        public String name;
        @KeepForSdk
        public String origin;
        @KeepForSdk
        public long timeToLive;
        @KeepForSdk
        public String timedOutEventName;
        @KeepForSdk
        public Bundle timedOutEventParams;
        @KeepForSdk
        public String triggerEventName;
        @KeepForSdk
        public long triggerTimeout;
        @KeepForSdk
        public String triggeredEventName;
        @KeepForSdk
        public Bundle triggeredEventParams;
        @KeepForSdk
        public long triggeredTimestamp;
        @KeepForSdk
        public Object value;
    }

    @KeepForSdk
    void clearConditionalUserProperty(String str, String str2, Bundle bundle);

    @KeepForSdk
    List<ConditionalUserProperty> getConditionalUserProperties(String str, String str2);

    @KeepForSdk
    int getMaxUserProperties(String str);

    @KeepForSdk
    Map<String, Object> getUserProperties(boolean z);

    @KeepForSdk
    void logEvent(String str, String str2, Bundle bundle);

    @KeepForSdk
    @DeferredApi
    AnalyticsConnectorHandle registerAnalyticsConnectorListener(String str, AnalyticsConnectorListener analyticsConnectorListener);

    @KeepForSdk
    void setConditionalUserProperty(ConditionalUserProperty conditionalUserProperty);

    @KeepForSdk
    void setUserProperty(String str, String str2, Object obj);
}