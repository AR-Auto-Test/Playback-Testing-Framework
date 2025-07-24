package com.google.firebase.crashlytics.internal.model;

import c.b.a.a.a;
import com.google.firebase.crashlytics.internal.DevelopmentPlatformProvider;
import com.google.firebase.crashlytics.internal.model.StaticSessionData;
import java.util.Objects;

/* loaded from: classes2.dex */
public final class AutoValue_StaticSessionData_AppData extends StaticSessionData.AppData {
    private final String appIdentifier;
    private final int deliveryMechanism;
    private final DevelopmentPlatformProvider developmentPlatformProvider;
    private final String installUuid;
    private final String versionCode;
    private final String versionName;

    public AutoValue_StaticSessionData_AppData(String str, String str2, String str3, String str4, int i, DevelopmentPlatformProvider developmentPlatformProvider) {
        Objects.requireNonNull(str, "Null appIdentifier");
        this.appIdentifier = str;
        Objects.requireNonNull(str2, "Null versionCode");
        this.versionCode = str2;
        Objects.requireNonNull(str3, "Null versionName");
        this.versionName = str3;
        Objects.requireNonNull(str4, "Null installUuid");
        this.installUuid = str4;
        this.deliveryMechanism = i;
        Objects.requireNonNull(developmentPlatformProvider, "Null developmentPlatformProvider");
        this.developmentPlatformProvider = developmentPlatformProvider;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public String appIdentifier() {
        return this.appIdentifier;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public int deliveryMechanism() {
        return this.deliveryMechanism;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public DevelopmentPlatformProvider developmentPlatformProvider() {
        return this.developmentPlatformProvider;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof StaticSessionData.AppData) {
            StaticSessionData.AppData appData = (StaticSessionData.AppData) obj;
            return this.appIdentifier.equals(appData.appIdentifier()) && this.versionCode.equals(appData.versionCode()) && this.versionName.equals(appData.versionName()) && this.installUuid.equals(appData.installUuid()) && this.deliveryMechanism == appData.deliveryMechanism() && this.developmentPlatformProvider.equals(appData.developmentPlatformProvider());
        }
        return false;
    }

    public int hashCode() {
        return ((((((((((this.appIdentifier.hashCode() ^ 1000003) * 1000003) ^ this.versionCode.hashCode()) * 1000003) ^ this.versionName.hashCode()) * 1000003) ^ this.installUuid.hashCode()) * 1000003) ^ this.deliveryMechanism) * 1000003) ^ this.developmentPlatformProvider.hashCode();
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public String installUuid() {
        return this.installUuid;
    }

    public String toString() {
        StringBuilder x = a.x("AppData{appIdentifier=");
        x.append(this.appIdentifier);
        x.append(", versionCode=");
        x.append(this.versionCode);
        x.append(", versionName=");
        x.append(this.versionName);
        x.append(", installUuid=");
        x.append(this.installUuid);
        x.append(", deliveryMechanism=");
        x.append(this.deliveryMechanism);
        x.append(", developmentPlatformProvider=");
        x.append(this.developmentPlatformProvider);
        x.append("}");
        return x.toString();
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public String versionCode() {
        return this.versionCode;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData.AppData
    public String versionName() {
        return this.versionName;
    }
}