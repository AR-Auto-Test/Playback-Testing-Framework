package com.google.firebase.crashlytics.internal.model;

import c.b.a.a.a;
import com.google.firebase.crashlytics.internal.model.StaticSessionData;
import java.util.Objects;

/* loaded from: classes2.dex */
public final class AutoValue_StaticSessionData extends StaticSessionData {
    private final StaticSessionData.AppData appData;
    private final StaticSessionData.DeviceData deviceData;
    private final StaticSessionData.OsData osData;

    public AutoValue_StaticSessionData(StaticSessionData.AppData appData, StaticSessionData.OsData osData, StaticSessionData.DeviceData deviceData) {
        Objects.requireNonNull(appData, "Null appData");
        this.appData = appData;
        Objects.requireNonNull(osData, "Null osData");
        this.osData = osData;
        Objects.requireNonNull(deviceData, "Null deviceData");
        this.deviceData = deviceData;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData
    public StaticSessionData.AppData appData() {
        return this.appData;
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData
    public StaticSessionData.DeviceData deviceData() {
        return this.deviceData;
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof StaticSessionData) {
            StaticSessionData staticSessionData = (StaticSessionData) obj;
            return this.appData.equals(staticSessionData.appData()) && this.osData.equals(staticSessionData.osData()) && this.deviceData.equals(staticSessionData.deviceData());
        }
        return false;
    }

    public int hashCode() {
        return ((((this.appData.hashCode() ^ 1000003) * 1000003) ^ this.osData.hashCode()) * 1000003) ^ this.deviceData.hashCode();
    }

    @Override // com.google.firebase.crashlytics.internal.model.StaticSessionData
    public StaticSessionData.OsData osData() {
        return this.osData;
    }

    public String toString() {
        StringBuilder x = a.x("StaticSessionData{appData=");
        x.append(this.appData);
        x.append(", osData=");
        x.append(this.osData);
        x.append(", deviceData=");
        x.append(this.deviceData);
        x.append("}");
        return x.toString();
    }
}