package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApk.java */
/* loaded from: classes.dex */
public enum d extends ArCoreApk.Availability {
    public /* synthetic */ d() {
        super("UNSUPPORTED_DEVICE_NOT_CAPABLE", 3, 100);
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isUnsupported() {
        return true;
    }
}