package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApk.java */
/* loaded from: classes.dex */
public enum f extends ArCoreApk.Availability {
    public /* synthetic */ f() {
        super("SUPPORTED_APK_TOO_OLD", 5, 202);
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isSupported() {
        return true;
    }
}