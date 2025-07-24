package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApk.java */
/* loaded from: classes.dex */
public enum b extends ArCoreApk.Availability {
    public /* synthetic */ b() {
        super("UNKNOWN_CHECKING", 1, 1);
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isTransient() {
        return true;
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isUnknown() {
        return true;
    }
}