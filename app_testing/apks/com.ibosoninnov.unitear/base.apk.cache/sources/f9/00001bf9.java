package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApk.java */
/* loaded from: classes.dex */
public enum c extends ArCoreApk.Availability {
    public /* synthetic */ c() {
        super("UNKNOWN_TIMED_OUT", 2, 2);
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isUnknown() {
        return true;
    }
}