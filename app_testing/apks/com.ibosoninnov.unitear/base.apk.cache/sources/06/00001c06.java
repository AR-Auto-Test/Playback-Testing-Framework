package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApk.java */
/* loaded from: classes.dex */
public enum e extends ArCoreApk.Availability {
    public /* synthetic */ e() {
        super("SUPPORTED_NOT_INSTALLED", 4, 201);
    }

    @Override // com.google.ar.core.ArCoreApk.Availability
    public final boolean isSupported() {
        return true;
    }
}