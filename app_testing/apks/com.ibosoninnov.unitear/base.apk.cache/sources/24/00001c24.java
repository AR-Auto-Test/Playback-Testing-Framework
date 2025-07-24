package com.google.ar.core;

import com.google.ar.core.ArCoreApk;

/* compiled from: ArCoreApkImpl.java */
/* loaded from: classes.dex */
public final class i implements h {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ j f5577a;

    public i(j jVar) {
        this.f5577a = jVar;
    }

    @Override // com.google.ar.core.h
    public final void a(ArCoreApk.Availability availability) {
        synchronized (this.f5577a) {
            this.f5577a.f5584g = availability;
            this.f5577a.f5585h = false;
        }
    }
}