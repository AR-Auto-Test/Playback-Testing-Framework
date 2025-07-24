package com.google.ar.core;

import com.google.ar.core.ArCoreApk;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: InstallActivity.java */
/* loaded from: classes.dex */
public final class o implements h {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ AtomicReference f5593a;

    public o(AtomicReference atomicReference) {
        this.f5593a = atomicReference;
    }

    @Override // com.google.ar.core.h
    public final void a(ArCoreApk.Availability availability) {
        this.f5593a.set(availability);
    }
}