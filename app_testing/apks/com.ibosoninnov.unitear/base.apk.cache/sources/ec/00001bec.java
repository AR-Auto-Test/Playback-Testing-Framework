package com.google.ar.core;

import android.util.Log;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class ab implements Runnable {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ AtomicBoolean f5542a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ac f5543b;

    public ab(ac acVar, AtomicBoolean atomicBoolean) {
        this.f5543b = acVar;
        this.f5542a = atomicBoolean;
    }

    @Override // java.lang.Runnable
    public final void run() {
        if (this.f5542a.getAndSet(true)) {
            return;
        }
        Log.w("ARCore-InstallService", "requestInstall timed out, launching fullscreen.");
        ac acVar = this.f5543b;
        u uVar = acVar.f5546c;
        u.o(acVar.f5544a, acVar.f5545b);
    }
}