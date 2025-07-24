package com.google.ar.core;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.RemoteException;
import android.util.Log;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicBoolean;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class ac implements Runnable {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Activity f5544a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ s f5545b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ u f5546c;

    public ac(u uVar, Activity activity, s sVar) {
        this.f5546c = uVar;
        this.f5544a = activity;
        this.f5545b = sVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        com.google.ar.core.dependencies.i iVar;
        try {
            AtomicBoolean atomicBoolean = new AtomicBoolean(false);
            iVar = this.f5546c.f5609c;
            iVar.e(this.f5544a.getApplicationInfo().packageName, Collections.singletonList(u.k()), new Bundle(), new aa(this, atomicBoolean));
            new Handler().postDelayed(new ab(this, atomicBoolean), 3000L);
        } catch (RemoteException e2) {
            Log.w("ARCore-InstallService", "requestInstall threw, launching fullscreen.", e2);
            u uVar = this.f5546c;
            u.o(this.f5544a, this.f5545b);
        }
    }
}