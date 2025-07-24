package com.google.ar.core;

import android.content.Context;
import android.os.RemoteException;
import android.util.Log;
import com.google.ar.core.ArCoreApk;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class x implements Runnable {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Context f5617a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ h f5618b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ u f5619c;

    public x(u uVar, Context context, h hVar) {
        this.f5619c = uVar;
        this.f5617a = context;
        this.f5618b = hVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        com.google.ar.core.dependencies.i iVar;
        try {
            iVar = this.f5619c.f5609c;
            iVar.d(this.f5617a.getApplicationInfo().packageName, u.k(), new w(this));
        } catch (RemoteException e2) {
            Log.e("ARCore-InstallService", "requestInfo threw", e2);
            this.f5618b.a(ArCoreApk.Availability.UNKNOWN_ERROR);
        }
    }
}