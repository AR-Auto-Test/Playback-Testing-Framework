package com.google.ar.core;

import android.content.pm.PackageInstaller;
import android.util.Log;
import java.util.HashMap;
import java.util.Map;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class y extends PackageInstaller.SessionCallback {

    /* renamed from: a  reason: collision with root package name */
    public final Map<Integer, PackageInstaller.SessionInfo> f5620a = new HashMap();

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ s f5621b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ u f5622c;

    public y(u uVar, s sVar) {
        this.f5622c = uVar;
        this.f5621b = sVar;
    }

    @Override // android.content.pm.PackageInstaller.SessionCallback
    public final void onActiveChanged(int i, boolean z) {
    }

    @Override // android.content.pm.PackageInstaller.SessionCallback
    public final void onBadgingChanged(int i) {
    }

    @Override // android.content.pm.PackageInstaller.SessionCallback
    public final void onCreated(int i) {
        PackageInstaller packageInstaller;
        packageInstaller = this.f5622c.f5613g;
        this.f5620a.put(Integer.valueOf(i), packageInstaller.getSessionInfo(i));
    }

    @Override // android.content.pm.PackageInstaller.SessionCallback
    public final void onFinished(int i, boolean z) {
        PackageInstaller.SessionInfo remove = this.f5620a.remove(Integer.valueOf(i));
        if (remove == null || !"com.google.ar.core".equals(remove.getAppPackageName())) {
            return;
        }
        Log.i("ARCore-InstallService", "Detected ARCore install completion");
        this.f5621b.a(t.COMPLETED);
    }

    @Override // android.content.pm.PackageInstaller.SessionCallback
    public final void onProgressChanged(int i, float f2) {
    }
}