package com.google.ar.core;

import android.os.Bundle;
import android.util.Log;
import com.google.ar.core.ArCoreApk;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class w extends com.google.ar.core.dependencies.j {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ x f5616a;

    public w(x xVar) {
        this.f5616a = xVar;
    }

    @Override // com.google.ar.core.dependencies.k
    public final void b(Bundle bundle) {
        int i = bundle.getInt("error.code", -100);
        if (i == -5) {
            Log.e("ARCore-InstallService", "The device is not supported.");
            this.f5616a.f5618b.a(ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE);
        } else if (i == -3) {
            Log.e("ARCore-InstallService", "The Google Play application must be updated.");
            this.f5616a.f5618b.a(ArCoreApk.Availability.UNKNOWN_ERROR);
        } else if (i != 0) {
            StringBuilder sb = new StringBuilder(33);
            sb.append("requestInfo returned: ");
            sb.append(i);
            Log.e("ARCore-InstallService", sb.toString());
            this.f5616a.f5618b.a(ArCoreApk.Availability.UNKNOWN_ERROR);
        } else {
            this.f5616a.f5618b.a(ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED);
        }
    }

    @Override // com.google.ar.core.dependencies.k
    public final void c(Bundle bundle) {
    }
}