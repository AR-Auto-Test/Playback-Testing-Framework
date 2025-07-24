package com.google.ar.core;

import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class v implements ServiceConnection {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ u f5615a;

    public v(u uVar) {
        this.f5615a = uVar;
    }

    @Override // android.content.ServiceConnection
    public final void onServiceConnected(ComponentName componentName, IBinder iBinder) {
        this.f5615a.l(iBinder);
    }

    @Override // android.content.ServiceConnection
    public final void onServiceDisconnected(ComponentName componentName) {
        this.f5615a.m();
    }
}