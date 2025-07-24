package com.google.ar.core;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

/* compiled from: InstallServiceImpl.java */
/* loaded from: classes.dex */
public final class z extends BroadcastReceiver {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ s f5623a;

    public z(s sVar) {
        this.f5623a = sVar;
    }

    @Override // android.content.BroadcastReceiver
    public final void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        Bundle extras = intent.getExtras();
        if ("com.google.android.play.core.install.ACTION_INSTALL_STATUS".equals(action) && extras != null && extras.containsKey("install.status")) {
            int i = extras.getInt("install.status");
            if (i == 1 || i == 2 || i == 3) {
                this.f5623a.a(t.ACCEPTED);
            } else if (i == 4) {
                this.f5623a.a(t.COMPLETED);
            } else if (i != 6) {
            } else {
                this.f5623a.a(t.CANCELLED);
            }
        }
    }
}