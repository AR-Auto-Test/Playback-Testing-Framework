package com.ibosoninnov.unitear;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import b.j.b.h;

/* loaded from: classes2.dex */
public class MediaProjectionService extends Service {

    /* renamed from: b  reason: collision with root package name */
    public NotificationManager f5685b;

    /* renamed from: c  reason: collision with root package name */
    public final IBinder f5686c = new a(this);

    /* loaded from: classes2.dex */
    public class a extends Binder {
        public a(MediaProjectionService mediaProjectionService) {
        }
    }

    @Override // android.app.Service
    public IBinder onBind(Intent intent) {
        return this.f5686c;
    }

    @Override // android.app.Service
    public void onCreate() {
        Log.i("LocalService", "OnCreate");
        this.f5685b = (NotificationManager) getSystemService("notification");
        h hVar = new h(this);
        hVar.c(16, false);
        hVar.o.icon = 2131165488;
        hVar.f2064e = h.b(getResources().getString(R.string.app_name));
        hVar.f2065f = h.b("screen recording");
        hVar.c(2, true);
        if (Build.VERSION.SDK_INT >= 26) {
            this.f5685b.createNotificationChannel(new NotificationChannel("unitear", getResources().getString(R.string.app_name), 2));
            hVar.m = "unitear";
        }
        startForeground(1, hVar.a());
    }

    @Override // android.app.Service
    public void onDestroy() {
        this.f5685b.cancel(1);
        Log.i("LocalService", "Stopped");
    }

    @Override // android.app.Service
    public int onStartCommand(Intent intent, int i, int i2) {
        Log.i("LocalService", "Received start id " + i2 + ": " + intent);
        return 1;
    }
}