package com.google.ar.core;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.ActivityNotFoundException;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.IntentSender;
import android.content.ServiceConnection;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInstaller;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.exceptions.FatalException;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.Queue;

/* compiled from: InstallService.java */
/* loaded from: classes.dex */
public final class u {

    /* renamed from: a  reason: collision with root package name */
    private final Queue f5607a;

    /* renamed from: b  reason: collision with root package name */
    private Context f5608b;

    /* renamed from: c  reason: collision with root package name */
    private com.google.ar.core.dependencies.i f5609c;

    /* renamed from: d  reason: collision with root package name */
    private final ServiceConnection f5610d;

    /* renamed from: e  reason: collision with root package name */
    private BroadcastReceiver f5611e;

    /* renamed from: f  reason: collision with root package name */
    private Context f5612f;

    /* renamed from: g  reason: collision with root package name */
    private PackageInstaller f5613g;

    /* renamed from: h  reason: collision with root package name */
    private PackageInstaller.SessionCallback f5614h;
    private volatile int i;

    public u() {
    }

    public u(byte[] bArr) {
        this();
        this.f5607a = new ArrayDeque();
        this.i = 1;
        this.f5610d = new v(this);
    }

    public static /* bridge */ /* synthetic */ Bundle k() {
        Bundle bundle = new Bundle();
        bundle.putCharSequence("package.name", "com.google.ar.core");
        return bundle;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final synchronized void l(IBinder iBinder) {
        com.google.ar.core.dependencies.i b2 = com.google.ar.core.dependencies.h.b(iBinder);
        Log.i("ARCore-InstallService", "Install service connected");
        this.f5609c = b2;
        this.i = 3;
        for (Runnable runnable : this.f5607a) {
            runnable.run();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public final synchronized void m() {
        Log.i("ARCore-InstallService", "Install service disconnected");
        this.i = 1;
        this.f5609c = null;
    }

    private final synchronized void n(Runnable runnable) {
        int i = this.i;
        int i2 = i - 1;
        if (i == 0) {
            throw null;
        }
        if (i2 == 0) {
            throw new ad();
        }
        if (i2 == 1) {
            this.f5607a.offer(runnable);
        } else if (i2 != 2) {
        } else {
            runnable.run();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static void o(Activity activity, s sVar) {
        boolean z;
        try {
            Intent intent = new Intent("android.intent.action.VIEW", Uri.parse("market://details?id=com.google.ar.core"));
            j a2 = j.a();
            Iterator<ResolveInfo> it = activity.getPackageManager().queryIntentActivities(intent, 65536).iterator();
            while (true) {
                if (!it.hasNext()) {
                    z = false;
                    break;
                }
                ActivityInfo activityInfo = it.next().activityInfo;
                if (activityInfo != null && activityInfo.name.equals("com.sec.android.app.samsungapps.MainForChina")) {
                    z = true;
                    break;
                }
            }
            a2.f5580b = !z;
            activity.startActivity(intent);
        } catch (ActivityNotFoundException e2) {
            sVar.b(new FatalException("Failed to launch installer.", e2));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static void p(Activity activity, Bundle bundle, s sVar) {
        PendingIntent pendingIntent = (PendingIntent) bundle.getParcelable("resolution.intent");
        if (pendingIntent != null) {
            try {
                activity.startIntentSenderForResult(pendingIntent.getIntentSender(), 1234, new Intent(activity, activity.getClass()), 0, 0, 0);
                return;
            } catch (IntentSender.SendIntentException e2) {
                sVar.b(new FatalException("Installation Intent failed", e2));
                return;
            }
        }
        Log.e("ARCore-InstallService", "Did not get pending intent.");
        sVar.b(new FatalException("Installation intent failed to unparcel."));
    }

    public final synchronized void a(Context context) {
        this.f5608b = context;
        if (context.bindService(new Intent("com.google.android.play.core.install.BIND_INSTALL_SERVICE").setPackage("com.android.vending"), this.f5610d, 1)) {
            this.i = 2;
            return;
        }
        this.i = 1;
        this.f5608b = null;
        Log.w("ARCore-InstallService", "bindService returned false.");
        context.unbindService(this.f5610d);
    }

    public final synchronized void b(Context context, h hVar) {
        try {
            n(new x(this, context, hVar));
        } catch (ad unused) {
            Log.e("ARCore-InstallService", "Play Store install service could not be bound.");
            hVar.a(ArCoreApk.Availability.UNKNOWN_ERROR);
        }
    }

    public final synchronized void c() {
        int i = this.i;
        int i2 = i - 1;
        if (i != 0) {
            if (i2 == 1 || i2 == 2) {
                this.f5608b.unbindService(this.f5610d);
                this.f5608b = null;
                this.i = 1;
            }
            BroadcastReceiver broadcastReceiver = this.f5611e;
            if (broadcastReceiver != null) {
                this.f5612f.unregisterReceiver(broadcastReceiver);
            }
            PackageInstaller.SessionCallback sessionCallback = this.f5614h;
            if (sessionCallback != null) {
                this.f5613g.unregisterSessionCallback(sessionCallback);
                this.f5614h = null;
                return;
            }
            return;
        }
        throw null;
    }

    public final void d(Activity activity, s sVar) {
        if (this.f5614h == null) {
            this.f5613g = activity.getPackageManager().getPackageInstaller();
            y yVar = new y(this, sVar);
            this.f5614h = yVar;
            this.f5613g.registerSessionCallback(yVar);
        }
        if (this.f5611e == null) {
            z zVar = new z(sVar);
            this.f5611e = zVar;
            this.f5612f = activity;
            activity.registerReceiver(zVar, new IntentFilter("com.google.android.play.core.install.ACTION_INSTALL_STATUS"));
        }
        try {
            n(new ac(this, activity, sVar));
        } catch (ad unused) {
            Log.w("ARCore-InstallService", "requestInstall bind failed, launching fullscreen.");
            o(activity, sVar);
        }
    }
}