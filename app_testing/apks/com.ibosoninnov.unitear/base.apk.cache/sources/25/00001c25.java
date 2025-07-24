package com.google.ar.core;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.IntentSender;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.exceptions.FatalException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

/* compiled from: ArCoreApkImpl.java */
/* loaded from: classes.dex */
public final class j extends ArCoreApk {

    /* renamed from: c  reason: collision with root package name */
    private static final j f5578c = new j();

    /* renamed from: a  reason: collision with root package name */
    public Exception f5579a;

    /* renamed from: b  reason: collision with root package name */
    public boolean f5580b = true;

    /* renamed from: d  reason: collision with root package name */
    private boolean f5581d;

    /* renamed from: e  reason: collision with root package name */
    private int f5582e;

    /* renamed from: f  reason: collision with root package name */
    private long f5583f;

    /* renamed from: g  reason: collision with root package name */
    private ArCoreApk.Availability f5584g;

    /* renamed from: h  reason: collision with root package name */
    private boolean f5585h;
    private u i;
    private boolean j;
    private boolean k;
    private int l;

    public static j a() {
        return f5578c;
    }

    private static int g(Context context) {
        try {
            PackageInfo packageInfo = context.getPackageManager().getPackageInfo("com.google.ar.core", 4);
            int i = packageInfo.versionCode;
            if (i == 0) {
                ServiceInfo[] serviceInfoArr = packageInfo.services;
                if (serviceInfoArr != null) {
                    if (serviceInfoArr.length != 0) {
                        return 0;
                    }
                }
                return -1;
            }
            return i;
        } catch (PackageManager.NameNotFoundException unused) {
            return -1;
        }
    }

    private final synchronized void h(Context context) {
        if (this.j) {
            return;
        }
        PackageManager packageManager = context.getPackageManager();
        String packageName = context.getPackageName();
        try {
            Bundle bundle = packageManager.getApplicationInfo(packageName, 128).metaData;
            if (bundle.containsKey("com.google.ar.core")) {
                String string = bundle.getString("com.google.ar.core");
                string.getClass();
                this.k = string.equals("required");
                if (bundle.containsKey("com.google.ar.core.min_apk_version")) {
                    this.l = bundle.getInt("com.google.ar.core.min_apk_version");
                    try {
                        ActivityInfo[] activityInfoArr = packageManager.getPackageInfo(packageName, 1).activities;
                        String canonicalName = InstallActivity.class.getCanonicalName();
                        for (ActivityInfo activityInfo : activityInfoArr) {
                            if (canonicalName.equals(activityInfo.name)) {
                                this.j = true;
                                return;
                            }
                        }
                        String valueOf = String.valueOf(canonicalName);
                        throw new FatalException(valueOf.length() != 0 ? "Application manifest must contain activity ".concat(valueOf) : new String("Application manifest must contain activity "));
                    } catch (PackageManager.NameNotFoundException e2) {
                        throw new FatalException("Could not load application package info", e2);
                    }
                }
                throw new FatalException("Application manifest must contain meta-data com.google.ar.core.min_apk_version");
            }
            throw new FatalException("Application manifest must contain meta-data com.google.ar.core");
        } catch (PackageManager.NameNotFoundException e3) {
            throw new FatalException("Could not load application package metadata", e3);
        }
    }

    private static boolean i() {
        return true;
    }

    private final boolean j(Context context) {
        h(context);
        return this.k;
    }

    private static final ArCoreApk.InstallStatus k(Activity activity) {
        PendingIntent a2 = ag.a(activity);
        if (a2 != null) {
            try {
                Log.i("ARCore-ArCoreApk", "Starting setup activity");
                activity.startIntentSender(a2.getIntentSender(), null, 0, 0, 0);
                return ArCoreApk.InstallStatus.INSTALL_REQUESTED;
            } catch (IntentSender.SendIntentException | RuntimeException e2) {
                Log.w("ARCore-ArCoreApk", "Setup activity launch failed", e2);
            }
        }
        return ArCoreApk.InstallStatus.INSTALLED;
    }

    public final synchronized u b(Context context) {
        if (this.i == null) {
            u uVar = new u(null);
            uVar.a(context.getApplicationContext());
            this.i = uVar;
        }
        return this.i;
    }

    @Override // com.google.ar.core.ArCoreApk
    public final ArCoreApk.Availability checkAvailability(Context context) {
        ArCoreApk.Availability availability;
        if (i()) {
            try {
                if (e(context)) {
                    d();
                    try {
                        if (ag.a(context) != null) {
                            availability = ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD;
                        } else {
                            availability = ArCoreApk.Availability.SUPPORTED_INSTALLED;
                        }
                        return availability;
                    } catch (UnavailableDeviceNotCompatibleException unused) {
                        return ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE;
                    } catch (UnavailableUserDeclinedInstallationException | RuntimeException unused2) {
                        return ArCoreApk.Availability.UNKNOWN_ERROR;
                    }
                }
                synchronized (this) {
                    ArCoreApk.Availability availability2 = this.f5584g;
                    if ((availability2 == null || availability2.isUnknown()) && !this.f5585h) {
                        this.f5585h = true;
                        i iVar = new i(this);
                        if (e(context)) {
                            iVar.a(ArCoreApk.Availability.SUPPORTED_INSTALLED);
                        } else if (g(context) != -1) {
                            iVar.a(ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD);
                        } else if (j(context)) {
                            iVar.a(ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED);
                        } else {
                            b(context).b(context, iVar);
                        }
                    }
                    ArCoreApk.Availability availability3 = this.f5584g;
                    if (availability3 != null) {
                        return availability3;
                    }
                    if (this.f5585h) {
                        return ArCoreApk.Availability.UNKNOWN_CHECKING;
                    }
                    Log.e("ARCore-ArCoreApk", "request not running but result is null?");
                    return ArCoreApk.Availability.UNKNOWN_ERROR;
                }
            } catch (FatalException e2) {
                Log.e("ARCore-ArCoreApk", "Error while checking app details and ARCore status", e2);
                return ArCoreApk.Availability.UNKNOWN_ERROR;
            }
        }
        return ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE;
    }

    public final synchronized void d() {
        if (this.f5579a == null) {
            this.f5582e = 0;
        }
        this.f5581d = false;
        u uVar = this.i;
        if (uVar != null) {
            uVar.c();
            this.i = null;
        }
    }

    public final boolean e(Context context) {
        h(context);
        return g(context) == 0 || g(context) >= this.l;
    }

    @Override // com.google.ar.core.ArCoreApk
    public final ArCoreApk.InstallStatus requestInstall(Activity activity, boolean z) {
        ArCoreApk.UserMessageType userMessageType;
        ArCoreApk.InstallBehavior installBehavior = j(activity) ? ArCoreApk.InstallBehavior.REQUIRED : ArCoreApk.InstallBehavior.OPTIONAL;
        if (j(activity)) {
            userMessageType = ArCoreApk.UserMessageType.APPLICATION;
        } else {
            userMessageType = ArCoreApk.UserMessageType.USER_ALREADY_INFORMED;
        }
        return requestInstall(activity, z, installBehavior, userMessageType);
    }

    @Override // com.google.ar.core.ArCoreApk
    public final ArCoreApk.InstallStatus requestInstall(Activity activity, boolean z, ArCoreApk.InstallBehavior installBehavior, ArCoreApk.UserMessageType userMessageType) {
        if (i()) {
            if (e(activity)) {
                d();
                return k(activity);
            } else if (this.f5581d) {
                return ArCoreApk.InstallStatus.INSTALL_REQUESTED;
            } else {
                Exception exc = this.f5579a;
                if (exc != null) {
                    if (z) {
                        Log.w("ARCore-ArCoreApk", "Clearing previous failure: ", exc);
                        this.f5579a = null;
                    } else if (!(exc instanceof UnavailableDeviceNotCompatibleException)) {
                        if (!(exc instanceof UnavailableUserDeclinedInstallationException)) {
                            if (exc instanceof RuntimeException) {
                                throw ((RuntimeException) exc);
                            }
                            throw new RuntimeException("Unexpected exception type", exc);
                        }
                        throw ((UnavailableUserDeclinedInstallationException) exc);
                    } else {
                        throw ((UnavailableDeviceNotCompatibleException) exc);
                    }
                }
                long uptimeMillis = SystemClock.uptimeMillis();
                if (uptimeMillis - this.f5583f > 5000) {
                    this.f5582e = 0;
                }
                int i = this.f5582e + 1;
                this.f5582e = i;
                this.f5583f = uptimeMillis;
                if (i <= 2) {
                    try {
                        activity.startActivity(new Intent(activity, InstallActivity.class).putExtra(InstallActivity.MESSAGE_TYPE_KEY, userMessageType).putExtra(InstallActivity.INSTALL_BEHAVIOR_KEY, installBehavior));
                        this.f5581d = true;
                        return ArCoreApk.InstallStatus.INSTALL_REQUESTED;
                    } catch (ActivityNotFoundException e2) {
                        throw new FatalException("Failed to launch InstallActivity", e2);
                    }
                }
                throw new FatalException("Requesting ARCore installation too rapidly.");
            }
        }
        throw new UnavailableDeviceNotCompatibleException();
    }
}