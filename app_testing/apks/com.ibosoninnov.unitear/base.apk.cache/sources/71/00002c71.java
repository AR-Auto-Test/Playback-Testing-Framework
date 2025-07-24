package com.google.vr.dynamite.client;

import android.content.Context;
import android.content.pm.PackageManager;
import android.os.IBinder;
import android.os.IInterface;
import java.lang.reflect.InvocationTargetException;

/* compiled from: RemoteLibraryLoader.java */
/* loaded from: classes2.dex */
public final class e {

    /* renamed from: a  reason: collision with root package name */
    private Context f5652a;

    /* renamed from: b  reason: collision with root package name */
    private ILoadedInstanceCreator f5653b;

    /* renamed from: c  reason: collision with root package name */
    private final g f5654c;

    public e(g gVar) {
        this.f5654c = gVar;
    }

    private static IBinder c(ClassLoader classLoader) {
        try {
            return (IBinder) classLoader.loadClass("com.google.vr.dynamite.LoadedInstanceCreator").getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
        } catch (ClassNotFoundException e2) {
            throw new IllegalStateException("Unable to find dynamic class ".concat("com.google.vr.dynamite.LoadedInstanceCreator"), e2);
        } catch (IllegalAccessException e3) {
            throw new IllegalStateException("Unable to call the default constructor of ".concat("com.google.vr.dynamite.LoadedInstanceCreator"), e3);
        } catch (InstantiationException e4) {
            throw new IllegalStateException("Unable to instantiate the remote class ".concat("com.google.vr.dynamite.LoadedInstanceCreator"), e4);
        } catch (NoSuchMethodException e5) {
            throw new IllegalStateException("No constructor for dynamic class ".concat("com.google.vr.dynamite.LoadedInstanceCreator"), e5);
        } catch (InvocationTargetException e6) {
            throw new IllegalStateException("Unable to invoke constructor of dynamic class ".concat("com.google.vr.dynamite.LoadedInstanceCreator"), e6);
        }
    }

    public final synchronized Context a(Context context) {
        if (this.f5652a == null) {
            try {
                this.f5652a = context.createPackageContext(this.f5654c.a(), 3);
            } catch (PackageManager.NameNotFoundException unused) {
                throw new d();
            }
        }
        return this.f5652a;
    }

    public final synchronized ILoadedInstanceCreator b(Context context) {
        ILoadedInstanceCreator aVar;
        if (this.f5653b == null) {
            IBinder c2 = c(a(context).getClassLoader());
            if (c2 == null) {
                aVar = null;
            } else {
                IInterface queryLocalInterface = c2.queryLocalInterface("com.google.vr.dynamite.client.ILoadedInstanceCreator");
                aVar = queryLocalInterface instanceof ILoadedInstanceCreator ? (ILoadedInstanceCreator) queryLocalInterface : new a(c2);
            }
            this.f5653b = aVar;
        }
        return this.f5653b;
    }
}