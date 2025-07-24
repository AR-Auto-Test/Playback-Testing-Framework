package com.google.vr.dynamite.client;

import android.content.Context;
import android.os.RemoteException;
import android.util.ArrayMap;
import android.util.Log;
import dalvik.system.DexClassLoader;

@UsedByNative
/* loaded from: classes2.dex */
public final class DynamiteClient {

    /* renamed from: a  reason: collision with root package name */
    private static final ArrayMap<g, e> f5650a = new ArrayMap<>();

    private DynamiteClient() {
    }

    @UsedByNative
    public static synchronized int checkVersion(Context context, String str, String str2, String str3) {
        synchronized (DynamiteClient.class) {
            g gVar = new g(str, str2);
            e remoteLibraryLoaderFromInfo = getRemoteLibraryLoaderFromInfo(gVar);
            try {
                INativeLibraryLoader newNativeLibraryLoader = remoteLibraryLoaderFromInfo.b(context).newNativeLibraryLoader(ObjectWrapper.b(remoteLibraryLoaderFromInfo.a(context)), ObjectWrapper.b(context));
                if (newNativeLibraryLoader == null) {
                    String gVar2 = gVar.toString();
                    StringBuilder sb = new StringBuilder(gVar2.length() + 72);
                    sb.append("Failed to load native library ");
                    sb.append(gVar2);
                    sb.append(" from remote package: no loader available.");
                    Log.e("DynamiteClient", sb.toString());
                    return -1;
                }
                return newNativeLibraryLoader.checkVersion(str3);
            } catch (RemoteException | d | IllegalArgumentException | IllegalStateException | SecurityException | UnsatisfiedLinkError e2) {
                String gVar3 = gVar.toString();
                StringBuilder sb2 = new StringBuilder(gVar3.length() + 54);
                sb2.append("Failed to load native library ");
                sb2.append(gVar3);
                sb2.append(" from remote package:\n  ");
                Log.e("DynamiteClient", sb2.toString(), e2);
                return -1;
            }
        }
    }

    @UsedByNative
    public static synchronized ClassLoader getRemoteClassLoader(Context context, String str, String str2) {
        synchronized (DynamiteClient.class) {
            Context remoteContext = getRemoteContext(context, str, str2);
            if (remoteContext == null) {
                return null;
            }
            return remoteContext.getClassLoader();
        }
    }

    @UsedByNative
    public static synchronized Context getRemoteContext(Context context, String str, String str2) {
        Context a2;
        synchronized (DynamiteClient.class) {
            g gVar = new g(str, str2);
            try {
                a2 = getRemoteLibraryLoaderFromInfo(gVar).a(context);
            } catch (d e2) {
                String gVar2 = gVar.toString();
                StringBuilder sb = new StringBuilder(gVar2.length() + 52);
                sb.append("Failed to get remote Context");
                sb.append(gVar2);
                sb.append(" from remote package:\n  ");
                Log.e("DynamiteClient", sb.toString(), e2);
                return null;
            }
        }
        return a2;
    }

    @UsedByNative
    public static synchronized ClassLoader getRemoteDexClassLoader(Context context, String str) {
        synchronized (DynamiteClient.class) {
            Context remoteContext = getRemoteContext(context, str, null);
            if (remoteContext == null) {
                return null;
            }
            try {
                return new DexClassLoader(remoteContext.getPackageCodePath(), context.getCodeCacheDir().getAbsolutePath(), remoteContext.getApplicationInfo().nativeLibraryDir, context.getClassLoader());
            } catch (RuntimeException e2) {
                Log.e("DynamiteClient", "Failed to create class loader for remote package\n ", e2);
                return null;
            }
        }
    }

    @UsedByNative
    private static synchronized e getRemoteLibraryLoaderFromInfo(g gVar) {
        synchronized (DynamiteClient.class) {
            ArrayMap<g, e> arrayMap = f5650a;
            e eVar = arrayMap.get(gVar);
            if (eVar == null) {
                e eVar2 = new e(gVar);
                arrayMap.put(gVar, eVar2);
                return eVar2;
            }
            return eVar;
        }
    }

    @UsedByNative
    public static synchronized long loadNativeRemoteLibrary(Context context, String str, String str2) {
        synchronized (DynamiteClient.class) {
            g gVar = new g(str, str2);
            e remoteLibraryLoaderFromInfo = getRemoteLibraryLoaderFromInfo(gVar);
            try {
                INativeLibraryLoader newNativeLibraryLoader = remoteLibraryLoaderFromInfo.b(context).newNativeLibraryLoader(ObjectWrapper.b(remoteLibraryLoaderFromInfo.a(context)), ObjectWrapper.b(context));
                if (newNativeLibraryLoader == null) {
                    String gVar2 = gVar.toString();
                    StringBuilder sb = new StringBuilder(gVar2.length() + 72);
                    sb.append("Failed to load native library ");
                    sb.append(gVar2);
                    sb.append(" from remote package: no loader available.");
                    Log.e("DynamiteClient", sb.toString());
                    return 0L;
                }
                return newNativeLibraryLoader.initializeAndLoadNativeLibrary(str2);
            } catch (RemoteException | d | IllegalArgumentException | IllegalStateException | SecurityException | UnsatisfiedLinkError e2) {
                String gVar3 = gVar.toString();
                StringBuilder sb2 = new StringBuilder(gVar3.length() + 54);
                sb2.append("Failed to load native library ");
                sb2.append(gVar3);
                sb2.append(" from remote package:\n  ");
                Log.e("DynamiteClient", sb2.toString(), e2);
                return 0L;
            }
        }
    }
}