package com.google.common.flogger.backend;

import com.google.common.flogger.backend.system.DefaultPlatform;
import java.lang.reflect.InvocationTargetException;

/* loaded from: classes.dex */
public final class PlatformProvider {
    private PlatformProvider() {
    }

    public static Platform getPlatform() {
        try {
            return (Platform) DefaultPlatform.class.getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
        } catch (IllegalAccessException | InstantiationException | NoClassDefFoundError | NoSuchMethodException | InvocationTargetException unused) {
            return null;
        }
    }
}