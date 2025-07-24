package com.google.common.flogger.util;

import com.google.common.base.Throwables;
import com.google.errorprone.annotations.CheckReturnValue;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

@CheckReturnValue
/* loaded from: classes.dex */
public final class FastStackGetter {
    private final Method getDepthMethod;
    private final Method getElementMethod;
    private final Object javaLangAccess;

    private FastStackGetter(Object obj, Method method, Method method2) {
        this.javaLangAccess = obj;
        this.getElementMethod = method;
        this.getDepthMethod = method2;
    }

    public static FastStackGetter createIfSupported() {
        try {
            Object invoke = Class.forName(Throwables.SHARED_SECRETS_CLASSNAME).getMethod("getJavaLangAccess", new Class[0]).invoke(null, new Object[0]);
            Method method = Class.forName("sun.misc.JavaLangAccess").getMethod("getStackTraceElement", Throwable.class, Integer.TYPE);
            Method method2 = Class.forName("sun.misc.JavaLangAccess").getMethod("getStackTraceDepth", Throwable.class);
            StackTraceElement stackTraceElement = (StackTraceElement) method.invoke(invoke, new Throwable(), 0);
            ((Integer) method2.invoke(invoke, new Throwable())).intValue();
            return new FastStackGetter(invoke, method, method2);
        } catch (ThreadDeath e2) {
            throw e2;
        } catch (Throwable unused) {
            return null;
        }
    }

    public int getStackTraceDepth(Throwable th) {
        try {
            return ((Integer) this.getDepthMethod.invoke(this.javaLangAccess, th)).intValue();
        } catch (IllegalAccessException e2) {
            throw new AssertionError(e2);
        } catch (InvocationTargetException e3) {
            if (!(e3.getCause() instanceof RuntimeException)) {
                if (e3.getCause() instanceof Error) {
                    throw ((Error) e3.getCause());
                }
                throw new RuntimeException(e3.getCause());
            }
            throw ((RuntimeException) e3.getCause());
        }
    }

    public StackTraceElement getStackTraceElement(Throwable th, int i) {
        try {
            return (StackTraceElement) this.getElementMethod.invoke(this.javaLangAccess, th, Integer.valueOf(i));
        } catch (IllegalAccessException e2) {
            throw new AssertionError(e2);
        } catch (InvocationTargetException e3) {
            if (!(e3.getCause() instanceof RuntimeException)) {
                if (e3.getCause() instanceof Error) {
                    throw ((Error) e3.getCause());
                }
                throw new RuntimeException(e3.getCause());
            }
            throw ((RuntimeException) e3.getCause());
        }
    }
}