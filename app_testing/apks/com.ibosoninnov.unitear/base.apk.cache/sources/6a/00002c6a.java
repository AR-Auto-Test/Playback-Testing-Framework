package com.google.vr.dynamite.client;

import android.os.IBinder;
import java.lang.reflect.Field;

@UsedByReflection
/* loaded from: classes2.dex */
public final class ObjectWrapper<T> extends c {
    @UsedByReflection
    private final T wrappedObject;

    private ObjectWrapper(T t) {
        this.wrappedObject = t;
    }

    public static <T> IObjectWrapper b(T t) {
        return new ObjectWrapper(t);
    }

    @UsedByReflection
    public static <T> T unwrap(IObjectWrapper iObjectWrapper, Class<T> cls) {
        if (iObjectWrapper == null) {
            return null;
        }
        if (iObjectWrapper instanceof ObjectWrapper) {
            return ((ObjectWrapper) iObjectWrapper).wrappedObject;
        }
        IBinder asBinder = iObjectWrapper.asBinder();
        Field[] declaredFields = asBinder.getClass().getDeclaredFields();
        int length = declaredFields.length;
        int i = 0;
        Field field = null;
        while (true) {
            if (i >= length) {
                break;
            }
            Field field2 = declaredFields[i];
            if (!field2.isSynthetic()) {
                if (field != null) {
                    field = null;
                    break;
                }
                field = field2;
            }
            i++;
        }
        if (field != null) {
            if (!field.isAccessible()) {
                field.setAccessible(true);
                try {
                    Object obj = field.get(asBinder);
                    if (obj == null) {
                        return null;
                    }
                    if (cls.isInstance(obj)) {
                        return cls.cast(obj);
                    }
                    throw new IllegalArgumentException("remoteBinder is the wrong class.");
                } catch (IllegalAccessException e2) {
                    throw new IllegalArgumentException("Could not access the field in remoteBinder.", e2);
                } catch (IllegalArgumentException e3) {
                    throw new IllegalArgumentException("remoteBinder is the wrong class.", e3);
                } catch (NullPointerException e4) {
                    throw new IllegalArgumentException("Binder object is null.", e4);
                }
            }
            throw new IllegalArgumentException("The concrete class implementing IObjectWrapper must have exactly one declared *private* field for the wrapped object. Preferably, this is an instance of the ObjectWrapper<T> class.");
        }
        throw new IllegalArgumentException("The concrete class implementing IObjectWrapper must have exactly *one* declared private field for the wrapped object.  Preferably, this is an instance of the ObjectWrapper<T> class.");
    }
}