package c.c.a.o;

import java.lang.reflect.InvocationTargetException;

/* compiled from: ManifestParser.java */
@Deprecated
/* loaded from: classes.dex */
public final class e {
    public static c a(String str) {
        Class cls;
        Class cls2;
        Class cls3;
        Class cls4;
        try {
            try {
                Object newInstance = Class.forName(str).getDeclaredConstructor(new Class[0]).newInstance(new Object[0]);
                if (newInstance instanceof c) {
                    return (c) newInstance;
                }
                throw new RuntimeException(c.b.a.a.a.p("Expected instanceof GlideModule, but found: ", newInstance));
            } catch (IllegalAccessException e2) {
                b(cls4, e2);
                throw null;
            } catch (InstantiationException e3) {
                b(cls3, e3);
                throw null;
            } catch (NoSuchMethodException e4) {
                b(cls2, e4);
                throw null;
            } catch (InvocationTargetException e5) {
                b(cls, e5);
                throw null;
            }
        } catch (ClassNotFoundException e6) {
            throw new IllegalArgumentException("Unable to find GlideModule implementation", e6);
        }
    }

    public static void b(Class<?> cls, Exception exc) {
        throw new RuntimeException("Unable to instantiate GlideModule implementation for " + cls, exc);
    }
}