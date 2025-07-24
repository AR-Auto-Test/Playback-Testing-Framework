package f.g0.j;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/* compiled from: OptionalMethod.java */
/* loaded from: classes2.dex */
public class e<T> {

    /* renamed from: a  reason: collision with root package name */
    public final Class<?> f6029a;

    /* renamed from: b  reason: collision with root package name */
    public final String f6030b;

    /* renamed from: c  reason: collision with root package name */
    public final Class[] f6031c;

    public e(Class<?> cls, String str, Class... clsArr) {
        this.f6029a = cls;
        this.f6030b = str;
        this.f6031c = clsArr;
    }

    /* JADX WARN: Code restructure failed: missing block: B:8:0x0011, code lost:
        if ((r4.getModifiers() & 1) == 0) goto L11;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Method a(Class<?> cls) {
        Method method;
        Class<?> cls2;
        String str = this.f6030b;
        if (str == null) {
            return null;
        }
        try {
            method = cls.getMethod(str, this.f6031c);
        } catch (NoSuchMethodException unused) {
        }
        method = null;
        if (method != null || (cls2 = this.f6029a) == null || cls2.isAssignableFrom(method.getReturnType())) {
            return method;
        }
        return null;
        if (method != null) {
        }
        return method;
    }

    public Object b(T t, Object... objArr) {
        Method a2 = a(t.getClass());
        if (a2 != null) {
            try {
                return a2.invoke(t, objArr);
            } catch (IllegalAccessException e2) {
                AssertionError assertionError = new AssertionError("Unexpectedly could not call: " + a2);
                assertionError.initCause(e2);
                throw assertionError;
            }
        }
        StringBuilder x = c.b.a.a.a.x("Method ");
        x.append(this.f6030b);
        x.append(" not supported for object ");
        x.append(t);
        throw new AssertionError(x.toString());
    }

    public Object c(T t, Object... objArr) {
        try {
            Method a2 = a(t.getClass());
            if (a2 != null) {
                try {
                } catch (IllegalAccessException unused) {
                    return null;
                }
            }
            return a2.invoke(t, objArr);
        } catch (InvocationTargetException e2) {
            Throwable targetException = e2.getTargetException();
            if (targetException instanceof RuntimeException) {
                throw ((RuntimeException) targetException);
            }
            AssertionError assertionError = new AssertionError("Unexpected exception");
            assertionError.initCause(targetException);
            throw assertionError;
        }
    }

    public Object d(T t, Object... objArr) {
        try {
            return b(t, objArr);
        } catch (InvocationTargetException e2) {
            Throwable targetException = e2.getTargetException();
            if (targetException instanceof RuntimeException) {
                throw ((RuntimeException) targetException);
            }
            AssertionError assertionError = new AssertionError("Unexpected exception");
            assertionError.initCause(targetException);
            throw assertionError;
        }
    }
}