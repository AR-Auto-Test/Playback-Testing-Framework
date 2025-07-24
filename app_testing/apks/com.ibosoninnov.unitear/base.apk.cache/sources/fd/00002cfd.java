package f.g0.f;

import java.io.IOException;
import java.lang.reflect.Method;

/* compiled from: RouteException.java */
/* loaded from: classes2.dex */
public final class e extends RuntimeException {

    /* renamed from: b  reason: collision with root package name */
    public static final Method f5798b;

    /* renamed from: c  reason: collision with root package name */
    public IOException f5799c;

    static {
        Method method;
        try {
            method = Throwable.class.getDeclaredMethod("addSuppressed", Throwable.class);
        } catch (Exception unused) {
            method = null;
        }
        f5798b = method;
    }

    public e(IOException iOException) {
        super(iOException);
        this.f5799c = iOException;
    }
}