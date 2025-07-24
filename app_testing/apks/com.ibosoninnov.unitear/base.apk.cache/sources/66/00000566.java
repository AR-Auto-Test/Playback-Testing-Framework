package b.t;

import android.app.Application;
import java.lang.reflect.InvocationTargetException;

/* compiled from: ViewModelProvider.java */
/* loaded from: classes.dex */
public class t extends w {

    /* renamed from: a  reason: collision with root package name */
    public static t f2602a;

    /* renamed from: b  reason: collision with root package name */
    public Application f2603b;

    public t(Application application) {
        this.f2603b = application;
    }

    @Override // b.t.u
    public <T extends s> T a(Class<T> cls) {
        if (a.class.isAssignableFrom(cls)) {
            try {
                return cls.getConstructor(Application.class).newInstance(this.f2603b);
            } catch (IllegalAccessException e2) {
                throw new RuntimeException("Cannot create an instance of " + cls, e2);
            } catch (InstantiationException e3) {
                throw new RuntimeException("Cannot create an instance of " + cls, e3);
            } catch (NoSuchMethodException e4) {
                throw new RuntimeException("Cannot create an instance of " + cls, e4);
            } catch (InvocationTargetException e5) {
                throw new RuntimeException("Cannot create an instance of " + cls, e5);
            }
        }
        try {
            return cls.newInstance();
        } catch (IllegalAccessException e6) {
            throw new RuntimeException("Cannot create an instance of " + cls, e6);
        } catch (InstantiationException e7) {
            throw new RuntimeException("Cannot create an instance of " + cls, e7);
        }
    }
}