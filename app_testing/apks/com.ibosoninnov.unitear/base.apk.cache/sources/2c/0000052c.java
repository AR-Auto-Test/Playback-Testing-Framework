package b.q.b;

import androidx.fragment.app.Fragment;

/* compiled from: FragmentFactory.java */
/* loaded from: classes.dex */
public class m {

    /* renamed from: a  reason: collision with root package name */
    public static final b.f.h<String, Class<?>> f2488a = new b.f.h<>();

    public static Class<?> b(ClassLoader classLoader, String str) {
        b.f.h<String, Class<?>> hVar = f2488a;
        Class<?> orDefault = hVar.getOrDefault(str, null);
        if (orDefault == null) {
            Class<?> cls = Class.forName(str, false, classLoader);
            hVar.put(str, cls);
            return cls;
        }
        return orDefault;
    }

    /* JADX DEBUG: Type inference failed for r3v3. Raw type applied. Possible types: java.lang.Class<?>, java.lang.Class<? extends androidx.fragment.app.Fragment> */
    public static Class<? extends Fragment> c(ClassLoader classLoader, String str) {
        try {
            return b(classLoader, str);
        } catch (ClassCastException e2) {
            throw new Fragment.e(c.b.a.a.a.r("Unable to instantiate fragment ", str, ": make sure class is a valid subclass of Fragment"), e2);
        } catch (ClassNotFoundException e3) {
            throw new Fragment.e(c.b.a.a.a.r("Unable to instantiate fragment ", str, ": make sure class name exists"), e3);
        }
    }

    public Fragment a(ClassLoader classLoader, String str) {
        throw null;
    }
}