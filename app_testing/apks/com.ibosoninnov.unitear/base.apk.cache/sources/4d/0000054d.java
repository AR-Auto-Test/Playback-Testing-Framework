package b.t;

import b.t.e;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/* compiled from: ClassesInfoCache.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public static b f2565a = new b();

    /* renamed from: b  reason: collision with root package name */
    public final Map<Class<?>, a> f2566b = new HashMap();

    /* renamed from: c  reason: collision with root package name */
    public final Map<Class<?>, Boolean> f2567c = new HashMap();

    /* compiled from: ClassesInfoCache.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public final Map<e.a, List<C0048b>> f2568a = new HashMap();

        /* renamed from: b  reason: collision with root package name */
        public final Map<C0048b, e.a> f2569b;

        public a(Map<C0048b, e.a> map) {
            this.f2569b = map;
            for (Map.Entry<C0048b, e.a> entry : map.entrySet()) {
                e.a value = entry.getValue();
                List<C0048b> list = this.f2568a.get(value);
                if (list == null) {
                    list = new ArrayList<>();
                    this.f2568a.put(value, list);
                }
                list.add(entry.getKey());
            }
        }

        public static void a(List<C0048b> list, h hVar, e.a aVar, Object obj) {
            if (list != null) {
                for (int size = list.size() - 1; size >= 0; size--) {
                    C0048b c0048b = list.get(size);
                    Objects.requireNonNull(c0048b);
                    try {
                        int i = c0048b.f2570a;
                        if (i == 0) {
                            c0048b.f2571b.invoke(obj, new Object[0]);
                        } else if (i == 1) {
                            c0048b.f2571b.invoke(obj, hVar);
                        } else if (i == 2) {
                            c0048b.f2571b.invoke(obj, hVar, aVar);
                        }
                    } catch (IllegalAccessException e2) {
                        throw new RuntimeException(e2);
                    } catch (InvocationTargetException e3) {
                        throw new RuntimeException("Failed to call observer method", e3.getCause());
                    }
                }
            }
        }
    }

    /* compiled from: ClassesInfoCache.java */
    /* renamed from: b.t.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0048b {

        /* renamed from: a  reason: collision with root package name */
        public final int f2570a;

        /* renamed from: b  reason: collision with root package name */
        public final Method f2571b;

        public C0048b(int i, Method method) {
            this.f2570a = i;
            this.f2571b = method;
            method.setAccessible(true);
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || C0048b.class != obj.getClass()) {
                return false;
            }
            C0048b c0048b = (C0048b) obj;
            return this.f2570a == c0048b.f2570a && this.f2571b.getName().equals(c0048b.f2571b.getName());
        }

        public int hashCode() {
            return this.f2571b.getName().hashCode() + (this.f2570a * 31);
        }
    }

    public final a a(Class<?> cls, Method[] methodArr) {
        int i;
        a b2;
        Class<?> superclass = cls.getSuperclass();
        HashMap hashMap = new HashMap();
        if (superclass != null && (b2 = b(superclass)) != null) {
            hashMap.putAll(b2.f2569b);
        }
        for (Class<?> cls2 : cls.getInterfaces()) {
            for (Map.Entry<C0048b, e.a> entry : b(cls2).f2569b.entrySet()) {
                c(hashMap, entry.getKey(), entry.getValue(), cls);
            }
        }
        if (methodArr == null) {
            try {
                methodArr = cls.getDeclaredMethods();
            } catch (NoClassDefFoundError e2) {
                throw new IllegalArgumentException("The observer class has some methods that use newer APIs which are not available in the current OS version. Lifecycles cannot access even other methods so you should make sure that your observer classes only access framework classes that are available in your min API level OR use lifecycle:compiler annotation processor.", e2);
            }
        }
        boolean z = false;
        for (Method method : methodArr) {
            o oVar = (o) method.getAnnotation(o.class);
            if (oVar != null) {
                Class<?>[] parameterTypes = method.getParameterTypes();
                if (parameterTypes.length <= 0) {
                    i = 0;
                } else if (!parameterTypes[0].isAssignableFrom(h.class)) {
                    throw new IllegalArgumentException("invalid parameter type. Must be one and instanceof LifecycleOwner");
                } else {
                    i = 1;
                }
                e.a value = oVar.value();
                if (parameterTypes.length > 1) {
                    if (parameterTypes[1].isAssignableFrom(e.a.class)) {
                        if (value != e.a.ON_ANY) {
                            throw new IllegalArgumentException("Second arg is supported only for ON_ANY value");
                        }
                        i = 2;
                    } else {
                        throw new IllegalArgumentException("invalid parameter type. second arg must be an event");
                    }
                }
                if (parameterTypes.length <= 2) {
                    c(hashMap, new C0048b(i, method), value, cls);
                    z = true;
                } else {
                    throw new IllegalArgumentException("cannot have more than 2 params");
                }
            }
        }
        a aVar = new a(hashMap);
        this.f2566b.put(cls, aVar);
        this.f2567c.put(cls, Boolean.valueOf(z));
        return aVar;
    }

    public a b(Class<?> cls) {
        a aVar = this.f2566b.get(cls);
        return aVar != null ? aVar : a(cls, null);
    }

    public final void c(Map<C0048b, e.a> map, C0048b c0048b, e.a aVar, Class<?> cls) {
        e.a aVar2 = map.get(c0048b);
        if (aVar2 == null || aVar == aVar2) {
            if (aVar2 == null) {
                map.put(c0048b, aVar);
                return;
            }
            return;
        }
        Method method = c0048b.f2571b;
        StringBuilder x = c.b.a.a.a.x("Method ");
        x.append(method.getName());
        x.append(" in ");
        x.append(cls.getName());
        x.append(" already declared with different @OnLifecycleEvent value: previous value ");
        x.append(aVar2);
        x.append(", new value ");
        x.append(aVar);
        throw new IllegalArgumentException(x.toString());
    }
}