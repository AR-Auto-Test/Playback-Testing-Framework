package c.c.a.m.v.c0;

import android.util.Log;
import java.util.HashMap;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;

/* compiled from: LruArrayPool.java */
/* loaded from: classes.dex */
public final class i implements c.c.a.m.v.c0.b {

    /* renamed from: a  reason: collision with root package name */
    public final g<a, Object> f3613a = new g<>();

    /* renamed from: b  reason: collision with root package name */
    public final b f3614b = new b();

    /* renamed from: c  reason: collision with root package name */
    public final Map<Class<?>, NavigableMap<Integer, Integer>> f3615c = new HashMap();

    /* renamed from: d  reason: collision with root package name */
    public final Map<Class<?>, c.c.a.m.v.c0.a<?>> f3616d = new HashMap();

    /* renamed from: e  reason: collision with root package name */
    public final int f3617e;

    /* renamed from: f  reason: collision with root package name */
    public int f3618f;

    /* compiled from: LruArrayPool.java */
    /* loaded from: classes.dex */
    public static final class a implements l {

        /* renamed from: a  reason: collision with root package name */
        public final b f3619a;

        /* renamed from: b  reason: collision with root package name */
        public int f3620b;

        /* renamed from: c  reason: collision with root package name */
        public Class<?> f3621c;

        public a(b bVar) {
            this.f3619a = bVar;
        }

        @Override // c.c.a.m.v.c0.l
        public void a() {
            this.f3619a.c(this);
        }

        public boolean equals(Object obj) {
            if (obj instanceof a) {
                a aVar = (a) obj;
                return this.f3620b == aVar.f3620b && this.f3621c == aVar.f3621c;
            }
            return false;
        }

        public int hashCode() {
            int i = this.f3620b * 31;
            Class<?> cls = this.f3621c;
            return i + (cls != null ? cls.hashCode() : 0);
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("Key{size=");
            x.append(this.f3620b);
            x.append("array=");
            x.append(this.f3621c);
            x.append('}');
            return x.toString();
        }
    }

    /* compiled from: LruArrayPool.java */
    /* loaded from: classes.dex */
    public static final class b extends c<a> {
        /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.c0.l' to match base method */
        @Override // c.c.a.m.v.c0.c
        public a a() {
            return new a(this);
        }

        public a d(int i, Class<?> cls) {
            a b2 = b();
            b2.f3620b = i;
            b2.f3621c = cls;
            return b2;
        }
    }

    public i(int i) {
        this.f3617e = i;
    }

    @Override // c.c.a.m.v.c0.b
    public synchronized void a(int i) {
        if (i >= 40) {
            synchronized (this) {
                f(0);
            }
        } else if (i >= 20 || i == 15) {
            f(this.f3617e / 2);
        }
    }

    @Override // c.c.a.m.v.c0.b
    public synchronized void b() {
        f(0);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: java.lang.Class<T> */
    /* JADX WARN: Multi-variable type inference failed */
    @Override // c.c.a.m.v.c0.b
    public synchronized <T> T c(int i, Class<T> cls) {
        a b2;
        b2 = this.f3614b.b();
        b2.f3620b = i;
        b2.f3621c = cls;
        return (T) h(b2, cls);
    }

    /* JADX DEBUG: Multi-variable search result rejected for r7v0, resolved type: java.lang.Class<T> */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:13:0x0023 A[Catch: all -> 0x004d, TryCatch #0 {, blocks: (B:3:0x0001, B:5:0x0013, B:7:0x0017, B:13:0x0023, B:18:0x002f, B:20:0x0047, B:19:0x003a), top: B:26:0x0001 }] */
    /* JADX WARN: Removed duplicated region for block: B:18:0x002f A[Catch: all -> 0x004d, TryCatch #0 {, blocks: (B:3:0x0001, B:5:0x0013, B:7:0x0017, B:13:0x0023, B:18:0x002f, B:20:0x0047, B:19:0x003a), top: B:26:0x0001 }] */
    /* JADX WARN: Removed duplicated region for block: B:19:0x003a A[Catch: all -> 0x004d, TryCatch #0 {, blocks: (B:3:0x0001, B:5:0x0013, B:7:0x0017, B:13:0x0023, B:18:0x002f, B:20:0x0047, B:19:0x003a), top: B:26:0x0001 }] */
    @Override // c.c.a.m.v.c0.b
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public synchronized <T> T d(int i, Class<T> cls) {
        a aVar;
        boolean z;
        Integer ceilingKey = i(cls).ceilingKey(Integer.valueOf(i));
        boolean z2 = true;
        if (ceilingKey != null) {
            int i2 = this.f3618f;
            if (i2 != 0 && this.f3617e / i2 < 2) {
                z = false;
                if (!z) {
                    if (ceilingKey.intValue() > i * 8) {
                    }
                }
                if (z2) {
                    aVar = this.f3614b.d(ceilingKey.intValue(), cls);
                } else {
                    a b2 = this.f3614b.b();
                    b2.f3620b = i;
                    b2.f3621c = cls;
                    aVar = b2;
                }
            }
            z = true;
            if (!z) {
            }
            if (z2) {
            }
        }
        z2 = false;
        if (z2) {
        }
        return (T) h(aVar, cls);
    }

    public final void e(int i, Class<?> cls) {
        NavigableMap<Integer, Integer> i2 = i(cls);
        Integer num = (Integer) i2.get(Integer.valueOf(i));
        if (num != null) {
            if (num.intValue() == 1) {
                i2.remove(Integer.valueOf(i));
                return;
            } else {
                i2.put(Integer.valueOf(i), Integer.valueOf(num.intValue() - 1));
                return;
            }
        }
        throw new NullPointerException("Tried to decrement empty size, size: " + i + ", this: " + this);
    }

    public final void f(int i) {
        while (this.f3618f > i) {
            Object c2 = this.f3613a.c();
            Objects.requireNonNull(c2, "Argument must not be null");
            c.c.a.m.v.c0.a g2 = g(c2.getClass());
            this.f3618f -= g2.a() * g2.b(c2);
            e(g2.b(c2), c2.getClass());
            if (Log.isLoggable(g2.getTag(), 2)) {
                String tag = g2.getTag();
                StringBuilder x = c.b.a.a.a.x("evicted: ");
                x.append(g2.b(c2));
                Log.v(tag, x.toString());
            }
        }
    }

    public final <T> c.c.a.m.v.c0.a<T> g(Class<T> cls) {
        h hVar = (c.c.a.m.v.c0.a<T>) this.f3616d.get(cls);
        if (hVar == null) {
            if (cls.equals(int[].class)) {
                hVar = new h();
            } else if (cls.equals(byte[].class)) {
                hVar = new f();
            } else {
                StringBuilder x = c.b.a.a.a.x("No array pool found for: ");
                x.append(cls.getSimpleName());
                throw new IllegalArgumentException(x.toString());
            }
            this.f3616d.put(cls, hVar);
        }
        return hVar;
    }

    public final <T> T h(a aVar, Class<T> cls) {
        c.c.a.m.v.c0.a<T> g2 = g(cls);
        T t = (T) this.f3613a.a(aVar);
        if (t != null) {
            this.f3618f -= g2.a() * g2.b(t);
            e(g2.b(t), cls);
        }
        if (t == null) {
            if (Log.isLoggable(g2.getTag(), 2)) {
                String tag = g2.getTag();
                StringBuilder x = c.b.a.a.a.x("Allocated ");
                x.append(aVar.f3620b);
                x.append(" bytes");
                Log.v(tag, x.toString());
            }
            return g2.newArray(aVar.f3620b);
        }
        return t;
    }

    public final NavigableMap<Integer, Integer> i(Class<?> cls) {
        NavigableMap<Integer, Integer> navigableMap = this.f3615c.get(cls);
        if (navigableMap == null) {
            TreeMap treeMap = new TreeMap();
            this.f3615c.put(cls, treeMap);
            return treeMap;
        }
        return navigableMap;
    }

    @Override // c.c.a.m.v.c0.b
    public synchronized <T> void put(T t) {
        Class<?> cls = t.getClass();
        c.c.a.m.v.c0.a<T> g2 = g(cls);
        int b2 = g2.b(t);
        int a2 = g2.a() * b2;
        int i = 1;
        if (a2 <= this.f3617e / 2) {
            a d2 = this.f3614b.d(b2, cls);
            this.f3613a.b(d2, t);
            NavigableMap<Integer, Integer> i2 = i(cls);
            Integer num = (Integer) i2.get(Integer.valueOf(d2.f3620b));
            Integer valueOf = Integer.valueOf(d2.f3620b);
            if (num != null) {
                i = 1 + num.intValue();
            }
            i2.put(valueOf, Integer.valueOf(i));
            this.f3618f += a2;
            f(this.f3617e);
        }
    }
}