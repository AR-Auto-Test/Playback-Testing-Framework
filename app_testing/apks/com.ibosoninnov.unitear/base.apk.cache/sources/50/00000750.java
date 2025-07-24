package c.c.a.m.v.c0;

import android.graphics.Bitmap;
import android.os.Build;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

/* compiled from: SizeConfigStrategy.java */
/* loaded from: classes.dex */
public class m implements k {

    /* renamed from: a  reason: collision with root package name */
    public static final Bitmap.Config[] f3630a;

    /* renamed from: b  reason: collision with root package name */
    public static final Bitmap.Config[] f3631b;

    /* renamed from: c  reason: collision with root package name */
    public static final Bitmap.Config[] f3632c;

    /* renamed from: d  reason: collision with root package name */
    public static final Bitmap.Config[] f3633d;

    /* renamed from: e  reason: collision with root package name */
    public static final Bitmap.Config[] f3634e;

    /* renamed from: f  reason: collision with root package name */
    public final c f3635f = new c();

    /* renamed from: g  reason: collision with root package name */
    public final g<b, Bitmap> f3636g = new g<>();

    /* renamed from: h  reason: collision with root package name */
    public final Map<Bitmap.Config, NavigableMap<Integer, Integer>> f3637h = new HashMap();

    /* compiled from: SizeConfigStrategy.java */
    /* loaded from: classes.dex */
    public static /* synthetic */ class a {

        /* renamed from: a  reason: collision with root package name */
        public static final /* synthetic */ int[] f3638a;

        static {
            int[] iArr = new int[Bitmap.Config.values().length];
            f3638a = iArr;
            try {
                iArr[Bitmap.Config.ARGB_8888.ordinal()] = 1;
            } catch (NoSuchFieldError unused) {
            }
            try {
                f3638a[Bitmap.Config.RGB_565.ordinal()] = 2;
            } catch (NoSuchFieldError unused2) {
            }
            try {
                f3638a[Bitmap.Config.ARGB_4444.ordinal()] = 3;
            } catch (NoSuchFieldError unused3) {
            }
            try {
                f3638a[Bitmap.Config.ALPHA_8.ordinal()] = 4;
            } catch (NoSuchFieldError unused4) {
            }
        }
    }

    /* compiled from: SizeConfigStrategy.java */
    /* loaded from: classes.dex */
    public static final class b implements l {

        /* renamed from: a  reason: collision with root package name */
        public final c f3639a;

        /* renamed from: b  reason: collision with root package name */
        public int f3640b;

        /* renamed from: c  reason: collision with root package name */
        public Bitmap.Config f3641c;

        public b(c cVar) {
            this.f3639a = cVar;
        }

        @Override // c.c.a.m.v.c0.l
        public void a() {
            this.f3639a.c(this);
        }

        public boolean equals(Object obj) {
            if (obj instanceof b) {
                b bVar = (b) obj;
                return this.f3640b == bVar.f3640b && c.c.a.s.j.b(this.f3641c, bVar.f3641c);
            }
            return false;
        }

        public int hashCode() {
            int i = this.f3640b * 31;
            Bitmap.Config config = this.f3641c;
            return i + (config != null ? config.hashCode() : 0);
        }

        public String toString() {
            return m.c(this.f3640b, this.f3641c);
        }
    }

    /* compiled from: SizeConfigStrategy.java */
    /* loaded from: classes.dex */
    public static class c extends c.c.a.m.v.c0.c<b> {
        /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.c0.l' to match base method */
        @Override // c.c.a.m.v.c0.c
        public b a() {
            return new b(this);
        }

        public b d(int i, Bitmap.Config config) {
            b b2 = b();
            b2.f3640b = i;
            b2.f3641c = config;
            return b2;
        }
    }

    static {
        Bitmap.Config[] configArr = {Bitmap.Config.ARGB_8888, null};
        if (Build.VERSION.SDK_INT >= 26) {
            configArr = (Bitmap.Config[]) Arrays.copyOf(configArr, 3);
            configArr[configArr.length - 1] = Bitmap.Config.RGBA_F16;
        }
        f3630a = configArr;
        f3631b = configArr;
        f3632c = new Bitmap.Config[]{Bitmap.Config.RGB_565};
        f3633d = new Bitmap.Config[]{Bitmap.Config.ARGB_4444};
        f3634e = new Bitmap.Config[]{Bitmap.Config.ALPHA_8};
    }

    public static String c(int i, Bitmap.Config config) {
        return "[" + i + "](" + config + ")";
    }

    public final void a(Integer num, Bitmap bitmap) {
        NavigableMap<Integer, Integer> d2 = d(bitmap.getConfig());
        Integer num2 = (Integer) d2.get(num);
        if (num2 != null) {
            if (num2.intValue() == 1) {
                d2.remove(num);
                return;
            } else {
                d2.put(num, Integer.valueOf(num2.intValue() - 1));
                return;
            }
        }
        throw new NullPointerException("Tried to decrement empty size, size: " + num + ", removed: " + e(bitmap) + ", this: " + this);
    }

    public Bitmap b(int i, int i2, Bitmap.Config config) {
        Bitmap.Config[] configArr;
        int c2 = c.c.a.s.j.c(i, i2, config);
        b b2 = this.f3635f.b();
        b2.f3640b = c2;
        b2.f3641c = config;
        int i3 = 0;
        if (Build.VERSION.SDK_INT >= 26 && Bitmap.Config.RGBA_F16.equals(config)) {
            configArr = f3631b;
        } else {
            int i4 = a.f3638a[config.ordinal()];
            if (i4 == 1) {
                configArr = f3630a;
            } else if (i4 == 2) {
                configArr = f3632c;
            } else if (i4 != 3) {
                configArr = i4 != 4 ? new Bitmap.Config[]{config} : f3634e;
            } else {
                configArr = f3633d;
            }
        }
        int length = configArr.length;
        while (true) {
            if (i3 >= length) {
                break;
            }
            Bitmap.Config config2 = configArr[i3];
            Integer ceilingKey = d(config2).ceilingKey(Integer.valueOf(c2));
            if (ceilingKey == null || ceilingKey.intValue() > c2 * 8) {
                i3++;
            } else if (ceilingKey.intValue() != c2 || (config2 != null ? !config2.equals(config) : config != null)) {
                this.f3635f.c(b2);
                b2 = this.f3635f.d(ceilingKey.intValue(), config2);
            }
        }
        Bitmap a2 = this.f3636g.a(b2);
        if (a2 != null) {
            a(Integer.valueOf(b2.f3640b), a2);
            a2.reconfigure(i, i2, config);
        }
        return a2;
    }

    public final NavigableMap<Integer, Integer> d(Bitmap.Config config) {
        NavigableMap<Integer, Integer> navigableMap = this.f3637h.get(config);
        if (navigableMap == null) {
            TreeMap treeMap = new TreeMap();
            this.f3637h.put(config, treeMap);
            return treeMap;
        }
        return navigableMap;
    }

    public String e(Bitmap bitmap) {
        return c(c.c.a.s.j.d(bitmap), bitmap.getConfig());
    }

    public void f(Bitmap bitmap) {
        b d2 = this.f3635f.d(c.c.a.s.j.d(bitmap), bitmap.getConfig());
        this.f3636g.b(d2, bitmap);
        NavigableMap<Integer, Integer> d3 = d(bitmap.getConfig());
        Integer num = (Integer) d3.get(Integer.valueOf(d2.f3640b));
        d3.put(Integer.valueOf(d2.f3640b), Integer.valueOf(num != null ? 1 + num.intValue() : 1));
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("SizeConfigStrategy{groupedMap=");
        x.append(this.f3636g);
        x.append(", sortedSizes=(");
        for (Map.Entry<Bitmap.Config, NavigableMap<Integer, Integer>> entry : this.f3637h.entrySet()) {
            x.append(entry.getKey());
            x.append('[');
            x.append(entry.getValue());
            x.append("], ");
        }
        if (!this.f3637h.isEmpty()) {
            x.replace(x.length() - 2, x.length(), "");
        }
        x.append(")}");
        return x.toString();
    }
}