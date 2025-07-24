package b.v;

import android.os.Bundle;
import android.os.Parcelable;
import java.io.Serializable;

/* compiled from: NavType.java */
/* loaded from: classes.dex */
public abstract class p<T> {

    /* renamed from: a  reason: collision with root package name */
    public static final p<Integer> f2669a = new c(false);

    /* renamed from: b  reason: collision with root package name */
    public static final p<Integer> f2670b = new d(false);

    /* renamed from: c  reason: collision with root package name */
    public static final p<int[]> f2671c = new e(true);

    /* renamed from: d  reason: collision with root package name */
    public static final p<Long> f2672d = new f(false);

    /* renamed from: e  reason: collision with root package name */
    public static final p<long[]> f2673e = new g(true);

    /* renamed from: f  reason: collision with root package name */
    public static final p<Float> f2674f = new h(false);

    /* renamed from: g  reason: collision with root package name */
    public static final p<float[]> f2675g = new i(true);

    /* renamed from: h  reason: collision with root package name */
    public static final p<Boolean> f2676h = new j(false);
    public static final p<boolean[]> i = new k(true);
    public static final p<String> j = new a(true);
    public static final p<String[]> k = new b(true);
    public final boolean l;

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class a extends p<String> {
        public a(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public String a(Bundle bundle, String str) {
            return (String) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "string";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public String c(String str) {
            return str;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, String str2) {
            bundle.putString(str, str2);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class b extends p<String[]> {
        public b(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public String[] a(Bundle bundle, String str) {
            return (String[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "string[]";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public String[] c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, String[] strArr) {
            bundle.putStringArray(str, strArr);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class c extends p<Integer> {
        public c(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Integer a(Bundle bundle, String str) {
            return (Integer) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "integer";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Integer c(String str) {
            if (str.startsWith("0x")) {
                return Integer.valueOf(Integer.parseInt(str.substring(2), 16));
            }
            return Integer.valueOf(Integer.parseInt(str));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Integer num) {
            bundle.putInt(str, num.intValue());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class d extends p<Integer> {
        public d(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Integer a(Bundle bundle, String str) {
            return (Integer) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "reference";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Integer c(String str) {
            if (str.startsWith("0x")) {
                return Integer.valueOf(Integer.parseInt(str.substring(2), 16));
            }
            return Integer.valueOf(Integer.parseInt(str));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Integer num) {
            bundle.putInt(str, num.intValue());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class e extends p<int[]> {
        public e(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public int[] a(Bundle bundle, String str) {
            return (int[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "integer[]";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public int[] c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, int[] iArr) {
            bundle.putIntArray(str, iArr);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class f extends p<Long> {
        public f(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Long a(Bundle bundle, String str) {
            return (Long) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "long";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Long c(String str) {
            if (str.endsWith("L")) {
                str = str.substring(0, str.length() - 1);
            }
            if (str.startsWith("0x")) {
                return Long.valueOf(Long.parseLong(str.substring(2), 16));
            }
            return Long.valueOf(Long.parseLong(str));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Long l) {
            bundle.putLong(str, l.longValue());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class g extends p<long[]> {
        public g(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public long[] a(Bundle bundle, String str) {
            return (long[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "long[]";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public long[] c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, long[] jArr) {
            bundle.putLongArray(str, jArr);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class h extends p<Float> {
        public h(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Float a(Bundle bundle, String str) {
            return (Float) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "float";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Float c(String str) {
            return Float.valueOf(Float.parseFloat(str));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Float f2) {
            bundle.putFloat(str, f2.floatValue());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class i extends p<float[]> {
        public i(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public float[] a(Bundle bundle, String str) {
            return (float[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "float[]";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public float[] c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, float[] fArr) {
            bundle.putFloatArray(str, fArr);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class j extends p<Boolean> {
        public j(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Boolean a(Bundle bundle, String str) {
            return (Boolean) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "boolean";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public Boolean c(String str) {
            if ("true".equals(str)) {
                return Boolean.TRUE;
            }
            if ("false".equals(str)) {
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("A boolean NavType only accepts \"true\" or \"false\" values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Boolean bool) {
            bundle.putBoolean(str, bool.booleanValue());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public class k extends p<boolean[]> {
        public k(boolean z) {
            super(z);
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public boolean[] a(Bundle bundle, String str) {
            return (boolean[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return "boolean[]";
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // b.v.p
        public boolean[] c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.os.Bundle, java.lang.String, java.lang.Object] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, boolean[] zArr) {
            bundle.putBooleanArray(str, zArr);
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public static final class l<D extends Enum> extends C0051p<D> {
        public final Class<D> n;

        public l(Class<D> cls) {
            super(false, cls);
            if (cls.isEnum()) {
                this.n = cls;
                return;
            }
            throw new IllegalArgumentException(cls + " is not an Enum type.");
        }

        @Override // b.v.p.C0051p, b.v.p
        public String b() {
            return this.n.getName();
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // b.v.p.C0051p
        /* renamed from: f */
        public D e(String str) {
            D[] enumConstants;
            for (D d2 : this.n.getEnumConstants()) {
                if (d2.name().equals(str)) {
                    return d2;
                }
            }
            StringBuilder B = c.b.a.a.a.B("Enum value ", str, " not found for type ");
            B.append(this.n.getName());
            B.append(".");
            throw new IllegalArgumentException(B.toString());
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public static final class m<D extends Parcelable> extends p<D[]> {
        public final Class<D[]> m;

        public m(Class<D> cls) {
            super(true);
            if (Parcelable.class.isAssignableFrom(cls)) {
                try {
                    this.m = (Class<D[]>) Class.forName("[L" + cls.getName() + ";");
                    return;
                } catch (ClassNotFoundException e2) {
                    throw new RuntimeException(e2);
                }
            }
            throw new IllegalArgumentException(cls + " does not implement Parcelable.");
        }

        @Override // b.v.p
        public Object a(Bundle bundle, String str) {
            return (Parcelable[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return this.m.getName();
        }

        @Override // b.v.p
        public Object c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        @Override // b.v.p
        public void d(Bundle bundle, String str, Object obj) {
            Parcelable[] parcelableArr = (Parcelable[]) obj;
            this.m.cast(parcelableArr);
            bundle.putParcelableArray(str, parcelableArr);
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || m.class != obj.getClass()) {
                return false;
            }
            return this.m.equals(((m) obj).m);
        }

        public int hashCode() {
            return this.m.hashCode();
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public static final class n<D> extends p<D> {
        public final Class<D> m;

        public n(Class<D> cls) {
            super(true);
            if (!Parcelable.class.isAssignableFrom(cls) && !Serializable.class.isAssignableFrom(cls)) {
                throw new IllegalArgumentException(cls + " does not implement Parcelable or Serializable.");
            }
            this.m = cls;
        }

        @Override // b.v.p
        public D a(Bundle bundle, String str) {
            return (D) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return this.m.getName();
        }

        @Override // b.v.p
        public D c(String str) {
            throw new UnsupportedOperationException("Parcelables don't support default values.");
        }

        @Override // b.v.p
        public void d(Bundle bundle, String str, D d2) {
            this.m.cast(d2);
            if (d2 != null && !(d2 instanceof Parcelable)) {
                if (d2 instanceof Serializable) {
                    bundle.putSerializable(str, (Serializable) d2);
                    return;
                }
                return;
            }
            bundle.putParcelable(str, (Parcelable) d2);
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || n.class != obj.getClass()) {
                return false;
            }
            return this.m.equals(((n) obj).m);
        }

        public int hashCode() {
            return this.m.hashCode();
        }
    }

    /* compiled from: NavType.java */
    /* loaded from: classes.dex */
    public static final class o<D extends Serializable> extends p<D[]> {
        public final Class<D[]> m;

        public o(Class<D> cls) {
            super(true);
            if (Serializable.class.isAssignableFrom(cls)) {
                try {
                    this.m = (Class<D[]>) Class.forName("[L" + cls.getName() + ";");
                    return;
                } catch (ClassNotFoundException e2) {
                    throw new RuntimeException(e2);
                }
            }
            throw new IllegalArgumentException(cls + " does not implement Serializable.");
        }

        @Override // b.v.p
        public Object a(Bundle bundle, String str) {
            return (Serializable[]) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return this.m.getName();
        }

        @Override // b.v.p
        public Object c(String str) {
            throw new UnsupportedOperationException("Arrays don't support default values.");
        }

        /* JADX DEBUG: Multi-variable search result rejected for r2v0, resolved type: android.os.Bundle */
        /* JADX WARN: Multi-variable type inference failed */
        /* JADX WARN: Type inference failed for: r4v1, types: [java.lang.Object, java.io.Serializable[], java.io.Serializable] */
        @Override // b.v.p
        public void d(Bundle bundle, String str, Object obj) {
            ?? r4 = (Serializable[]) obj;
            this.m.cast(r4);
            bundle.putSerializable(str, r4);
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || o.class != obj.getClass()) {
                return false;
            }
            return this.m.equals(((o) obj).m);
        }

        public int hashCode() {
            return this.m.hashCode();
        }
    }

    public p(boolean z) {
        this.l = z;
    }

    public abstract T a(Bundle bundle, String str);

    public abstract String b();

    public abstract T c(String str);

    public abstract void d(Bundle bundle, String str, T t);

    public String toString() {
        return b();
    }

    /* compiled from: NavType.java */
    /* renamed from: b.v.p$p  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0051p<D extends Serializable> extends p<D> {
        public final Class<D> m;

        public C0051p(Class<D> cls) {
            super(true);
            if (Serializable.class.isAssignableFrom(cls)) {
                if (!cls.isEnum()) {
                    this.m = cls;
                    return;
                }
                throw new IllegalArgumentException(cls + " is an Enum. You should use EnumType instead.");
            }
            throw new IllegalArgumentException(cls + " does not implement Serializable.");
        }

        @Override // b.v.p
        public Object a(Bundle bundle, String str) {
            return (Serializable) bundle.get(str);
        }

        @Override // b.v.p
        public String b() {
            return this.m.getName();
        }

        @Override // b.v.p
        public void d(Bundle bundle, String str, Object obj) {
            Serializable serializable = (Serializable) obj;
            this.m.cast(serializable);
            bundle.putSerializable(str, serializable);
        }

        /* JADX DEBUG: Method merged with bridge method */
        @Override // b.v.p
        /* renamed from: e */
        public D c(String str) {
            throw new UnsupportedOperationException("Serializables don't support default values.");
        }

        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj instanceof C0051p) {
                return this.m.equals(((C0051p) obj).m);
            }
            return false;
        }

        public int hashCode() {
            return this.m.hashCode();
        }

        public C0051p(boolean z, Class<D> cls) {
            super(z);
            if (Serializable.class.isAssignableFrom(cls)) {
                this.m = cls;
                return;
            }
            throw new IllegalArgumentException(cls + " does not implement Serializable.");
        }
    }
}