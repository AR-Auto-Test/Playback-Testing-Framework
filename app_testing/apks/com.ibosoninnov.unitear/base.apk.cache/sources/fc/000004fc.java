package b.o.a;

import android.util.FloatProperty;

/* compiled from: FloatPropertyCompat.java */
/* loaded from: classes.dex */
public abstract class c<T> {
    public final String mPropertyName;

    /* compiled from: FloatPropertyCompat.java */
    /* loaded from: classes.dex */
    public static class a extends c<T> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ FloatProperty f2361a;

        /* JADX WARN: 'super' call moved to the top of the method (can break code semantics) */
        public a(String str, FloatProperty floatProperty) {
            super(str);
            this.f2361a = floatProperty;
        }

        @Override // b.o.a.c
        public float getValue(T t) {
            return ((Float) this.f2361a.get(t)).floatValue();
        }

        @Override // b.o.a.c
        public void setValue(T t, float f2) {
            this.f2361a.setValue(t, f2);
        }
    }

    public c(String str) {
        this.mPropertyName = str;
    }

    public static <T> c<T> createFloatPropertyCompat(FloatProperty<T> floatProperty) {
        return new a(floatProperty.getName(), floatProperty);
    }

    public abstract float getValue(T t);

    public abstract void setValue(T t, float f2);
}