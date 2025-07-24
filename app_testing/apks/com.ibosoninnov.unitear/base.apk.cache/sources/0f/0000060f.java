package b.z;

import android.graphics.Rect;
import android.os.Build;
import android.util.Property;
import android.view.View;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: ViewUtils.java */
/* loaded from: classes.dex */
public class s {

    /* renamed from: a  reason: collision with root package name */
    public static final y f2921a;

    /* renamed from: b  reason: collision with root package name */
    public static final Property<View, Float> f2922b;

    /* compiled from: ViewUtils.java */
    /* loaded from: classes.dex */
    public static class a extends Property<View, Float> {
        public a(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public Float get(View view) {
            return Float.valueOf(s.a(view));
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(View view, Float f2) {
            float floatValue = f2.floatValue();
            s.f2921a.e(view, floatValue);
        }
    }

    /* compiled from: ViewUtils.java */
    /* loaded from: classes.dex */
    public static class b extends Property<View, Rect> {
        public b(Class cls, String str) {
            super(cls, str);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // android.util.Property
        public Rect get(View view) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            return view.getClipBounds();
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // android.util.Property
        public void set(View view, Rect rect) {
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            view.setClipBounds(rect);
        }
    }

    static {
        if (Build.VERSION.SDK_INT >= 29) {
            f2921a = new x();
        } else {
            f2921a = new w();
        }
        f2922b = new a(Float.class, "translationAlpha");
        new b(Rect.class, "clipBounds");
    }

    public static float a(View view) {
        return f2921a.b(view);
    }

    public static void b(View view, int i, int i2, int i3, int i4) {
        f2921a.d(view, i, i2, i3, i4);
    }
}