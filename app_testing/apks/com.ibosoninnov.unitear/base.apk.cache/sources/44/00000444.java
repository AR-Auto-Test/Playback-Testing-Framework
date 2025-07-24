package b.j.d;

import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Typeface;
import android.graphics.fonts.FontVariationAxis;
import android.net.Uri;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import b.j.g.l;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/* compiled from: TypefaceCompatApi26Impl.java */
/* loaded from: classes.dex */
public class g extends e {

    /* renamed from: g  reason: collision with root package name */
    public final Class<?> f2114g;

    /* renamed from: h  reason: collision with root package name */
    public final Constructor<?> f2115h;
    public final Method i;
    public final Method j;
    public final Method k;
    public final Method l;
    public final Method m;

    public g() {
        Method method;
        Method method2;
        Constructor<?> constructor;
        Method method3;
        Method method4;
        Method method5;
        Class<?> cls = null;
        try {
            Class<?> cls2 = Class.forName("android.graphics.FontFamily");
            constructor = cls2.getConstructor(new Class[0]);
            method3 = n(cls2);
            method4 = o(cls2);
            method5 = cls2.getMethod("freeze", new Class[0]);
            method2 = cls2.getMethod("abortCreation", new Class[0]);
            method = p(cls2);
            cls = cls2;
        } catch (ClassNotFoundException | NoSuchMethodException e2) {
            StringBuilder x = c.b.a.a.a.x("Unable to collect necessary methods for class ");
            x.append(e2.getClass().getName());
            Log.e("TypefaceCompatApi26Impl", x.toString(), e2);
            method = null;
            method2 = null;
            constructor = null;
            method3 = null;
            method4 = null;
            method5 = null;
        }
        this.f2114g = cls;
        this.f2115h = constructor;
        this.i = method3;
        this.j = method4;
        this.k = method5;
        this.l = method2;
        this.m = method;
    }

    private Object m() {
        try {
            return this.f2115h.newInstance(new Object[0]);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException unused) {
            return null;
        }
    }

    @Override // b.j.d.e, b.j.d.k
    public Typeface a(Context context, b.j.c.b.b bVar, Resources resources, int i) {
        b.j.c.b.c[] cVarArr;
        if (!l()) {
            return super.a(context, bVar, resources, i);
        }
        Object m = m();
        if (m == null) {
            return null;
        }
        for (b.j.c.b.c cVar : bVar.f2076a) {
            if (!i(context, m, cVar.f2077a, cVar.f2081e, cVar.f2078b, cVar.f2079c ? 1 : 0, FontVariationAxis.fromFontVariationSettings(cVar.f2080d))) {
                h(m);
                return null;
            }
        }
        if (k(m)) {
            return j(m);
        }
        return null;
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:71:0x00cb */
    @Override // b.j.d.e, b.j.d.k
    public Typeface b(Context context, CancellationSignal cancellationSignal, l[] lVarArr, int i) {
        Typeface j;
        boolean z;
        if (lVarArr.length < 1) {
            return null;
        }
        if (!l()) {
            l lVar = (l) k.e(lVarArr, i, new j(this));
            try {
                ParcelFileDescriptor openFileDescriptor = context.getContentResolver().openFileDescriptor(lVar.f2152a, "r", cancellationSignal);
                if (openFileDescriptor == null) {
                    if (openFileDescriptor != null) {
                        openFileDescriptor.close();
                    }
                    return null;
                }
                Typeface build = new Typeface.Builder(openFileDescriptor.getFileDescriptor()).setWeight(lVar.f2154c).setItalic(lVar.f2155d).build();
                openFileDescriptor.close();
                return build;
            } catch (IOException unused) {
                return null;
            }
        }
        HashMap hashMap = new HashMap();
        for (l lVar2 : lVarArr) {
            if (lVar2.f2156e == 0) {
                Uri uri = lVar2.f2152a;
                if (!hashMap.containsKey(uri)) {
                    hashMap.put(uri, b.j.b.d.F(context, cancellationSignal, uri));
                }
            }
        }
        Map unmodifiableMap = Collections.unmodifiableMap(hashMap);
        Object m = m();
        if (m == null) {
            return null;
        }
        int length = lVarArr.length;
        int i2 = 0;
        boolean z2 = false;
        while (i2 < length) {
            l lVar3 = lVarArr[i2];
            ByteBuffer byteBuffer = (ByteBuffer) unmodifiableMap.get(lVar3.f2152a);
            if (byteBuffer != null) {
                try {
                    z = ((Boolean) this.j.invoke(m, byteBuffer, Integer.valueOf(lVar3.f2153b), null, Integer.valueOf(lVar3.f2154c), Integer.valueOf(lVar3.f2155d ? 1 : 0))).booleanValue();
                } catch (IllegalAccessException | InvocationTargetException unused2) {
                    z = false;
                }
                if (!z) {
                    h(m);
                    return null;
                }
                z2 = true;
            }
            i2++;
            z2 = z2;
        }
        if (!z2) {
            h(m);
            return null;
        } else if (k(m) && (j = j(m)) != null) {
            return Typeface.create(j, i);
        } else {
            return null;
        }
    }

    @Override // b.j.d.k
    public Typeface d(Context context, Resources resources, int i, String str, int i2) {
        if (!l()) {
            return super.d(context, resources, i, str, i2);
        }
        Object m = m();
        if (m == null) {
            return null;
        }
        if (!i(context, m, str, 0, -1, -1, null)) {
            h(m);
            return null;
        } else if (k(m)) {
            return j(m);
        } else {
            return null;
        }
    }

    public final void h(Object obj) {
        try {
            this.l.invoke(obj, new Object[0]);
        } catch (IllegalAccessException | InvocationTargetException unused) {
        }
    }

    public final boolean i(Context context, Object obj, String str, int i, int i2, int i3, FontVariationAxis[] fontVariationAxisArr) {
        try {
            return ((Boolean) this.i.invoke(obj, context.getAssets(), str, 0, Boolean.FALSE, Integer.valueOf(i), Integer.valueOf(i2), Integer.valueOf(i3), fontVariationAxisArr)).booleanValue();
        } catch (IllegalAccessException | InvocationTargetException unused) {
            return false;
        }
    }

    public Typeface j(Object obj) {
        try {
            Object newInstance = Array.newInstance(this.f2114g, 1);
            Array.set(newInstance, 0, obj);
            return (Typeface) this.m.invoke(null, newInstance, -1, -1);
        } catch (IllegalAccessException | InvocationTargetException unused) {
            return null;
        }
    }

    public final boolean k(Object obj) {
        try {
            return ((Boolean) this.k.invoke(obj, new Object[0])).booleanValue();
        } catch (IllegalAccessException | InvocationTargetException unused) {
            return false;
        }
    }

    public final boolean l() {
        if (this.i == null) {
            Log.w("TypefaceCompatApi26Impl", "Unable to collect necessary private methods. Fallback to legacy implementation.");
        }
        return this.i != null;
    }

    public Method n(Class<?> cls) {
        Class<?> cls2 = Integer.TYPE;
        return cls.getMethod("addFontFromAssetManager", AssetManager.class, String.class, cls2, Boolean.TYPE, cls2, cls2, cls2, FontVariationAxis[].class);
    }

    public Method o(Class<?> cls) {
        Class<?> cls2 = Integer.TYPE;
        return cls.getMethod("addFontFromBuffer", ByteBuffer.class, cls2, FontVariationAxis[].class, cls2, cls2);
    }

    public Method p(Class<?> cls) {
        Class cls2 = Integer.TYPE;
        Method declaredMethod = Typeface.class.getDeclaredMethod("createFromFamiliesWithDefault", Array.newInstance(cls, 1).getClass(), cls2, cls2);
        declaredMethod.setAccessible(true);
        return declaredMethod;
    }
}