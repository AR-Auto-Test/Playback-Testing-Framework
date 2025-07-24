package b.j.d;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.CancellationSignal;
import android.util.Log;
import b.j.g.l;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

/* compiled from: TypefaceCompatApi24Impl.java */
/* loaded from: classes.dex */
public class f extends k {

    /* renamed from: b  reason: collision with root package name */
    public static final Class<?> f2110b;

    /* renamed from: c  reason: collision with root package name */
    public static final Constructor<?> f2111c;

    /* renamed from: d  reason: collision with root package name */
    public static final Method f2112d;

    /* renamed from: e  reason: collision with root package name */
    public static final Method f2113e;

    static {
        Class<?> cls;
        Method method;
        Method method2;
        Constructor<?> constructor = null;
        try {
            cls = Class.forName("android.graphics.FontFamily");
            Constructor<?> constructor2 = cls.getConstructor(new Class[0]);
            Class<?> cls2 = Integer.TYPE;
            method2 = cls.getMethod("addFontWeightStyle", ByteBuffer.class, cls2, List.class, cls2, Boolean.TYPE);
            method = Typeface.class.getMethod("createFromFamiliesWithDefault", Array.newInstance(cls, 1).getClass());
            constructor = constructor2;
        } catch (ClassNotFoundException | NoSuchMethodException e2) {
            Log.e("TypefaceCompatApi24Impl", e2.getClass().getName(), e2);
            cls = null;
            method = null;
            method2 = null;
        }
        f2111c = constructor;
        f2110b = cls;
        f2112d = method2;
        f2113e = method;
    }

    public static boolean f(Object obj, ByteBuffer byteBuffer, int i, int i2, boolean z) {
        try {
            return ((Boolean) f2112d.invoke(obj, byteBuffer, Integer.valueOf(i), null, Integer.valueOf(i2), Boolean.valueOf(z))).booleanValue();
        } catch (IllegalAccessException | InvocationTargetException unused) {
            return false;
        }
    }

    public static Typeface g(Object obj) {
        try {
            Object newInstance = Array.newInstance(f2110b, 1);
            Array.set(newInstance, 0, obj);
            return (Typeface) f2113e.invoke(null, newInstance);
        } catch (IllegalAccessException | InvocationTargetException unused) {
            return null;
        }
    }

    @Override // b.j.d.k
    public Typeface a(Context context, b.j.c.b.b bVar, Resources resources, int i) {
        Object obj;
        b.j.c.b.c[] cVarArr;
        MappedByteBuffer mappedByteBuffer;
        try {
            obj = f2111c.newInstance(new Object[0]);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException unused) {
            obj = null;
        }
        if (obj == null) {
            return null;
        }
        for (b.j.c.b.c cVar : bVar.f2076a) {
            int i2 = cVar.f2082f;
            File x = b.j.b.d.x(context);
            if (x != null) {
                try {
                    if (b.j.b.d.o(x, resources, i2)) {
                        try {
                            FileInputStream fileInputStream = new FileInputStream(x);
                            try {
                                FileChannel channel = fileInputStream.getChannel();
                                mappedByteBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0L, channel.size());
                                fileInputStream.close();
                            } finally {
                                break;
                            }
                        } catch (IOException unused2) {
                            mappedByteBuffer = null;
                        }
                        if (mappedByteBuffer != null || !f(obj, mappedByteBuffer, cVar.f2081e, cVar.f2078b, cVar.f2079c)) {
                            return null;
                        }
                    }
                } finally {
                    x.delete();
                }
            }
            mappedByteBuffer = null;
            if (mappedByteBuffer != null) {
                return null;
            }
        }
        return g(obj);
    }

    @Override // b.j.d.k
    public Typeface b(Context context, CancellationSignal cancellationSignal, l[] lVarArr, int i) {
        Object obj;
        try {
            obj = f2111c.newInstance(new Object[0]);
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException unused) {
            obj = null;
        }
        if (obj == null) {
            return null;
        }
        b.f.h hVar = new b.f.h();
        for (l lVar : lVarArr) {
            Uri uri = lVar.f2152a;
            ByteBuffer byteBuffer = (ByteBuffer) hVar.get(uri);
            if (byteBuffer == null) {
                byteBuffer = b.j.b.d.F(context, cancellationSignal, uri);
                hVar.put(uri, byteBuffer);
            }
            if (byteBuffer == null || !f(obj, byteBuffer, lVar.f2153b, lVar.f2154c, lVar.f2155d)) {
                return null;
            }
        }
        Typeface g2 = g(obj);
        if (g2 == null) {
            return null;
        }
        return Typeface.create(g2, i);
    }
}