package b.j.d;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Typeface;
import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;
import android.util.Log;
import b.j.g.l;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/* compiled from: TypefaceCompatApi21Impl.java */
/* loaded from: classes.dex */
public class e extends k {

    /* renamed from: b  reason: collision with root package name */
    public static Class<?> f2105b = null;

    /* renamed from: c  reason: collision with root package name */
    public static Constructor<?> f2106c = null;

    /* renamed from: d  reason: collision with root package name */
    public static Method f2107d = null;

    /* renamed from: e  reason: collision with root package name */
    public static Method f2108e = null;

    /* renamed from: f  reason: collision with root package name */
    public static boolean f2109f = false;

    public static boolean f(Object obj, String str, int i, boolean z) {
        g();
        try {
            return ((Boolean) f2107d.invoke(obj, str, Integer.valueOf(i), Boolean.valueOf(z))).booleanValue();
        } catch (IllegalAccessException | InvocationTargetException e2) {
            throw new RuntimeException(e2);
        }
    }

    public static void g() {
        Method method;
        Class<?> cls;
        Method method2;
        if (f2109f) {
            return;
        }
        f2109f = true;
        Constructor<?> constructor = null;
        try {
            cls = Class.forName("android.graphics.FontFamily");
            Constructor<?> constructor2 = cls.getConstructor(new Class[0]);
            method2 = cls.getMethod("addFontWeightStyle", String.class, Integer.TYPE, Boolean.TYPE);
            method = Typeface.class.getMethod("createFromFamiliesWithDefault", Array.newInstance(cls, 1).getClass());
            constructor = constructor2;
        } catch (ClassNotFoundException | NoSuchMethodException e2) {
            Log.e("TypefaceCompatApi21Impl", e2.getClass().getName(), e2);
            method = null;
            cls = null;
            method2 = null;
        }
        f2106c = constructor;
        f2105b = cls;
        f2107d = method2;
        f2108e = method;
    }

    @Override // b.j.d.k
    public Typeface a(Context context, b.j.c.b.b bVar, Resources resources, int i) {
        b.j.c.b.c[] cVarArr;
        g();
        try {
            Object newInstance = f2106c.newInstance(new Object[0]);
            for (b.j.c.b.c cVar : bVar.f2076a) {
                File x = b.j.b.d.x(context);
                if (x == null) {
                    return null;
                }
                try {
                    if (!b.j.b.d.o(x, resources, cVar.f2082f)) {
                        return null;
                    }
                    if (!f(newInstance, x.getPath(), cVar.f2078b, cVar.f2079c)) {
                        return null;
                    }
                    x.delete();
                } catch (RuntimeException unused) {
                    return null;
                } finally {
                    x.delete();
                }
            }
            g();
            try {
                Object newInstance2 = Array.newInstance(f2105b, 1);
                Array.set(newInstance2, 0, newInstance);
                return (Typeface) f2108e.invoke(null, newInstance2);
            } catch (IllegalAccessException | InvocationTargetException e2) {
                throw new RuntimeException(e2);
            }
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException e3) {
            throw new RuntimeException(e3);
        }
    }

    @Override // b.j.d.k
    public Typeface b(Context context, CancellationSignal cancellationSignal, l[] lVarArr, int i) {
        File file;
        String readlink;
        if (lVarArr.length < 1) {
            return null;
        }
        try {
            ParcelFileDescriptor openFileDescriptor = context.getContentResolver().openFileDescriptor(((l) k.e(lVarArr, i, new j(this))).f2152a, "r", cancellationSignal);
            if (openFileDescriptor == null) {
                if (openFileDescriptor != null) {
                    openFileDescriptor.close();
                }
                return null;
            }
            try {
                readlink = Os.readlink("/proc/self/fd/" + openFileDescriptor.getFd());
            } catch (ErrnoException unused) {
            }
            if (OsConstants.S_ISREG(Os.stat(readlink).st_mode)) {
                file = new File(readlink);
                if (file != null && file.canRead()) {
                    Typeface createFromFile = Typeface.createFromFile(file);
                    openFileDescriptor.close();
                    return createFromFile;
                }
                FileInputStream fileInputStream = new FileInputStream(openFileDescriptor.getFileDescriptor());
                Typeface c2 = c(context, fileInputStream);
                fileInputStream.close();
                openFileDescriptor.close();
                return c2;
            }
            file = null;
            if (file != null) {
                Typeface createFromFile2 = Typeface.createFromFile(file);
                openFileDescriptor.close();
                return createFromFile2;
            }
            FileInputStream fileInputStream2 = new FileInputStream(openFileDescriptor.getFileDescriptor());
            Typeface c22 = c(context, fileInputStream2);
            fileInputStream2.close();
            openFileDescriptor.close();
            return c22;
        } catch (IOException unused2) {
            return null;
        }
    }
}