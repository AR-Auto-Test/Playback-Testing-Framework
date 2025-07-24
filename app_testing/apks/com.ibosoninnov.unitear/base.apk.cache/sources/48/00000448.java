package b.j.d;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Typeface;
import android.os.CancellationSignal;
import b.j.g.l;
import java.io.File;
import java.io.InputStream;
import java.util.concurrent.ConcurrentHashMap;

/* compiled from: TypefaceCompatBaseImpl.java */
/* loaded from: classes.dex */
public class k {
    @SuppressLint({"BanConcurrentHashMap"})

    /* renamed from: a  reason: collision with root package name */
    public ConcurrentHashMap<Long, b.j.c.b.b> f2116a = new ConcurrentHashMap<>();

    /* compiled from: TypefaceCompatBaseImpl.java */
    /* loaded from: classes.dex */
    public interface a<T> {
        int a(T t);

        boolean b(T t);
    }

    public static <T> T e(T[] tArr, int i, a<T> aVar) {
        int i2 = (i & 1) == 0 ? 400 : 700;
        boolean z = (i & 2) != 0;
        T t = null;
        int i3 = Integer.MAX_VALUE;
        for (T t2 : tArr) {
            int abs = (Math.abs(aVar.a(t2) - i2) * 2) + (aVar.b(t2) == z ? 0 : 1);
            if (t == null || i3 > abs) {
                t = t2;
                i3 = abs;
            }
        }
        return t;
    }

    public Typeface a(Context context, b.j.c.b.b bVar, Resources resources, int i) {
        throw null;
    }

    public Typeface b(Context context, CancellationSignal cancellationSignal, l[] lVarArr, int i) {
        throw null;
    }

    public Typeface c(Context context, InputStream inputStream) {
        File x = b.j.b.d.x(context);
        if (x == null) {
            return null;
        }
        try {
            if (b.j.b.d.p(x, inputStream)) {
                return Typeface.createFromFile(x.getPath());
            }
            return null;
        } catch (RuntimeException unused) {
            return null;
        } finally {
            x.delete();
        }
    }

    public Typeface d(Context context, Resources resources, int i, String str, int i2) {
        File x = b.j.b.d.x(context);
        if (x == null) {
            return null;
        }
        try {
            if (b.j.b.d.o(x, resources, i)) {
                return Typeface.createFromFile(x.getPath());
            }
            return null;
        } catch (RuntimeException unused) {
            return null;
        } finally {
            x.delete();
        }
    }
}