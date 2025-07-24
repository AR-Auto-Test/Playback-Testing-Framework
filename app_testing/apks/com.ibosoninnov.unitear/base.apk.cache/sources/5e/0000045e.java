package b.j.g;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/* compiled from: FontRequestWorker.java */
/* loaded from: classes.dex */
public class j {

    /* renamed from: a  reason: collision with root package name */
    public static final b.f.f<String, Typeface> f2144a = new b.f.f<>(16);

    /* renamed from: b  reason: collision with root package name */
    public static final ExecutorService f2145b;

    /* renamed from: c  reason: collision with root package name */
    public static final Object f2146c;

    /* renamed from: d  reason: collision with root package name */
    public static final b.f.h<String, ArrayList<b.j.i.a<a>>> f2147d;

    static {
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(0, 1, 10000, TimeUnit.MILLISECONDS, new LinkedBlockingDeque(), new n("fonts-androidx", 10));
        threadPoolExecutor.allowCoreThreadTimeOut(true);
        f2145b = threadPoolExecutor;
        f2146c = new Object();
        f2147d = new b.f.h<>();
    }

    public static a a(String str, Context context, e eVar, int i) {
        int i2;
        Typeface typeface = f2144a.get(str);
        if (typeface != null) {
            return new a(typeface);
        }
        try {
            k a2 = d.a(context, eVar, null);
            int i3 = a2.f2150a;
            int i4 = 1;
            if (i3 != 0) {
                if (i3 == 1) {
                    i2 = -2;
                }
                i2 = -3;
            } else {
                l[] lVarArr = a2.f2151b;
                if (lVarArr != null && lVarArr.length != 0) {
                    for (l lVar : lVarArr) {
                        int i5 = lVar.f2156e;
                        if (i5 != 0) {
                            if (i5 >= 0) {
                                i2 = i5;
                            }
                            i2 = -3;
                        }
                    }
                    i4 = 0;
                }
                i2 = i4;
            }
            if (i2 != 0) {
                return new a(i2);
            }
            Typeface b2 = b.j.d.d.f2102a.b(context, null, a2.f2151b, i);
            if (b2 != null) {
                f2144a.put(str, b2);
                return new a(b2);
            }
            return new a(-3);
        } catch (PackageManager.NameNotFoundException unused) {
            return new a(-1);
        }
    }

    /* compiled from: FontRequestWorker.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final Typeface f2148a;

        /* renamed from: b  reason: collision with root package name */
        public final int f2149b;

        public a(int i) {
            this.f2148a = null;
            this.f2149b = i;
        }

        @SuppressLint({"WrongConstant"})
        public a(Typeface typeface) {
            this.f2148a = typeface;
            this.f2149b = 0;
        }
    }
}