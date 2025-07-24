package b.j.d;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import b.j.g.j;
import b.j.g.m;
import b.j.g.o;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/* compiled from: TypefaceCompat.java */
@SuppressLint({"NewApi"})
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public static final k f2102a;

    /* renamed from: b  reason: collision with root package name */
    public static final b.f.f<String, Typeface> f2103b;

    /* compiled from: TypefaceCompat.java */
    /* loaded from: classes.dex */
    public static class a extends m {

        /* renamed from: a  reason: collision with root package name */
        public b.j.c.b.e f2104a;

        public a(b.j.c.b.e eVar) {
            this.f2104a = eVar;
        }
    }

    static {
        int i = Build.VERSION.SDK_INT;
        if (i >= 29) {
            f2102a = new i();
        } else if (i >= 28) {
            f2102a = new h();
        } else if (i >= 26) {
            f2102a = new g();
        } else {
            Method method = f.f2112d;
            if (method == null) {
                Log.w("TypefaceCompatApi24Impl", "Unable to collect necessary private methods.Fallback to legacy implementation.");
            }
            if (method != null) {
                f2102a = new f();
            } else {
                f2102a = new e();
            }
        }
        f2103b = new b.f.f<>(16);
    }

    /* JADX WARN: Code restructure failed: missing block: B:12:0x0024, code lost:
        if (r0.equals(r4) == false) goto L11;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Typeface a(Context context, b.j.c.b.a aVar, Resources resources, int i, int i2, b.j.c.b.e eVar, Handler handler, boolean z) {
        Typeface a2;
        Typeface typeface;
        Handler handler2;
        if (aVar instanceof b.j.c.b.d) {
            b.j.c.b.d dVar = (b.j.c.b.d) aVar;
            String str = dVar.f2086d;
            a2 = null;
            boolean z2 = false;
            if (str != null && !str.isEmpty()) {
                typeface = Typeface.create(str, 0);
                Typeface create = Typeface.create(Typeface.DEFAULT, 0);
                if (typeface != null) {
                }
            }
            typeface = null;
            if (typeface != null) {
                if (eVar != null) {
                    eVar.callbackSuccessAsync(typeface, handler);
                }
                return typeface;
            }
            if (!z ? eVar == null : dVar.f2085c == 0) {
                z2 = true;
            }
            int i3 = z ? dVar.f2084b : -1;
            Handler handler3 = b.j.c.b.e.getHandler(handler);
            a aVar2 = new a(eVar);
            b.j.g.e eVar2 = dVar.f2083a;
            b.j.g.c cVar = new b.j.g.c(aVar2, handler3);
            if (z2) {
                b.f.f<String, Typeface> fVar = b.j.g.j.f2144a;
                String str2 = eVar2.f2133e + "-" + i2;
                Typeface typeface2 = b.j.g.j.f2144a.get(str2);
                if (typeface2 != null) {
                    handler3.post(new b.j.g.a(cVar, aVar2, typeface2));
                    a2 = typeface2;
                } else if (i3 == -1) {
                    j.a a3 = b.j.g.j.a(str2, context, eVar2, i2);
                    cVar.a(a3);
                    a2 = a3.f2148a;
                } else {
                    try {
                        try {
                            try {
                                j.a aVar3 = (j.a) b.j.g.j.f2145b.submit(new b.j.g.f(str2, context, eVar2, i2)).get(i3, TimeUnit.MILLISECONDS);
                                cVar.a(aVar3);
                                a2 = aVar3.f2148a;
                            } catch (InterruptedException e2) {
                                throw e2;
                            } catch (TimeoutException unused) {
                                throw new InterruptedException("timeout");
                            }
                        } catch (ExecutionException e3) {
                            throw new RuntimeException(e3);
                        }
                    } catch (InterruptedException unused2) {
                        cVar.f2127b.post(new b.j.g.b(cVar, cVar.f2126a, -3));
                    }
                }
            } else {
                b.f.f<String, Typeface> fVar2 = b.j.g.j.f2144a;
                String str3 = eVar2.f2133e + "-" + i2;
                Typeface typeface3 = b.j.g.j.f2144a.get(str3);
                if (typeface3 != null) {
                    handler3.post(new b.j.g.a(cVar, aVar2, typeface3));
                    a2 = typeface3;
                } else {
                    b.j.g.g gVar = new b.j.g.g(cVar);
                    synchronized (b.j.g.j.f2146c) {
                        b.f.h<String, ArrayList<b.j.i.a<j.a>>> hVar = b.j.g.j.f2147d;
                        ArrayList<b.j.i.a<j.a>> arrayList = hVar.get(str3);
                        if (arrayList != null) {
                            arrayList.add(gVar);
                        } else {
                            ArrayList<b.j.i.a<j.a>> arrayList2 = new ArrayList<>();
                            arrayList2.add(gVar);
                            hVar.put(str3, arrayList2);
                            b.j.g.h hVar2 = new b.j.g.h(str3, context, eVar2, i2);
                            ExecutorService executorService = b.j.g.j.f2145b;
                            b.j.g.i iVar = new b.j.g.i(str3);
                            if (Looper.myLooper() == null) {
                                handler2 = new Handler(Looper.getMainLooper());
                            } else {
                                handler2 = new Handler();
                            }
                            executorService.execute(new o(handler2, hVar2, iVar));
                        }
                    }
                }
            }
        } else {
            a2 = f2102a.a(context, (b.j.c.b.b) aVar, resources, i2);
            if (eVar != null) {
                if (a2 != null) {
                    eVar.callbackSuccessAsync(a2, handler);
                } else {
                    eVar.callbackFailAsync(-3, handler);
                }
            }
        }
        if (a2 != null) {
            f2103b.put(c(resources, i, i2), a2);
        }
        return a2;
    }

    public static Typeface b(Context context, Resources resources, int i, String str, int i2) {
        Typeface d2 = f2102a.d(context, resources, i, str, i2);
        if (d2 != null) {
            f2103b.put(c(resources, i, i2), d2);
        }
        return d2;
    }

    public static String c(Resources resources, int i, int i2) {
        return resources.getResourcePackageName(i) + "-" + i + "-" + i2;
    }
}