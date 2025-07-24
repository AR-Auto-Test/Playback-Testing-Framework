package c.c.a;

import android.app.Activity;
import android.content.ComponentCallbacks2;
import android.content.ContentResolver;
import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.text.TextUtils;
import android.util.Log;
import c.c.a.c;
import c.c.a.e;
import c.c.a.m.r;
import c.c.a.m.u.k;
import c.c.a.m.v.d0.j;
import c.c.a.m.v.e0.a;
import c.c.a.m.v.l;
import c.c.a.m.w.a;
import c.c.a.m.w.b;
import c.c.a.m.w.d;
import c.c.a.m.w.e;
import c.c.a.m.w.f;
import c.c.a.m.w.k;
import c.c.a.m.w.s;
import c.c.a.m.w.u;
import c.c.a.m.w.v;
import c.c.a.m.w.w;
import c.c.a.m.w.x;
import c.c.a.m.w.y.a;
import c.c.a.m.w.y.b;
import c.c.a.m.w.y.c;
import c.c.a.m.w.y.d;
import c.c.a.m.w.y.e;
import c.c.a.m.x.c.b0;
import c.c.a.m.x.c.c0;
import c.c.a.m.x.c.k;
import c.c.a.m.x.c.m;
import c.c.a.m.x.c.t;
import c.c.a.m.x.c.v;
import c.c.a.m.x.c.x;
import c.c.a.m.x.c.z;
import c.c.a.m.x.d.a;
import c.c.a.n.p;
import com.bumptech.glide.GeneratedAppGlideModule;
import com.bumptech.glide.load.ImageHeaderParser;
import com.bumptech.glide.load.data.ParcelFileDescriptorRewinder;
import java.io.File;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/* compiled from: Glide.java */
/* loaded from: classes.dex */
public class b implements ComponentCallbacks2 {

    /* renamed from: b  reason: collision with root package name */
    public static volatile b f3410b;

    /* renamed from: c  reason: collision with root package name */
    public static volatile boolean f3411c;

    /* renamed from: d  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f3412d;

    /* renamed from: e  reason: collision with root package name */
    public final c.c.a.m.v.d0.i f3413e;

    /* renamed from: f  reason: collision with root package name */
    public final d f3414f;

    /* renamed from: g  reason: collision with root package name */
    public final g f3415g;

    /* renamed from: h  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f3416h;
    public final p i;
    public final c.c.a.n.d j;
    public final List<i> k = new ArrayList();

    /* compiled from: Glide.java */
    /* loaded from: classes.dex */
    public interface a {
    }

    public b(Context context, l lVar, c.c.a.m.v.d0.i iVar, c.c.a.m.v.c0.d dVar, c.c.a.m.v.c0.b bVar, p pVar, c.c.a.n.d dVar2, int i, a aVar, Map<Class<?>, j<?, ?>> map, List<c.c.a.q.e<Object>> list, e eVar) {
        r gVar;
        r zVar;
        this.f3412d = dVar;
        this.f3416h = bVar;
        this.f3413e = iVar;
        this.i = pVar;
        this.j = dVar2;
        Resources resources = context.getResources();
        g gVar2 = new g();
        this.f3415g = gVar2;
        k kVar = new k();
        c.c.a.p.b bVar2 = gVar2.f3446g;
        synchronized (bVar2) {
            bVar2.f4112a.add(kVar);
        }
        int i2 = Build.VERSION.SDK_INT;
        if (i2 >= 27) {
            c.c.a.m.x.c.p pVar2 = new c.c.a.m.x.c.p();
            c.c.a.p.b bVar3 = gVar2.f3446g;
            synchronized (bVar3) {
                bVar3.f4112a.add(pVar2);
            }
        }
        List<ImageHeaderParser> e2 = gVar2.e();
        c.c.a.m.x.g.a aVar2 = new c.c.a.m.x.g.a(context, e2, dVar, bVar);
        c0 c0Var = new c0(dVar, new c0.g());
        m mVar = new m(gVar2.e(), resources.getDisplayMetrics(), dVar, bVar);
        if (eVar.f3433a.containsKey(c.b.class) && i2 >= 28) {
            zVar = new t();
            gVar = new c.c.a.m.x.c.h();
        } else {
            gVar = new c.c.a.m.x.c.g(mVar);
            zVar = new z(mVar, bVar);
        }
        c.c.a.m.x.e.d dVar3 = new c.c.a.m.x.e.d(context);
        s.c cVar = new s.c(resources);
        s.d dVar4 = new s.d(resources);
        s.b bVar4 = new s.b(resources);
        s.a aVar3 = new s.a(resources);
        c.c.a.m.x.c.c cVar2 = new c.c.a.m.x.c.c(bVar);
        c.c.a.m.x.h.a aVar4 = new c.c.a.m.x.h.a();
        c.c.a.m.x.h.d dVar5 = new c.c.a.m.x.h.d();
        ContentResolver contentResolver = context.getContentResolver();
        gVar2.a(ByteBuffer.class, new c.c.a.m.w.c());
        gVar2.a(InputStream.class, new c.c.a.m.w.t(bVar));
        gVar2.d("Bitmap", ByteBuffer.class, Bitmap.class, gVar);
        gVar2.d("Bitmap", InputStream.class, Bitmap.class, zVar);
        gVar2.d("Bitmap", ParcelFileDescriptor.class, Bitmap.class, new v(mVar));
        gVar2.d("Bitmap", ParcelFileDescriptor.class, Bitmap.class, c0Var);
        gVar2.d("Bitmap", AssetFileDescriptor.class, Bitmap.class, new c0(dVar, new c0.c(null)));
        v.a<?> aVar5 = v.a.f3897a;
        gVar2.c(Bitmap.class, Bitmap.class, aVar5);
        gVar2.d("Bitmap", Bitmap.class, Bitmap.class, new b0());
        gVar2.b(Bitmap.class, cVar2);
        gVar2.d("BitmapDrawable", ByteBuffer.class, BitmapDrawable.class, new c.c.a.m.x.c.a(resources, gVar));
        gVar2.d("BitmapDrawable", InputStream.class, BitmapDrawable.class, new c.c.a.m.x.c.a(resources, zVar));
        gVar2.d("BitmapDrawable", ParcelFileDescriptor.class, BitmapDrawable.class, new c.c.a.m.x.c.a(resources, c0Var));
        gVar2.b(BitmapDrawable.class, new c.c.a.m.x.c.b(dVar, cVar2));
        gVar2.d("Gif", InputStream.class, c.c.a.m.x.g.c.class, new c.c.a.m.x.g.j(e2, aVar2, bVar));
        gVar2.d("Gif", ByteBuffer.class, c.c.a.m.x.g.c.class, aVar2);
        gVar2.b(c.c.a.m.x.g.c.class, new c.c.a.m.x.g.d());
        gVar2.c(c.c.a.l.a.class, c.c.a.l.a.class, aVar5);
        gVar2.d("Bitmap", c.c.a.l.a.class, Bitmap.class, new c.c.a.m.x.g.h(dVar));
        gVar2.d("legacy_append", Uri.class, Drawable.class, dVar3);
        gVar2.d("legacy_append", Uri.class, Bitmap.class, new x(dVar3, dVar));
        gVar2.g(new a.C0081a());
        gVar2.c(File.class, ByteBuffer.class, new d.b());
        gVar2.c(File.class, InputStream.class, new f.e());
        gVar2.d("legacy_append", File.class, File.class, new c.c.a.m.x.f.a());
        gVar2.c(File.class, ParcelFileDescriptor.class, new f.b());
        gVar2.c(File.class, File.class, aVar5);
        gVar2.g(new k.a(bVar));
        gVar2.g(new ParcelFileDescriptorRewinder.a());
        Class cls = Integer.TYPE;
        gVar2.c(cls, InputStream.class, cVar);
        gVar2.c(cls, ParcelFileDescriptor.class, bVar4);
        gVar2.c(Integer.class, InputStream.class, cVar);
        gVar2.c(Integer.class, ParcelFileDescriptor.class, bVar4);
        gVar2.c(Integer.class, Uri.class, dVar4);
        gVar2.c(cls, AssetFileDescriptor.class, aVar3);
        gVar2.c(Integer.class, AssetFileDescriptor.class, aVar3);
        gVar2.c(cls, Uri.class, dVar4);
        gVar2.c(String.class, InputStream.class, new e.c());
        gVar2.c(Uri.class, InputStream.class, new e.c());
        gVar2.c(String.class, InputStream.class, new u.c());
        gVar2.c(String.class, ParcelFileDescriptor.class, new u.b());
        gVar2.c(String.class, AssetFileDescriptor.class, new u.a());
        gVar2.c(Uri.class, InputStream.class, new a.c(context.getAssets()));
        gVar2.c(Uri.class, ParcelFileDescriptor.class, new a.b(context.getAssets()));
        gVar2.c(Uri.class, InputStream.class, new b.a(context));
        gVar2.c(Uri.class, InputStream.class, new c.a(context));
        if (i2 >= 29) {
            gVar2.c(Uri.class, InputStream.class, new d.c(context));
            gVar2.c(Uri.class, ParcelFileDescriptor.class, new d.b(context));
        }
        gVar2.c(Uri.class, InputStream.class, new w.d(contentResolver));
        gVar2.c(Uri.class, ParcelFileDescriptor.class, new w.b(contentResolver));
        gVar2.c(Uri.class, AssetFileDescriptor.class, new w.a(contentResolver));
        gVar2.c(Uri.class, InputStream.class, new x.a());
        gVar2.c(URL.class, InputStream.class, new e.a());
        gVar2.c(Uri.class, File.class, new k.a(context));
        gVar2.c(c.c.a.m.w.g.class, InputStream.class, new a.C0077a());
        gVar2.c(byte[].class, ByteBuffer.class, new b.a());
        gVar2.c(byte[].class, InputStream.class, new b.d());
        gVar2.c(Uri.class, Uri.class, aVar5);
        gVar2.c(Drawable.class, Drawable.class, aVar5);
        gVar2.d("legacy_append", Drawable.class, Drawable.class, new c.c.a.m.x.e.e());
        gVar2.h(Bitmap.class, BitmapDrawable.class, new c.c.a.m.x.h.b(resources));
        gVar2.h(Bitmap.class, byte[].class, aVar4);
        gVar2.h(Drawable.class, byte[].class, new c.c.a.m.x.h.c(dVar, aVar4, dVar5));
        gVar2.h(c.c.a.m.x.g.c.class, byte[].class, dVar5);
        c0 c0Var2 = new c0(dVar, new c0.d());
        gVar2.d("legacy_append", ByteBuffer.class, Bitmap.class, c0Var2);
        gVar2.d("legacy_append", ByteBuffer.class, BitmapDrawable.class, new c.c.a.m.x.c.a(resources, c0Var2));
        this.f3414f = new d(context, bVar, gVar2, new c.c.a.q.j.f(), aVar, map, list, lVar, eVar, i);
    }

    public static void a(Context context, GeneratedAppGlideModule generatedAppGlideModule) {
        if (!f3411c) {
            f3411c = true;
            c cVar = new c();
            Context applicationContext = context.getApplicationContext();
            Collections.emptyList();
            if (Log.isLoggable("ManifestParser", 3)) {
                Log.d("ManifestParser", "Loading Glide modules");
            }
            ArrayList arrayList = new ArrayList();
            try {
                ApplicationInfo applicationInfo = applicationContext.getPackageManager().getApplicationInfo(applicationContext.getPackageName(), 128);
                if (applicationInfo.metaData == null) {
                    if (Log.isLoggable("ManifestParser", 3)) {
                        Log.d("ManifestParser", "Got null app info metadata");
                    }
                } else {
                    if (Log.isLoggable("ManifestParser", 2)) {
                        Log.v("ManifestParser", "Got app info metadata: " + applicationInfo.metaData);
                    }
                    for (String str : applicationInfo.metaData.keySet()) {
                        if ("GlideModule".equals(applicationInfo.metaData.get(str))) {
                            arrayList.add(c.c.a.o.e.a(str));
                            if (Log.isLoggable("ManifestParser", 3)) {
                                Log.d("ManifestParser", "Loaded Glide module: " + str);
                            }
                        }
                    }
                    if (Log.isLoggable("ManifestParser", 3)) {
                        Log.d("ManifestParser", "Finished loading Glide modules");
                    }
                }
                if (generatedAppGlideModule != null && !generatedAppGlideModule.c().isEmpty()) {
                    Set<Class<?>> c2 = generatedAppGlideModule.c();
                    Iterator it = arrayList.iterator();
                    while (it.hasNext()) {
                        c.c.a.o.c cVar2 = (c.c.a.o.c) it.next();
                        if (c2.contains(cVar2.getClass())) {
                            if (Log.isLoggable("Glide", 3)) {
                                Log.d("Glide", "AppGlideModule excludes manifest GlideModule: " + cVar2);
                            }
                            it.remove();
                        }
                    }
                }
                if (Log.isLoggable("Glide", 3)) {
                    Iterator it2 = arrayList.iterator();
                    while (it2.hasNext()) {
                        StringBuilder x = c.b.a.a.a.x("Discovered GlideModule from manifest: ");
                        x.append(((c.c.a.o.c) it2.next()).getClass());
                        Log.d("Glide", x.toString());
                    }
                }
                cVar.n = null;
                Iterator it3 = arrayList.iterator();
                while (it3.hasNext()) {
                    ((c.c.a.o.c) it3.next()).a(applicationContext, cVar);
                }
                if (cVar.f3423g == null) {
                    int a2 = c.c.a.m.v.e0.a.a();
                    if (!TextUtils.isEmpty("source")) {
                        cVar.f3423g = new c.c.a.m.v.e0.a(new ThreadPoolExecutor(a2, a2, 0L, TimeUnit.MILLISECONDS, new PriorityBlockingQueue(), new a.ThreadFactoryC0069a("source", a.b.f3688b, false)));
                    } else {
                        throw new IllegalArgumentException(c.b.a.a.a.q("Name must be non-null and non-empty, but given: ", "source"));
                    }
                }
                if (cVar.f3424h == null) {
                    int i = c.c.a.m.v.e0.a.f3681c;
                    if (!TextUtils.isEmpty("disk-cache")) {
                        cVar.f3424h = new c.c.a.m.v.e0.a(new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new PriorityBlockingQueue(), new a.ThreadFactoryC0069a("disk-cache", a.b.f3688b, true)));
                    } else {
                        throw new IllegalArgumentException(c.b.a.a.a.q("Name must be non-null and non-empty, but given: ", "disk-cache"));
                    }
                }
                if (cVar.o == null) {
                    int i2 = c.c.a.m.v.e0.a.a() >= 4 ? 2 : 1;
                    if (!TextUtils.isEmpty("animation")) {
                        cVar.o = new c.c.a.m.v.e0.a(new ThreadPoolExecutor(i2, i2, 0L, TimeUnit.MILLISECONDS, new PriorityBlockingQueue(), new a.ThreadFactoryC0069a("animation", a.b.f3688b, true)));
                    } else {
                        throw new IllegalArgumentException(c.b.a.a.a.q("Name must be non-null and non-empty, but given: ", "animation"));
                    }
                }
                if (cVar.j == null) {
                    cVar.j = new c.c.a.m.v.d0.j(new j.a(applicationContext));
                }
                if (cVar.k == null) {
                    cVar.k = new c.c.a.n.f();
                }
                if (cVar.f3420d == null) {
                    int i3 = cVar.j.f3664a;
                    if (i3 > 0) {
                        cVar.f3420d = new c.c.a.m.v.c0.j(i3);
                    } else {
                        cVar.f3420d = new c.c.a.m.v.c0.e();
                    }
                }
                if (cVar.f3421e == null) {
                    cVar.f3421e = new c.c.a.m.v.c0.i(cVar.j.f3667d);
                }
                if (cVar.f3422f == null) {
                    cVar.f3422f = new c.c.a.m.v.d0.h(cVar.j.f3665b);
                }
                if (cVar.i == null) {
                    cVar.i = new c.c.a.m.v.d0.g(applicationContext);
                }
                if (cVar.f3419c == null) {
                    cVar.f3419c = new l(cVar.f3422f, cVar.i, cVar.f3424h, cVar.f3423g, new c.c.a.m.v.e0.a(new ThreadPoolExecutor(0, Integer.MAX_VALUE, c.c.a.m.v.e0.a.f3680b, TimeUnit.MILLISECONDS, new SynchronousQueue(), new a.ThreadFactoryC0069a("source-unlimited", a.b.f3688b, false))), cVar.o, false);
                }
                List<c.c.a.q.e<Object>> list = cVar.p;
                if (list == null) {
                    cVar.p = Collections.emptyList();
                } else {
                    cVar.p = Collections.unmodifiableList(list);
                }
                e.a aVar = cVar.f3418b;
                Objects.requireNonNull(aVar);
                e eVar = new e(aVar);
                b bVar = new b(applicationContext, cVar.f3419c, cVar.f3422f, cVar.f3420d, cVar.f3421e, new p(cVar.n, eVar), cVar.k, cVar.l, cVar.m, cVar.f3417a, cVar.p, eVar);
                Iterator it4 = arrayList.iterator();
                while (it4.hasNext()) {
                    c.c.a.o.c cVar3 = (c.c.a.o.c) it4.next();
                    try {
                        cVar3.b(applicationContext, bVar, bVar.f3415g);
                    } catch (AbstractMethodError e2) {
                        StringBuilder x2 = c.b.a.a.a.x("Attempting to register a Glide v3 module. If you see this, you or one of your dependencies may be including Glide v3 even though you're using Glide v4. You'll need to find and remove (or update) the offending dependency. The v3 module name is: ");
                        x2.append(cVar3.getClass().getName());
                        throw new IllegalStateException(x2.toString(), e2);
                    }
                }
                applicationContext.registerComponentCallbacks(bVar);
                f3410b = bVar;
                f3411c = false;
                return;
            } catch (PackageManager.NameNotFoundException e3) {
                throw new RuntimeException("Unable to find metadata to parse GlideModules", e3);
            }
        }
        throw new IllegalStateException("You cannot call Glide.get() in registerComponents(), use the provided Glide instance instead");
    }

    public static b b(Context context) {
        if (f3410b == null) {
            GeneratedAppGlideModule generatedAppGlideModule = null;
            try {
                generatedAppGlideModule = (GeneratedAppGlideModule) Class.forName("com.bumptech.glide.GeneratedAppGlideModuleImpl").getDeclaredConstructor(Context.class).newInstance(context.getApplicationContext().getApplicationContext());
            } catch (ClassNotFoundException unused) {
                if (Log.isLoggable("Glide", 5)) {
                    Log.w("Glide", "Failed to find GeneratedAppGlideModule. You should include an annotationProcessor compile dependency on com.github.bumptech.glide:compiler in your application and a @GlideModule annotated AppGlideModule implementation or LibraryGlideModules will be silently ignored");
                }
            } catch (IllegalAccessException e2) {
                c(e2);
                throw null;
            } catch (InstantiationException e3) {
                c(e3);
                throw null;
            } catch (NoSuchMethodException e4) {
                c(e4);
                throw null;
            } catch (InvocationTargetException e5) {
                c(e5);
                throw null;
            }
            synchronized (b.class) {
                if (f3410b == null) {
                    a(context, generatedAppGlideModule);
                }
            }
        }
        return f3410b;
    }

    public static void c(Exception exc) {
        throw new IllegalStateException("GeneratedAppGlideModuleImpl is implemented incorrectly. If you've manually implemented this class, remove your implementation. The Annotation processor will generate a correct implementation.", exc);
    }

    public static i d(Activity activity) {
        Objects.requireNonNull(activity, "You cannot start a load on a not yet attached View or a Fragment where getActivity() returns null (which usually occurs when getActivity() is called before the Fragment is attached or after the Fragment is destroyed).");
        return b(activity).i.b(activity);
    }

    public static i e(Context context) {
        Objects.requireNonNull(context, "You cannot start a load on a not yet attached View or a Fragment where getActivity() returns null (which usually occurs when getActivity() is called before the Fragment is attached or after the Fragment is destroyed).");
        return b(context).i.c(context);
    }

    @Override // android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
    }

    @Override // android.content.ComponentCallbacks
    public void onLowMemory() {
        c.c.a.s.j.a();
        ((c.c.a.s.g) this.f3413e).e(0L);
        this.f3412d.b();
        this.f3416h.b();
    }

    @Override // android.content.ComponentCallbacks2
    public void onTrimMemory(int i) {
        long j;
        c.c.a.s.j.a();
        synchronized (this.k) {
            for (i iVar : this.k) {
                Objects.requireNonNull(iVar);
            }
        }
        c.c.a.m.v.d0.h hVar = (c.c.a.m.v.d0.h) this.f3413e;
        Objects.requireNonNull(hVar);
        if (i >= 40) {
            hVar.e(0L);
        } else if (i >= 20 || i == 15) {
            synchronized (hVar) {
                j = hVar.f4189b;
            }
            hVar.e(j / 2);
        }
        this.f3412d.a(i);
        this.f3416h.a(i);
    }
}