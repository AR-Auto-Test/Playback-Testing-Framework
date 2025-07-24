package c.a.a;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/* compiled from: LottieCompositionFactory.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public static final Map<String, r<d>> f3059a = new HashMap();

    /* compiled from: LottieCompositionFactory.java */
    /* loaded from: classes.dex */
    public class a implements l<d> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ String f3060a;

        public a(String str) {
            this.f3060a = str;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // c.a.a.l
        public void a(d dVar) {
            e.f3059a.remove(this.f3060a);
        }
    }

    /* compiled from: LottieCompositionFactory.java */
    /* loaded from: classes.dex */
    public class b implements l<Throwable> {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ String f3061a;

        public b(String str) {
            this.f3061a = str;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // c.a.a.l
        public void a(Throwable th) {
            e.f3059a.remove(this.f3061a);
        }
    }

    /* compiled from: LottieCompositionFactory.java */
    /* loaded from: classes.dex */
    public class c implements Callable<p<d>> {

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ d f3062b;

        public c(d dVar) {
            this.f3062b = dVar;
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // java.util.concurrent.Callable
        public p<d> call() {
            return new p<>(this.f3062b);
        }
    }

    public static r<d> a(String str, Callable<p<d>> callable) {
        d dVar;
        if (str == null) {
            dVar = null;
        } else {
            c.a.a.z.g gVar = c.a.a.z.g.f3279a;
            Objects.requireNonNull(gVar);
            dVar = gVar.f3280b.get(str);
        }
        if (dVar != null) {
            return new r<>(new c(dVar));
        }
        if (str != null) {
            Map<String, r<d>> map = f3059a;
            if (map.containsKey(str)) {
                return map.get(str);
            }
        }
        r<d> rVar = new r<>(callable);
        if (str != null) {
            rVar.b(new a(str));
            rVar.a(new b(str));
            f3059a.put(str, rVar);
        }
        return rVar;
    }

    public static p<d> b(InputStream inputStream, String str) {
        try {
            g.s sVar = new g.s(g.o.c(inputStream));
            String[] strArr = c.a.a.b0.h0.c.f2973b;
            return c(new c.a.a.b0.h0.d(sVar), str, true);
        } finally {
            c.a.a.c0.g.b(inputStream);
        }
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[IF]}, finally: {[IF, INVOKE] complete} */
    public static p<d> c(c.a.a.b0.h0.c cVar, String str, boolean z) {
        try {
            try {
                d a2 = c.a.a.b0.s.a(cVar);
                if (str != null) {
                    c.a.a.z.g gVar = c.a.a.z.g.f3279a;
                    Objects.requireNonNull(gVar);
                    gVar.f3280b.put(str, a2);
                }
                p<d> pVar = new p<>(a2);
                if (z) {
                    c.a.a.c0.g.b(cVar);
                }
                return pVar;
            } catch (Exception e2) {
                p<d> pVar2 = new p<>(e2);
                if (z) {
                    c.a.a.c0.g.b(cVar);
                }
                return pVar2;
            }
        } catch (Throwable th) {
            if (z) {
                c.a.a.c0.g.b(cVar);
            }
            throw th;
        }
    }

    public static p<d> d(ZipInputStream zipInputStream, String str) {
        try {
            return e(zipInputStream, str);
        } finally {
            c.a.a.c0.g.b(zipInputStream);
        }
    }

    public static p<d> e(ZipInputStream zipInputStream, String str) {
        k kVar;
        String[] split;
        HashMap hashMap = new HashMap();
        try {
            ZipEntry nextEntry = zipInputStream.getNextEntry();
            d dVar = null;
            while (nextEntry != null) {
                String name = nextEntry.getName();
                if (name.contains("__MACOSX")) {
                    zipInputStream.closeEntry();
                } else if (nextEntry.getName().contains(".json")) {
                    g.s sVar = new g.s(g.o.c(zipInputStream));
                    String[] strArr = c.a.a.b0.h0.c.f2973b;
                    dVar = c(new c.a.a.b0.h0.d(sVar), null, false).f3122a;
                } else {
                    if (!name.contains(".png") && !name.contains(".webp")) {
                        zipInputStream.closeEntry();
                    }
                    hashMap.put(name.split("/")[split.length - 1], BitmapFactory.decodeStream(zipInputStream));
                }
                nextEntry = zipInputStream.getNextEntry();
            }
            if (dVar == null) {
                return new p<>(new IllegalArgumentException("Unable to parse composition"));
            }
            for (Map.Entry entry : hashMap.entrySet()) {
                String str2 = (String) entry.getKey();
                Iterator<k> it = dVar.f3040d.values().iterator();
                while (true) {
                    if (!it.hasNext()) {
                        kVar = null;
                        break;
                    }
                    kVar = it.next();
                    if (kVar.f3112d.equals(str2)) {
                        break;
                    }
                }
                if (kVar != null) {
                    kVar.f3113e = c.a.a.c0.g.e((Bitmap) entry.getValue(), kVar.f3109a, kVar.f3110b);
                }
            }
            for (Map.Entry<String, k> entry2 : dVar.f3040d.entrySet()) {
                if (entry2.getValue().f3113e == null) {
                    StringBuilder x = c.b.a.a.a.x("There is no image for ");
                    x.append(entry2.getValue().f3112d);
                    return new p<>(new IllegalStateException(x.toString()));
                }
            }
            if (str != null) {
                c.a.a.z.g gVar = c.a.a.z.g.f3279a;
                Objects.requireNonNull(gVar);
                gVar.f3280b.put(str, dVar);
            }
            return new p<>(dVar);
        } catch (IOException e2) {
            return new p<>(e2);
        }
    }

    public static String f(Context context, int i) {
        StringBuilder x = c.b.a.a.a.x("rawRes");
        x.append((context.getResources().getConfiguration().uiMode & 48) == 32 ? "_night_" : "_day_");
        x.append(i);
        return x.toString();
    }
}