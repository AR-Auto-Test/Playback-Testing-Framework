package c.a.a;

import android.content.Context;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.Callable;
import java.util.zip.ZipInputStream;

/* compiled from: LottieCompositionFactory.java */
/* loaded from: classes.dex */
public class f implements Callable<p<d>> {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Context f3063b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ String f3064c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ String f3065d;

    public f(Context context, String str, String str2) {
        this.f3063b = context;
        this.f3064c = str;
        this.f3065d = str2;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    /* JADX WARN: Removed duplicated region for block: B:23:0x0077  */
    @Override // java.util.concurrent.Callable
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public p<d> call() {
        b.j.i.c cVar;
        p<d> b2;
        File file;
        c.a.a.a0.a aVar;
        c.a.a.a0.c cVar2 = new c.a.a.a0.c(this.f3063b, this.f3064c, this.f3065d);
        c.a.a.a0.a aVar2 = c.a.a.a0.a.ZIP;
        c.a.a.a0.b bVar = cVar2.f2953c;
        d dVar = null;
        if (bVar != null) {
            String str = cVar2.f2952b;
            try {
                File b3 = bVar.b();
                aVar = c.a.a.a0.a.JSON;
                file = new File(b3, c.a.a.a0.b.a(str, aVar, false));
                if (!file.exists()) {
                    file = new File(bVar.b(), c.a.a.a0.b.a(str, aVar2, false));
                    if (!file.exists()) {
                        file = null;
                    }
                }
            } catch (FileNotFoundException unused) {
            }
            if (file == null) {
                cVar = null;
                if (cVar != null) {
                    c.a.a.a0.a aVar3 = (c.a.a.a0.a) cVar.f2192a;
                    InputStream inputStream = (InputStream) cVar.f2193b;
                    if (aVar3 == aVar2) {
                        b2 = e.d(new ZipInputStream(inputStream), cVar2.f2952b);
                    } else {
                        b2 = e.b(inputStream, cVar2.f2952b);
                    }
                    d dVar2 = b2.f3122a;
                    if (dVar2 != null) {
                        dVar = dVar2;
                    }
                }
            } else {
                FileInputStream fileInputStream = new FileInputStream(file);
                if (file.getAbsolutePath().endsWith(".zip")) {
                    aVar = aVar2;
                }
                StringBuilder B = c.b.a.a.a.B("Cache hit for ", str, " at ");
                B.append(file.getAbsolutePath());
                c.a.a.c0.c.a(B.toString());
                cVar = new b.j.i.c(aVar, fileInputStream);
                if (cVar != null) {
                }
            }
        }
        if (dVar != null) {
            return new p<>(dVar);
        }
        StringBuilder x = c.b.a.a.a.x("Animation for ");
        x.append(cVar2.f2952b);
        x.append(" not found in cache. Fetching from network.");
        c.a.a.c0.c.a(x.toString());
        try {
            return cVar2.a();
        } catch (IOException e2) {
            return new p<>(e2);
        }
    }
}