package c.c.a.m;

import c.c.a.m.x.c.w;
import com.bumptech.glide.load.ImageHeaderParser;
import com.bumptech.glide.load.data.ParcelFileDescriptorRewinder;
import java.io.FileInputStream;
import java.io.IOException;

/* compiled from: ImageHeaderParserUtils.java */
/* loaded from: classes.dex */
public class j implements k {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ParcelFileDescriptorRewinder f3535a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ c.c.a.m.v.c0.b f3536b;

    public j(ParcelFileDescriptorRewinder parcelFileDescriptorRewinder, c.c.a.m.v.c0.b bVar) {
        this.f3535a = parcelFileDescriptorRewinder;
        this.f3536b = bVar;
    }

    @Override // c.c.a.m.k
    public int a(ImageHeaderParser imageHeaderParser) {
        w wVar;
        w wVar2 = null;
        try {
            wVar = new w(new FileInputStream(this.f3535a.a().getFileDescriptor()), this.f3536b);
        } catch (Throwable th) {
            th = th;
        }
        try {
            int c2 = imageHeaderParser.c(wVar, this.f3536b);
            try {
                wVar.close();
            } catch (IOException unused) {
            }
            this.f3535a.a();
            return c2;
        } catch (Throwable th2) {
            th = th2;
            wVar2 = wVar;
            if (wVar2 != null) {
                try {
                    wVar2.close();
                } catch (IOException unused2) {
                }
            }
            this.f3535a.a();
            throw th;
        }
    }
}