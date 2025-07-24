package c.c.a.m;

import c.c.a.m.x.c.w;
import com.bumptech.glide.load.ImageHeaderParser;
import com.bumptech.glide.load.data.ParcelFileDescriptorRewinder;
import java.io.FileInputStream;
import java.io.IOException;

/* compiled from: ImageHeaderParserUtils.java */
/* loaded from: classes.dex */
public class h implements l {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ParcelFileDescriptorRewinder f3531a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ c.c.a.m.v.c0.b f3532b;

    public h(ParcelFileDescriptorRewinder parcelFileDescriptorRewinder, c.c.a.m.v.c0.b bVar) {
        this.f3531a = parcelFileDescriptorRewinder;
        this.f3532b = bVar;
    }

    @Override // c.c.a.m.l
    public ImageHeaderParser.ImageType a(ImageHeaderParser imageHeaderParser) {
        w wVar;
        w wVar2 = null;
        try {
            wVar = new w(new FileInputStream(this.f3531a.a().getFileDescriptor()), this.f3532b);
        } catch (Throwable th) {
            th = th;
        }
        try {
            ImageHeaderParser.ImageType b2 = imageHeaderParser.b(wVar);
            try {
                wVar.close();
            } catch (IOException unused) {
            }
            this.f3531a.a();
            return b2;
        } catch (Throwable th2) {
            th = th2;
            wVar2 = wVar;
            if (wVar2 != null) {
                try {
                    wVar2.close();
                } catch (IOException unused2) {
                }
            }
            this.f3531a.a();
            throw th;
        }
    }
}