package c.c.a.m;

import com.bumptech.glide.load.ImageHeaderParser;
import java.io.InputStream;

/* compiled from: ImageHeaderParserUtils.java */
/* loaded from: classes.dex */
public class i implements k {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ InputStream f3533a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ c.c.a.m.v.c0.b f3534b;

    public i(InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        this.f3533a = inputStream;
        this.f3534b = bVar;
    }

    @Override // c.c.a.m.k
    public int a(ImageHeaderParser imageHeaderParser) {
        try {
            return imageHeaderParser.c(this.f3533a, this.f3534b);
        } finally {
            this.f3533a.reset();
        }
    }
}