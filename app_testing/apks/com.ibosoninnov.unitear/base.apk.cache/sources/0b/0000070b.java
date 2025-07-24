package c.c.a.m;

import com.bumptech.glide.load.ImageHeaderParser;
import java.io.InputStream;

/* compiled from: ImageHeaderParserUtils.java */
/* loaded from: classes.dex */
public class f implements l {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ InputStream f3529a;

    public f(InputStream inputStream) {
        this.f3529a = inputStream;
    }

    @Override // c.c.a.m.l
    public ImageHeaderParser.ImageType a(ImageHeaderParser imageHeaderParser) {
        try {
            return imageHeaderParser.b(this.f3529a);
        } finally {
            this.f3529a.reset();
        }
    }
}