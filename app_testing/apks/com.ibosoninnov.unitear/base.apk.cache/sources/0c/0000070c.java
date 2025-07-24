package c.c.a.m;

import com.bumptech.glide.load.ImageHeaderParser;
import java.nio.ByteBuffer;

/* compiled from: ImageHeaderParserUtils.java */
/* loaded from: classes.dex */
public class g implements l {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ ByteBuffer f3530a;

    public g(ByteBuffer byteBuffer) {
        this.f3530a = byteBuffer;
    }

    @Override // c.c.a.m.l
    public ImageHeaderParser.ImageType a(ImageHeaderParser imageHeaderParser) {
        return imageHeaderParser.a(this.f3530a);
    }
}