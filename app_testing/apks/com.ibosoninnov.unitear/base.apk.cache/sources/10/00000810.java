package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import java.nio.ByteBuffer;

/* compiled from: ByteBufferBitmapImageDecoderResourceDecoder.java */
/* loaded from: classes.dex */
public final class h implements c.c.a.m.r<ByteBuffer, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final d f3961a = new d();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ boolean a(ByteBuffer byteBuffer, c.c.a.m.p pVar) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(ByteBuffer byteBuffer, int i, int i2, c.c.a.m.p pVar) {
        return this.f3961a.b(ImageDecoder.createSource(byteBuffer), i, i2, pVar);
    }
}