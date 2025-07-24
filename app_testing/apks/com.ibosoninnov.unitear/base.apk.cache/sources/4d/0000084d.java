package c.c.a.m.x.h;

import android.graphics.Bitmap;
import c.c.a.m.p;
import c.c.a.m.v.w;
import java.io.ByteArrayOutputStream;

/* compiled from: BitmapBytesTranscoder.java */
/* loaded from: classes.dex */
public class a implements e<Bitmap, byte[]> {

    /* renamed from: a  reason: collision with root package name */
    public final Bitmap.CompressFormat f4064a = Bitmap.CompressFormat.JPEG;

    /* renamed from: b  reason: collision with root package name */
    public final int f4065b = 100;

    @Override // c.c.a.m.x.h.e
    public w<byte[]> a(w<Bitmap> wVar, p pVar) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        wVar.get().compress(this.f4064a, this.f4065b, byteArrayOutputStream);
        wVar.a();
        return new c.c.a.m.x.d.b(byteArrayOutputStream.toByteArray());
    }
}