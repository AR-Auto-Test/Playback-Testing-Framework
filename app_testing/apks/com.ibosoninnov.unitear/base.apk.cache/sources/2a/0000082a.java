package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;
import org.opencv.calib3d.Calib3d;

/* compiled from: InputStreamBitmapImageDecoderResourceDecoder.java */
/* loaded from: classes.dex */
public final class t implements c.c.a.m.r<InputStream, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final d f4002a = new d();

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ boolean a(InputStream inputStream, c.c.a.m.p pVar) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(InputStream inputStream, int i, int i2, c.c.a.m.p pVar) {
        InputStream inputStream2 = inputStream;
        AtomicReference<byte[]> atomicReference = c.c.a.s.a.f4173a;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(Calib3d.CALIB_RATIONAL_MODEL);
        byte[] andSet = c.c.a.s.a.f4173a.getAndSet(null);
        if (andSet == null) {
            andSet = new byte[Calib3d.CALIB_RATIONAL_MODEL];
        }
        while (true) {
            int read = inputStream2.read(andSet);
            if (read >= 0) {
                byteArrayOutputStream.write(andSet, 0, read);
            } else {
                c.c.a.s.a.f4173a.set(andSet);
                byte[] byteArray = byteArrayOutputStream.toByteArray();
                return this.f4002a.b(ImageDecoder.createSource((ByteBuffer) ByteBuffer.allocateDirect(byteArray.length).put(byteArray).position(0)), i, i2, pVar);
            }
        }
    }
}