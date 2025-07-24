package c.c.a.m.x.g;

import android.util.Log;
import c.c.a.m.p;
import c.c.a.m.r;
import c.c.a.m.v.w;
import com.bumptech.glide.load.ImageHeaderParser;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;
import org.opencv.calib3d.Calib3d;

/* compiled from: StreamGifDecoder.java */
/* loaded from: classes.dex */
public class j implements r<InputStream, c> {

    /* renamed from: a  reason: collision with root package name */
    public final List<ImageHeaderParser> f4061a;

    /* renamed from: b  reason: collision with root package name */
    public final r<ByteBuffer, c> f4062b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.v.c0.b f4063c;

    public j(List<ImageHeaderParser> list, r<ByteBuffer, c> rVar, c.c.a.m.v.c0.b bVar) {
        this.f4061a = list;
        this.f4062b = rVar;
        this.f4063c = bVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(InputStream inputStream, p pVar) {
        return !((Boolean) pVar.c(i.f4060b)).booleanValue() && b.v.u.c.k(this.f4061a, inputStream, this.f4063c) == ImageHeaderParser.ImageType.GIF;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public w<c> b(InputStream inputStream, int i, int i2, p pVar) {
        byte[] bArr;
        InputStream inputStream2 = inputStream;
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(Calib3d.CALIB_RATIONAL_MODEL);
        try {
            byte[] bArr2 = new byte[Calib3d.CALIB_RATIONAL_MODEL];
            while (true) {
                int read = inputStream2.read(bArr2);
                if (read == -1) {
                    break;
                }
                byteArrayOutputStream.write(bArr2, 0, read);
            }
            byteArrayOutputStream.flush();
            bArr = byteArrayOutputStream.toByteArray();
        } catch (IOException e2) {
            if (Log.isLoggable("StreamGifDecoder", 5)) {
                Log.w("StreamGifDecoder", "Error reading data from stream", e2);
            }
            bArr = null;
        }
        if (bArr == null) {
            return null;
        }
        return this.f4062b.b(ByteBuffer.wrap(bArr), i, i2, pVar);
    }
}