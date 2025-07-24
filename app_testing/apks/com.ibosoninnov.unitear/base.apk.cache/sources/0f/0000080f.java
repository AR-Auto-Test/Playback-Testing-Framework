package c.c.a.m.x.c;

import android.graphics.Bitmap;
import c.c.a.s.a;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: ByteBufferBitmapDecoder.java */
/* loaded from: classes.dex */
public class g implements c.c.a.m.r<ByteBuffer, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final m f3960a;

    public g(m mVar) {
        this.f3960a = mVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(ByteBuffer byteBuffer, c.c.a.m.p pVar) {
        Objects.requireNonNull(this.f3960a);
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(ByteBuffer byteBuffer, int i, int i2, c.c.a.m.p pVar) {
        AtomicReference<byte[]> atomicReference = c.c.a.s.a.f4173a;
        return this.f3960a.b(new a.C0086a(byteBuffer), i, i2, pVar, m.f3981f);
    }
}