package c.c.a.m.x.h;

import c.c.a.m.p;
import c.c.a.m.v.w;
import c.c.a.s.a;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;

/* compiled from: GifDrawableBytesTranscoder.java */
/* loaded from: classes.dex */
public class d implements e<c.c.a.m.x.g.c, byte[]> {
    @Override // c.c.a.m.x.h.e
    public w<byte[]> a(w<c.c.a.m.x.g.c> wVar, p pVar) {
        byte[] bArr;
        ByteBuffer asReadOnlyBuffer = wVar.get().f4036b.f4043a.f4045a.e().asReadOnlyBuffer();
        AtomicReference<byte[]> atomicReference = c.c.a.s.a.f4173a;
        a.b bVar = (asReadOnlyBuffer.isReadOnly() || !asReadOnlyBuffer.hasArray()) ? null : new a.b(asReadOnlyBuffer.array(), asReadOnlyBuffer.arrayOffset(), asReadOnlyBuffer.limit());
        if (bVar != null && bVar.f4176a == 0 && bVar.f4177b == bVar.f4178c.length) {
            bArr = asReadOnlyBuffer.array();
        } else {
            ByteBuffer asReadOnlyBuffer2 = asReadOnlyBuffer.asReadOnlyBuffer();
            byte[] bArr2 = new byte[asReadOnlyBuffer2.limit()];
            asReadOnlyBuffer2.position(0);
            asReadOnlyBuffer2.get(bArr2);
            bArr = bArr2;
        }
        return new c.c.a.m.x.d.b(bArr);
    }
}