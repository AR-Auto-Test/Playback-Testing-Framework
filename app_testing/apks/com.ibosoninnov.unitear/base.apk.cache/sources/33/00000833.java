package c.c.a.m.x.d;

import c.c.a.m.u.e;
import java.nio.ByteBuffer;

/* compiled from: ByteBufferRewinder.java */
/* loaded from: classes.dex */
public class a implements e<ByteBuffer> {

    /* renamed from: a  reason: collision with root package name */
    public final ByteBuffer f4020a;

    /* compiled from: ByteBufferRewinder.java */
    /* renamed from: c.c.a.m.x.d.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0081a implements e.a<ByteBuffer> {
        @Override // c.c.a.m.u.e.a
        public Class<ByteBuffer> a() {
            return ByteBuffer.class;
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        /* JADX DEBUG: Return type fixed from 'c.c.a.m.u.e' to match base method */
        @Override // c.c.a.m.u.e.a
        public e<ByteBuffer> b(ByteBuffer byteBuffer) {
            return new a(byteBuffer);
        }
    }

    public a(ByteBuffer byteBuffer) {
        this.f4020a = byteBuffer;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.u.e
    public ByteBuffer a() {
        this.f4020a.position(0);
        return this.f4020a;
    }

    @Override // c.c.a.m.u.e
    public void b() {
    }
}