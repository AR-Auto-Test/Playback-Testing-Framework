package c.c.a.m.w;

import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

/* compiled from: ByteArrayLoader.java */
/* loaded from: classes.dex */
public class b<Data> implements n<byte[], Data> {

    /* renamed from: a  reason: collision with root package name */
    public final InterfaceC0075b<Data> f3825a;

    /* compiled from: ByteArrayLoader.java */
    /* loaded from: classes.dex */
    public static class a implements o<byte[], ByteBuffer> {

        /* compiled from: ByteArrayLoader.java */
        /* renamed from: c.c.a.m.w.b$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0074a implements InterfaceC0075b<ByteBuffer> {
            public C0074a(a aVar) {
            }

            @Override // c.c.a.m.w.b.InterfaceC0075b
            public Class<ByteBuffer> a() {
                return ByteBuffer.class;
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.m.w.b.InterfaceC0075b
            public ByteBuffer b(byte[] bArr) {
                return ByteBuffer.wrap(bArr);
            }
        }

        @Override // c.c.a.m.w.o
        public n<byte[], ByteBuffer> b(r rVar) {
            return new b(new C0074a(this));
        }
    }

    /* compiled from: ByteArrayLoader.java */
    /* renamed from: c.c.a.m.w.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0075b<Data> {
        Class<Data> a();

        Data b(byte[] bArr);
    }

    /* compiled from: ByteArrayLoader.java */
    /* loaded from: classes.dex */
    public static class c<Data> implements c.c.a.m.u.d<Data> {

        /* renamed from: b  reason: collision with root package name */
        public final byte[] f3826b;

        /* renamed from: c  reason: collision with root package name */
        public final InterfaceC0075b<Data> f3827c;

        public c(byte[] bArr, InterfaceC0075b<Data> interfaceC0075b) {
            this.f3826b = bArr;
            this.f3827c = interfaceC0075b;
        }

        @Override // c.c.a.m.u.d
        public Class<Data> a() {
            return this.f3827c.a();
        }

        @Override // c.c.a.m.u.d
        public void b() {
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        /* JADX DEBUG: Type inference failed for r2v2. Raw type applied. Possible types: Data, ? super Data */
        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super Data> aVar) {
            aVar.f((Data) this.f3827c.b(this.f3826b));
        }
    }

    /* compiled from: ByteArrayLoader.java */
    /* loaded from: classes.dex */
    public static class d implements o<byte[], InputStream> {

        /* compiled from: ByteArrayLoader.java */
        /* loaded from: classes.dex */
        public class a implements InterfaceC0075b<InputStream> {
            public a(d dVar) {
            }

            @Override // c.c.a.m.w.b.InterfaceC0075b
            public Class<InputStream> a() {
                return InputStream.class;
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.m.w.b.InterfaceC0075b
            public InputStream b(byte[] bArr) {
                return new ByteArrayInputStream(bArr);
            }
        }

        @Override // c.c.a.m.w.o
        public n<byte[], InputStream> b(r rVar) {
            return new b(new a(this));
        }
    }

    public b(InterfaceC0075b<Data> interfaceC0075b) {
        this.f3825a = interfaceC0075b;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(byte[] bArr) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(byte[] bArr, int i, int i2, c.c.a.m.p pVar) {
        byte[] bArr2 = bArr;
        return new n.a(new c.c.a.r.d(bArr2), new c(bArr2, this.f3825a));
    }
}