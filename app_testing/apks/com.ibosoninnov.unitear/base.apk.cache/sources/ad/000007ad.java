package c.c.a.m.w;

import android.util.Log;
import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

/* compiled from: ByteBufferFileLoader.java */
/* loaded from: classes.dex */
public class d implements n<File, ByteBuffer> {

    /* compiled from: ByteBufferFileLoader.java */
    /* loaded from: classes.dex */
    public static final class a implements c.c.a.m.u.d<ByteBuffer> {

        /* renamed from: b  reason: collision with root package name */
        public final File f3828b;

        public a(File file) {
            this.f3828b = file;
        }

        @Override // c.c.a.m.u.d
        public Class<ByteBuffer> a() {
            return ByteBuffer.class;
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

        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super ByteBuffer> aVar) {
            try {
                aVar.f(c.c.a.s.a.a(this.f3828b));
            } catch (IOException e2) {
                if (Log.isLoggable("ByteBufferFileLoader", 3)) {
                    Log.d("ByteBufferFileLoader", "Failed to obtain ByteBuffer for file", e2);
                }
                aVar.c(e2);
            }
        }
    }

    /* compiled from: ByteBufferFileLoader.java */
    /* loaded from: classes.dex */
    public static class b implements o<File, ByteBuffer> {
        @Override // c.c.a.m.w.o
        public n<File, ByteBuffer> b(r rVar) {
            return new d();
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(File file) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.w.n$a' to match base method */
    @Override // c.c.a.m.w.n
    public n.a<ByteBuffer> b(File file, int i, int i2, c.c.a.m.p pVar) {
        File file2 = file;
        return new n.a<>(new c.c.a.r.d(file2), new a(file2));
    }
}