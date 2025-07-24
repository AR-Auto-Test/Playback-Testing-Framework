package c.c.a.m.w;

import android.os.ParcelFileDescriptor;
import android.util.Log;
import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

/* compiled from: FileLoader.java */
/* loaded from: classes.dex */
public class f<Data> implements n<File, Data> {

    /* renamed from: a  reason: collision with root package name */
    public final d<Data> f3834a;

    /* compiled from: FileLoader.java */
    /* loaded from: classes.dex */
    public static class a<Data> implements o<File, Data> {

        /* renamed from: a  reason: collision with root package name */
        public final d<Data> f3835a;

        public a(d<Data> dVar) {
            this.f3835a = dVar;
        }

        @Override // c.c.a.m.w.o
        public final n<File, Data> b(r rVar) {
            return new f(this.f3835a);
        }
    }

    /* compiled from: FileLoader.java */
    /* loaded from: classes.dex */
    public static class b extends a<ParcelFileDescriptor> {

        /* compiled from: FileLoader.java */
        /* loaded from: classes.dex */
        public class a implements d<ParcelFileDescriptor> {
            @Override // c.c.a.m.w.f.d
            public Class<ParcelFileDescriptor> a() {
                return ParcelFileDescriptor.class;
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.m.w.f.d
            public ParcelFileDescriptor b(File file) {
                return ParcelFileDescriptor.open(file, 268435456);
            }

            /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
            @Override // c.c.a.m.w.f.d
            public void c(ParcelFileDescriptor parcelFileDescriptor) {
                parcelFileDescriptor.close();
            }
        }

        public b() {
            super(new a());
        }
    }

    /* compiled from: FileLoader.java */
    /* loaded from: classes.dex */
    public static final class c<Data> implements c.c.a.m.u.d<Data> {

        /* renamed from: b  reason: collision with root package name */
        public final File f3836b;

        /* renamed from: c  reason: collision with root package name */
        public final d<Data> f3837c;

        /* renamed from: d  reason: collision with root package name */
        public Data f3838d;

        public c(File file, d<Data> dVar) {
            this.f3836b = file;
            this.f3837c = dVar;
        }

        @Override // c.c.a.m.u.d
        public Class<Data> a() {
            return this.f3837c.a();
        }

        @Override // c.c.a.m.u.d
        public void b() {
            Data data = this.f3838d;
            if (data != null) {
                try {
                    this.f3837c.c(data);
                } catch (IOException unused) {
                }
            }
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        /* JADX WARN: Type inference failed for: r3v3, types: [java.lang.Object, Data] */
        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super Data> aVar) {
            try {
                Data b2 = this.f3837c.b(this.f3836b);
                this.f3838d = b2;
                aVar.f(b2);
            } catch (FileNotFoundException e2) {
                if (Log.isLoggable("FileLoader", 3)) {
                    Log.d("FileLoader", "Failed to open file", e2);
                }
                aVar.c(e2);
            }
        }
    }

    /* compiled from: FileLoader.java */
    /* loaded from: classes.dex */
    public interface d<Data> {
        Class<Data> a();

        Data b(File file);

        void c(Data data);
    }

    /* compiled from: FileLoader.java */
    /* loaded from: classes.dex */
    public static class e extends a<InputStream> {

        /* compiled from: FileLoader.java */
        /* loaded from: classes.dex */
        public class a implements d<InputStream> {
            @Override // c.c.a.m.w.f.d
            public Class<InputStream> a() {
                return InputStream.class;
            }

            /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
            @Override // c.c.a.m.w.f.d
            public InputStream b(File file) {
                return new FileInputStream(file);
            }

            /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
            @Override // c.c.a.m.w.f.d
            public void c(InputStream inputStream) {
                inputStream.close();
            }
        }

        public e() {
            super(new a());
        }
    }

    public f(d<Data> dVar) {
        this.f3834a = dVar;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public /* bridge */ /* synthetic */ boolean a(File file) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(File file, int i, int i2, c.c.a.m.p pVar) {
        File file2 = file;
        return new n.a(new c.c.a.r.d(file2), new c(file2, this.f3834a));
    }
}