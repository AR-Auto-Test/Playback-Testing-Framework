package c.c.a.m.w;

import android.util.Base64;
import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/* compiled from: DataUrlLoader.java */
/* loaded from: classes.dex */
public final class e<Model, Data> implements n<Model, Data> {

    /* renamed from: a  reason: collision with root package name */
    public final a<Data> f3829a;

    /* compiled from: DataUrlLoader.java */
    /* loaded from: classes.dex */
    public interface a<Data> {
    }

    /* compiled from: DataUrlLoader.java */
    /* loaded from: classes.dex */
    public static final class b<Data> implements c.c.a.m.u.d<Data> {

        /* renamed from: b  reason: collision with root package name */
        public final String f3830b;

        /* renamed from: c  reason: collision with root package name */
        public final a<Data> f3831c;

        /* renamed from: d  reason: collision with root package name */
        public Data f3832d;

        public b(String str, a<Data> aVar) {
            this.f3830b = str;
            this.f3831c = aVar;
        }

        @Override // c.c.a.m.u.d
        public Class<Data> a() {
            Objects.requireNonNull((c.a) this.f3831c);
            return InputStream.class;
        }

        @Override // c.c.a.m.u.d
        public void b() {
            try {
                a<Data> aVar = this.f3831c;
                Data data = this.f3832d;
                Objects.requireNonNull((c.a) aVar);
                ((InputStream) data).close();
            } catch (IOException unused) {
            }
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        /* JADX WARN: Type inference failed for: r2v4, types: [java.lang.Object, Data] */
        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super Data> aVar) {
            try {
                ?? r2 = (Data) ((c.a) this.f3831c).a(this.f3830b);
                this.f3832d = r2;
                aVar.f(r2);
            } catch (IllegalArgumentException e2) {
                aVar.c(e2);
            }
        }
    }

    /* compiled from: DataUrlLoader.java */
    /* loaded from: classes.dex */
    public static final class c<Model> implements o<Model, InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final a<InputStream> f3833a = new a(this);

        /* compiled from: DataUrlLoader.java */
        /* loaded from: classes.dex */
        public class a implements a<InputStream> {
            public a(c cVar) {
            }

            public Object a(String str) {
                if (str.startsWith("data:image")) {
                    int indexOf = str.indexOf(44);
                    if (indexOf != -1) {
                        if (str.substring(0, indexOf).endsWith(";base64")) {
                            return new ByteArrayInputStream(Base64.decode(str.substring(indexOf + 1), 0));
                        }
                        throw new IllegalArgumentException("Not a base64 image data URL.");
                    }
                    throw new IllegalArgumentException("Missing comma in data URL.");
                }
                throw new IllegalArgumentException("Not a valid image data URL.");
            }
        }

        @Override // c.c.a.m.w.o
        public n<Model, InputStream> b(r rVar) {
            return new e(this.f3833a);
        }
    }

    public e(a<Data> aVar) {
        this.f3829a = aVar;
    }

    @Override // c.c.a.m.w.n
    public boolean a(Model model) {
        return model.toString().startsWith("data:image");
    }

    @Override // c.c.a.m.w.n
    public n.a<Data> b(Model model, int i, int i2, c.c.a.m.p pVar) {
        return new n.a<>(new c.c.a.r.d(model), new b(model.toString(), this.f3829a));
    }
}