package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ParcelFileDescriptor;
import com.bumptech.glide.load.ImageHeaderParser;
import com.bumptech.glide.load.data.ParcelFileDescriptorRewinder;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;

/* compiled from: ImageReader.java */
/* loaded from: classes.dex */
public interface s {

    /* compiled from: ImageReader.java */
    /* loaded from: classes.dex */
    public static final class a implements s {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.u.k f3996a;

        /* renamed from: b  reason: collision with root package name */
        public final c.c.a.m.v.c0.b f3997b;

        /* renamed from: c  reason: collision with root package name */
        public final List<ImageHeaderParser> f3998c;

        public a(InputStream inputStream, List<ImageHeaderParser> list, c.c.a.m.v.c0.b bVar) {
            Objects.requireNonNull(bVar, "Argument must not be null");
            this.f3997b = bVar;
            Objects.requireNonNull(list, "Argument must not be null");
            this.f3998c = list;
            this.f3996a = new c.c.a.m.u.k(inputStream, bVar);
        }

        @Override // c.c.a.m.x.c.s
        public int a() {
            return b.v.u.c.i(this.f3998c, this.f3996a.a(), this.f3997b);
        }

        @Override // c.c.a.m.x.c.s
        public Bitmap b(BitmapFactory.Options options) {
            return BitmapFactory.decodeStream(this.f3996a.a(), null, options);
        }

        @Override // c.c.a.m.x.c.s
        public void c() {
            w wVar = this.f3996a.f3569a;
            synchronized (wVar) {
                wVar.f4008d = wVar.f4006b.length;
            }
        }

        @Override // c.c.a.m.x.c.s
        public ImageHeaderParser.ImageType d() {
            return b.v.u.c.k(this.f3998c, this.f3996a.a(), this.f3997b);
        }
    }

    /* compiled from: ImageReader.java */
    /* loaded from: classes.dex */
    public static final class b implements s {

        /* renamed from: a  reason: collision with root package name */
        public final c.c.a.m.v.c0.b f3999a;

        /* renamed from: b  reason: collision with root package name */
        public final List<ImageHeaderParser> f4000b;

        /* renamed from: c  reason: collision with root package name */
        public final ParcelFileDescriptorRewinder f4001c;

        public b(ParcelFileDescriptor parcelFileDescriptor, List<ImageHeaderParser> list, c.c.a.m.v.c0.b bVar) {
            Objects.requireNonNull(bVar, "Argument must not be null");
            this.f3999a = bVar;
            Objects.requireNonNull(list, "Argument must not be null");
            this.f4000b = list;
            this.f4001c = new ParcelFileDescriptorRewinder(parcelFileDescriptor);
        }

        @Override // c.c.a.m.x.c.s
        public int a() {
            return b.v.u.c.j(this.f4000b, new c.c.a.m.j(this.f4001c, this.f3999a));
        }

        @Override // c.c.a.m.x.c.s
        public Bitmap b(BitmapFactory.Options options) {
            return BitmapFactory.decodeFileDescriptor(this.f4001c.a().getFileDescriptor(), null, options);
        }

        @Override // c.c.a.m.x.c.s
        public void c() {
        }

        @Override // c.c.a.m.x.c.s
        public ImageHeaderParser.ImageType d() {
            return b.v.u.c.l(this.f4000b, new c.c.a.m.h(this.f4001c, this.f3999a));
        }
    }

    int a();

    Bitmap b(BitmapFactory.Options options);

    void c();

    ImageHeaderParser.ImageType d();
}