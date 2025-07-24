package c.c.a.m.x;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.graphics.ImageDecoder;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import c.c.a.m.o;
import c.c.a.m.p;
import c.c.a.m.q;
import c.c.a.m.r;
import c.c.a.m.v.w;
import c.c.a.m.x.c.d;
import c.c.a.m.x.c.e;
import c.c.a.m.x.c.l;
import c.c.a.m.x.c.m;

/* compiled from: ImageDecoderResourceDecoder.java */
/* loaded from: classes.dex */
public abstract class a<T> implements r<ImageDecoder.Source, T> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.x.c.r f3927a = c.c.a.m.x.c.r.a();

    /* compiled from: ImageDecoderResourceDecoder.java */
    /* renamed from: c.c.a.m.x.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0079a implements ImageDecoder.OnHeaderDecodedListener {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f3928a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f3929b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ boolean f3930c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ c.c.a.m.b f3931d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ l f3932e;

        /* renamed from: f  reason: collision with root package name */
        public final /* synthetic */ q f3933f;

        /* compiled from: ImageDecoderResourceDecoder.java */
        /* renamed from: c.c.a.m.x.a$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0080a implements ImageDecoder.OnPartialImageListener {
            public C0080a(C0079a c0079a) {
            }

            @Override // android.graphics.ImageDecoder.OnPartialImageListener
            public boolean onPartialImage(ImageDecoder.DecodeException decodeException) {
                return false;
            }
        }

        public C0079a(int i, int i2, boolean z, c.c.a.m.b bVar, l lVar, q qVar) {
            this.f3928a = i;
            this.f3929b = i2;
            this.f3930c = z;
            this.f3931d = bVar;
            this.f3932e = lVar;
            this.f3933f = qVar;
        }

        @Override // android.graphics.ImageDecoder.OnHeaderDecodedListener
        @SuppressLint({"Override"})
        public void onHeaderDecoded(ImageDecoder imageDecoder, ImageDecoder.ImageInfo imageInfo, ImageDecoder.Source source) {
            boolean z = false;
            if (a.this.f3927a.b(this.f3928a, this.f3929b, this.f3930c, false)) {
                imageDecoder.setAllocator(3);
            } else {
                imageDecoder.setAllocator(1);
            }
            if (this.f3931d == c.c.a.m.b.PREFER_RGB_565) {
                imageDecoder.setMemorySizePolicy(0);
            }
            imageDecoder.setOnPartialImageListener(new C0080a(this));
            Size size = imageInfo.getSize();
            int i = this.f3928a;
            if (i == Integer.MIN_VALUE) {
                i = size.getWidth();
            }
            int i2 = this.f3929b;
            if (i2 == Integer.MIN_VALUE) {
                i2 = size.getHeight();
            }
            float b2 = this.f3932e.b(size.getWidth(), size.getHeight(), i, i2);
            int round = Math.round(size.getWidth() * b2);
            int round2 = Math.round(size.getHeight() * b2);
            if (Log.isLoggable("ImageDecoder", 2)) {
                StringBuilder x = c.b.a.a.a.x("Resizing from [");
                x.append(size.getWidth());
                x.append("x");
                x.append(size.getHeight());
                x.append("] to [");
                x.append(round);
                x.append("x");
                x.append(round2);
                x.append("] scaleFactor: ");
                x.append(b2);
                Log.v("ImageDecoder", x.toString());
            }
            imageDecoder.setTargetSize(round, round2);
            int i3 = Build.VERSION.SDK_INT;
            if (i3 < 28) {
                if (i3 >= 26) {
                    imageDecoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
                    return;
                }
                return;
            }
            if (this.f3933f == q.DISPLAY_P3 && imageInfo.getColorSpace() != null && imageInfo.getColorSpace().isWideGamut()) {
                z = true;
            }
            imageDecoder.setTargetColorSpace(ColorSpace.get(z ? ColorSpace.Named.DISPLAY_P3 : ColorSpace.Named.SRGB));
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ boolean a(ImageDecoder.Source source, p pVar) {
        return true;
    }

    /* JADX DEBUG: Method merged with bridge method */
    @Override // c.c.a.m.r
    /* renamed from: c */
    public final w<T> b(ImageDecoder.Source source, int i, int i2, p pVar) {
        c.c.a.m.b bVar = (c.c.a.m.b) pVar.c(m.f3976a);
        l lVar = (l) pVar.c(l.f3974f);
        o<Boolean> oVar = m.f3979d;
        boolean z = pVar.c(oVar) != null && ((Boolean) pVar.c(oVar)).booleanValue();
        d dVar = (d) this;
        Bitmap decodeBitmap = ImageDecoder.decodeBitmap(source, new C0079a(i, i2, z, bVar, lVar, (q) pVar.c(m.f3977b)));
        if (Log.isLoggable("BitmapImageDecoder", 2)) {
            StringBuilder x = c.b.a.a.a.x("Decoded [");
            x.append(decodeBitmap.getWidth());
            x.append("x");
            x.append(decodeBitmap.getHeight());
            x.append("] for [");
            x.append(i);
            x.append("x");
            x.append(i2);
            x.append("]");
            Log.v("BitmapImageDecoder", x.toString());
        }
        return new e(decodeBitmap, dVar.f3956b);
    }
}