package c.c.a.m.x.c;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import c.c.a.m.o;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.Objects;

/* compiled from: VideoDecoder.java */
/* loaded from: classes.dex */
public class c0<T> implements c.c.a.m.r<T, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public static final c.c.a.m.o<Long> f3948a = new c.c.a.m.o<>("com.bumptech.glide.load.resource.bitmap.VideoBitmapDecode.TargetFrame", -1L, new a());

    /* renamed from: b  reason: collision with root package name */
    public static final c.c.a.m.o<Integer> f3949b = new c.c.a.m.o<>("com.bumptech.glide.load.resource.bitmap.VideoBitmapDecode.FrameOption", 2, new b());

    /* renamed from: c  reason: collision with root package name */
    public static final e f3950c = new e();

    /* renamed from: d  reason: collision with root package name */
    public final f<T> f3951d;

    /* renamed from: e  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f3952e;

    /* renamed from: f  reason: collision with root package name */
    public final e f3953f;

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public class a implements o.b<Long> {

        /* renamed from: a  reason: collision with root package name */
        public final ByteBuffer f3954a = ByteBuffer.allocate(8);

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [byte[], java.lang.Object, java.security.MessageDigest] */
        @Override // c.c.a.m.o.b
        public void a(byte[] bArr, Long l, MessageDigest messageDigest) {
            Long l2 = l;
            messageDigest.update(bArr);
            synchronized (this.f3954a) {
                this.f3954a.position(0);
                messageDigest.update(this.f3954a.putLong(l2.longValue()).array());
            }
        }
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public class b implements o.b<Integer> {

        /* renamed from: a  reason: collision with root package name */
        public final ByteBuffer f3955a = ByteBuffer.allocate(4);

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [byte[], java.lang.Object, java.security.MessageDigest] */
        @Override // c.c.a.m.o.b
        public void a(byte[] bArr, Integer num, MessageDigest messageDigest) {
            Integer num2 = num;
            if (num2 == null) {
                return;
            }
            messageDigest.update(bArr);
            synchronized (this.f3955a) {
                this.f3955a.position(0);
                messageDigest.update(this.f3955a.putInt(num2.intValue()).array());
            }
        }
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public static final class c implements f<AssetFileDescriptor> {
        public c(a aVar) {
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.media.MediaMetadataRetriever, java.lang.Object] */
        @Override // c.c.a.m.x.c.c0.f
        public void a(MediaMetadataRetriever mediaMetadataRetriever, AssetFileDescriptor assetFileDescriptor) {
            AssetFileDescriptor assetFileDescriptor2 = assetFileDescriptor;
            mediaMetadataRetriever.setDataSource(assetFileDescriptor2.getFileDescriptor(), assetFileDescriptor2.getStartOffset(), assetFileDescriptor2.getLength());
        }
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public static final class d implements f<ByteBuffer> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.media.MediaMetadataRetriever, java.lang.Object] */
        @Override // c.c.a.m.x.c.c0.f
        public void a(MediaMetadataRetriever mediaMetadataRetriever, ByteBuffer byteBuffer) {
            mediaMetadataRetriever.setDataSource(new d0(this, byteBuffer));
        }
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public static class e {
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public interface f<T> {
        void a(MediaMetadataRetriever mediaMetadataRetriever, T t);
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public static final class g implements f<ParcelFileDescriptor> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [android.media.MediaMetadataRetriever, java.lang.Object] */
        @Override // c.c.a.m.x.c.c0.f
        public void a(MediaMetadataRetriever mediaMetadataRetriever, ParcelFileDescriptor parcelFileDescriptor) {
            mediaMetadataRetriever.setDataSource(parcelFileDescriptor.getFileDescriptor());
        }
    }

    /* compiled from: VideoDecoder.java */
    /* loaded from: classes.dex */
    public static final class h extends RuntimeException {
        public h() {
            super("MediaMetadataRetriever failed to retrieve a frame without throwing, check the adb logs for .*MetadataRetriever.* prior to this exception for details");
        }
    }

    public c0(c.c.a.m.v.c0.d dVar, f<T> fVar) {
        e eVar = f3950c;
        this.f3952e = dVar;
        this.f3951d = fVar;
        this.f3953f = eVar;
    }

    /* JADX WARN: Removed duplicated region for block: B:23:0x0063  */
    /* JADX WARN: Removed duplicated region for block: B:25:0x0069 A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:26:0x006a  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static Bitmap c(MediaMetadataRetriever mediaMetadataRetriever, long j, int i, int i2, int i3, l lVar) {
        Bitmap bitmap;
        if (Build.VERSION.SDK_INT >= 27 && i2 != Integer.MIN_VALUE && i3 != Integer.MIN_VALUE && lVar != l.f3972d) {
            try {
                int parseInt = Integer.parseInt(mediaMetadataRetriever.extractMetadata(18));
                int parseInt2 = Integer.parseInt(mediaMetadataRetriever.extractMetadata(19));
                int parseInt3 = Integer.parseInt(mediaMetadataRetriever.extractMetadata(24));
                if (parseInt3 == 90 || parseInt3 == 270) {
                    parseInt2 = parseInt;
                    parseInt = parseInt2;
                }
                float b2 = lVar.b(parseInt, parseInt2, i2, i3);
                bitmap = mediaMetadataRetriever.getScaledFrameAtTime(j, i, Math.round(parseInt * b2), Math.round(b2 * parseInt2));
            } catch (Throwable th) {
                if (Log.isLoggable("VideoDecoder", 3)) {
                    Log.d("VideoDecoder", "Exception trying to decode a scaled frame on oreo+, falling back to a fullsize frame", th);
                }
            }
            if (bitmap == null) {
                bitmap = mediaMetadataRetriever.getFrameAtTime(j, i);
            }
            if (bitmap == null) {
                return bitmap;
            }
            throw new h();
        }
        bitmap = null;
        if (bitmap == null) {
        }
        if (bitmap == null) {
        }
    }

    @Override // c.c.a.m.r
    public boolean a(T t, c.c.a.m.p pVar) {
        return true;
    }

    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(T t, int i, int i2, c.c.a.m.p pVar) {
        long longValue = ((Long) pVar.c(f3948a)).longValue();
        if (longValue < 0 && longValue != -1) {
            throw new IllegalArgumentException(c.b.a.a.a.l("Requested frame must be non-negative, or DEFAULT_FRAME, given: ", longValue));
        }
        Integer num = (Integer) pVar.c(f3949b);
        if (num == null) {
            num = 2;
        }
        l lVar = (l) pVar.c(l.f3974f);
        if (lVar == null) {
            lVar = l.f3973e;
        }
        l lVar2 = lVar;
        Objects.requireNonNull(this.f3953f);
        MediaMetadataRetriever mediaMetadataRetriever = new MediaMetadataRetriever();
        try {
            this.f3951d.a(mediaMetadataRetriever, t);
            Bitmap c2 = c(mediaMetadataRetriever, longValue, num.intValue(), i, i2, lVar2);
            mediaMetadataRetriever.release();
            return c.c.a.m.x.c.e.b(c2, this.f3952e);
        } catch (Throwable th) {
            mediaMetadataRetriever.release();
            throw th;
        }
    }
}