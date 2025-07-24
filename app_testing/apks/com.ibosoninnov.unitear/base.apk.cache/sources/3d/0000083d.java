package c.c.a.m.x.g;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import c.c.a.m.p;
import c.c.a.m.r;
import c.c.a.m.v.w;
import com.bumptech.glide.load.ImageHeaderParser;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Queue;

/* compiled from: ByteBufferGifDecoder.java */
/* loaded from: classes.dex */
public class a implements r<ByteBuffer, c> {

    /* renamed from: a  reason: collision with root package name */
    public static final C0082a f4026a = new C0082a();

    /* renamed from: b  reason: collision with root package name */
    public static final b f4027b = new b();

    /* renamed from: c  reason: collision with root package name */
    public final Context f4028c;

    /* renamed from: d  reason: collision with root package name */
    public final List<ImageHeaderParser> f4029d;

    /* renamed from: e  reason: collision with root package name */
    public final b f4030e;

    /* renamed from: f  reason: collision with root package name */
    public final C0082a f4031f;

    /* renamed from: g  reason: collision with root package name */
    public final c.c.a.m.x.g.b f4032g;

    /* compiled from: ByteBufferGifDecoder.java */
    /* renamed from: c.c.a.m.x.g.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0082a {
    }

    /* compiled from: ByteBufferGifDecoder.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public final Queue<c.c.a.l.d> f4033a;

        public b() {
            char[] cArr = c.c.a.s.j.f4197a;
            this.f4033a = new ArrayDeque(0);
        }

        public synchronized void a(c.c.a.l.d dVar) {
            dVar.f3505b = null;
            dVar.f3506c = null;
            this.f4033a.offer(dVar);
        }
    }

    public a(Context context, List<ImageHeaderParser> list, c.c.a.m.v.c0.d dVar, c.c.a.m.v.c0.b bVar) {
        b bVar2 = f4027b;
        C0082a c0082a = f4026a;
        this.f4028c = context.getApplicationContext();
        this.f4029d = list;
        this.f4031f = c0082a;
        this.f4032g = new c.c.a.m.x.g.b(dVar, bVar);
        this.f4030e = bVar2;
    }

    public static int d(c.c.a.l.c cVar, int i, int i2) {
        int min = Math.min(cVar.f3502g / i2, cVar.f3501f / i);
        int max = Math.max(1, min == 0 ? 0 : Integer.highestOneBit(min));
        if (Log.isLoggable("BufferGifDecoder", 2) && max > 1) {
            StringBuilder z = c.b.a.a.a.z("Downsampling GIF, sampleSize: ", max, ", target dimens: [", i, "x");
            z.append(i2);
            z.append("], actual dimens: [");
            z.append(cVar.f3501f);
            z.append("x");
            z.append(cVar.f3502g);
            z.append("]");
            Log.v("BufferGifDecoder", z.toString());
        }
        return max;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(ByteBuffer byteBuffer, p pVar) {
        ImageHeaderParser.ImageType l;
        ByteBuffer byteBuffer2 = byteBuffer;
        if (!((Boolean) pVar.c(i.f4060b)).booleanValue()) {
            List<ImageHeaderParser> list = this.f4029d;
            if (byteBuffer2 == null) {
                l = ImageHeaderParser.ImageType.UNKNOWN;
            } else {
                l = b.v.u.c.l(list, new c.c.a.m.g(byteBuffer2));
            }
            if (l == ImageHeaderParser.ImageType.GIF) {
                return true;
            }
        }
        return false;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public w<c> b(ByteBuffer byteBuffer, int i, int i2, p pVar) {
        c.c.a.l.d dVar;
        ByteBuffer byteBuffer2 = byteBuffer;
        b bVar = this.f4030e;
        synchronized (bVar) {
            c.c.a.l.d poll = bVar.f4033a.poll();
            if (poll == null) {
                poll = new c.c.a.l.d();
            }
            dVar = poll;
            dVar.f3505b = null;
            Arrays.fill(dVar.f3504a, (byte) 0);
            dVar.f3506c = new c.c.a.l.c();
            dVar.f3507d = 0;
            ByteBuffer asReadOnlyBuffer = byteBuffer2.asReadOnlyBuffer();
            dVar.f3505b = asReadOnlyBuffer;
            asReadOnlyBuffer.position(0);
            dVar.f3505b.order(ByteOrder.LITTLE_ENDIAN);
        }
        try {
            return c(byteBuffer2, i, i2, dVar, pVar);
        } finally {
            this.f4030e.a(dVar);
        }
    }

    /* JADX DEBUG: Another duplicated slice has different insns count: {[INVOKE]}, finally: {[INVOKE, INVOKE, INVOKE, INVOKE, INVOKE, INVOKE, IF] complete} */
    public final e c(ByteBuffer byteBuffer, int i, int i2, c.c.a.l.d dVar, p pVar) {
        Bitmap.Config config;
        int i3 = c.c.a.s.f.f4187b;
        long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        try {
            c.c.a.l.c b2 = dVar.b();
            if (b2.f3498c > 0 && b2.f3497b == 0) {
                if (pVar.c(i.f4059a) == c.c.a.m.b.PREFER_RGB_565) {
                    config = Bitmap.Config.RGB_565;
                } else {
                    config = Bitmap.Config.ARGB_8888;
                }
                int d2 = d(b2, i, i2);
                C0082a c0082a = this.f4031f;
                c.c.a.m.x.g.b bVar = this.f4032g;
                Objects.requireNonNull(c0082a);
                c.c.a.l.e eVar = new c.c.a.l.e(bVar, b2, byteBuffer, d2);
                eVar.i(config);
                eVar.l = (eVar.l + 1) % eVar.m.f3498c;
                Bitmap a2 = eVar.a();
                if (a2 == null) {
                    return null;
                }
                e eVar2 = new e(new c(this.f4028c, eVar, (c.c.a.m.x.b) c.c.a.m.x.b.f3935b, i, i2, a2));
                if (Log.isLoggable("BufferGifDecoder", 2)) {
                    StringBuilder x = c.b.a.a.a.x("Decoded GIF from stream in ");
                    x.append(c.c.a.s.f.a(elapsedRealtimeNanos));
                    Log.v("BufferGifDecoder", x.toString());
                }
                return eVar2;
            }
            if (Log.isLoggable("BufferGifDecoder", 2)) {
                StringBuilder x2 = c.b.a.a.a.x("Decoded GIF from stream in ");
                x2.append(c.c.a.s.f.a(elapsedRealtimeNanos));
                Log.v("BufferGifDecoder", x2.toString());
            }
            return null;
        } finally {
            if (Log.isLoggable("BufferGifDecoder", 2)) {
                StringBuilder x3 = c.b.a.a.a.x("Decoded GIF from stream in ");
                x3.append(c.c.a.s.f.a(elapsedRealtimeNanos));
                Log.v("BufferGifDecoder", x3.toString());
            }
        }
    }
}