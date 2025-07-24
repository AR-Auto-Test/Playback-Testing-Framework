package c.c.a.m.x.c;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ColorSpace;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Build;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import c.c.a.m.x.c.s;
import com.bumptech.glide.load.ImageHeaderParser;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.locks.Lock;

/* compiled from: Downsampler.java */
/* loaded from: classes.dex */
public final class m {

    /* renamed from: a  reason: collision with root package name */
    public static final c.c.a.m.o<c.c.a.m.b> f3976a = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.Downsampler.DecodeFormat", c.c.a.m.b.PREFER_ARGB_8888);

    /* renamed from: b  reason: collision with root package name */
    public static final c.c.a.m.o<c.c.a.m.q> f3977b = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.Downsampler.PreferredColorSpace", c.c.a.m.q.SRGB);

    /* renamed from: c  reason: collision with root package name */
    public static final c.c.a.m.o<Boolean> f3978c;

    /* renamed from: d  reason: collision with root package name */
    public static final c.c.a.m.o<Boolean> f3979d;

    /* renamed from: e  reason: collision with root package name */
    public static final Set<String> f3980e;

    /* renamed from: f  reason: collision with root package name */
    public static final b f3981f;

    /* renamed from: g  reason: collision with root package name */
    public static final Set<ImageHeaderParser.ImageType> f3982g;

    /* renamed from: h  reason: collision with root package name */
    public static final Queue<BitmapFactory.Options> f3983h;
    public final c.c.a.m.v.c0.d i;
    public final DisplayMetrics j;
    public final c.c.a.m.v.c0.b k;
    public final List<ImageHeaderParser> l;
    public final r m = r.a();

    /* compiled from: Downsampler.java */
    /* loaded from: classes.dex */
    public class a implements b {
        @Override // c.c.a.m.x.c.m.b
        public void a(c.c.a.m.v.c0.d dVar, Bitmap bitmap) {
        }

        @Override // c.c.a.m.x.c.m.b
        public void b() {
        }
    }

    /* compiled from: Downsampler.java */
    /* loaded from: classes.dex */
    public interface b {
        void a(c.c.a.m.v.c0.d dVar, Bitmap bitmap);

        void b();
    }

    static {
        c.c.a.m.o<l> oVar = l.f3974f;
        Boolean bool = Boolean.FALSE;
        f3978c = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.Downsampler.FixBitmapSize", bool);
        f3979d = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.Downsampler.AllowHardwareDecode", bool);
        f3980e = Collections.unmodifiableSet(new HashSet(Arrays.asList("image/vnd.wap.wbmp", "image/x-ico")));
        f3981f = new a();
        f3982g = Collections.unmodifiableSet(EnumSet.of(ImageHeaderParser.ImageType.JPEG, ImageHeaderParser.ImageType.PNG_A, ImageHeaderParser.ImageType.PNG));
        char[] cArr = c.c.a.s.j.f4197a;
        f3983h = new ArrayDeque(0);
    }

    public m(List<ImageHeaderParser> list, DisplayMetrics displayMetrics, c.c.a.m.v.c0.d dVar, c.c.a.m.v.c0.b bVar) {
        this.l = list;
        Objects.requireNonNull(displayMetrics, "Argument must not be null");
        this.j = displayMetrics;
        Objects.requireNonNull(dVar, "Argument must not be null");
        this.i = dVar;
        Objects.requireNonNull(bVar, "Argument must not be null");
        this.k = bVar;
    }

    public static Bitmap d(s sVar, BitmapFactory.Options options, b bVar, c.c.a.m.v.c0.d dVar) {
        if (!options.inJustDecodeBounds) {
            bVar.b();
            sVar.c();
        }
        int i = options.outWidth;
        int i2 = options.outHeight;
        String str = options.outMimeType;
        Lock lock = a0.f3941d;
        lock.lock();
        try {
            try {
                Bitmap b2 = sVar.b(options);
                lock.unlock();
                return b2;
            } catch (IllegalArgumentException e2) {
                IOException i3 = i(e2, i, i2, str, options);
                if (Log.isLoggable("Downsampler", 3)) {
                    Log.d("Downsampler", "Failed to decode with inBitmap, trying again without Bitmap re-use", i3);
                }
                Bitmap bitmap = options.inBitmap;
                if (bitmap != null) {
                    try {
                        dVar.d(bitmap);
                        options.inBitmap = null;
                        Bitmap d2 = d(sVar, options, bVar, dVar);
                        a0.f3941d.unlock();
                        return d2;
                    } catch (IOException unused) {
                        throw i3;
                    }
                }
                throw i3;
            }
        } catch (Throwable th) {
            a0.f3941d.unlock();
            throw th;
        }
    }

    @TargetApi(19)
    public static String e(Bitmap bitmap) {
        if (bitmap == null) {
            return null;
        }
        StringBuilder x = c.b.a.a.a.x(" (");
        x.append(bitmap.getAllocationByteCount());
        x.append(")");
        String sb = x.toString();
        StringBuilder x2 = c.b.a.a.a.x("[");
        x2.append(bitmap.getWidth());
        x2.append("x");
        x2.append(bitmap.getHeight());
        x2.append("] ");
        x2.append(bitmap.getConfig());
        x2.append(sb);
        return x2.toString();
    }

    public static int f(double d2) {
        if (d2 > 1.0d) {
            d2 = 1.0d / d2;
        }
        return (int) Math.round(d2 * 2.147483647E9d);
    }

    public static int[] g(s sVar, BitmapFactory.Options options, b bVar, c.c.a.m.v.c0.d dVar) {
        options.inJustDecodeBounds = true;
        d(sVar, options, bVar, dVar);
        options.inJustDecodeBounds = false;
        return new int[]{options.outWidth, options.outHeight};
    }

    public static boolean h(int i) {
        return i == 90 || i == 270;
    }

    public static IOException i(IllegalArgumentException illegalArgumentException, int i, int i2, String str, BitmapFactory.Options options) {
        StringBuilder z = c.b.a.a.a.z("Exception decoding bitmap, outWidth: ", i, ", outHeight: ", i2, ", outMimeType: ");
        z.append(str);
        z.append(", inBitmap: ");
        z.append(e(options.inBitmap));
        return new IOException(z.toString(), illegalArgumentException);
    }

    public static void j(BitmapFactory.Options options) {
        options.inTempStorage = null;
        options.inDither = false;
        options.inScaled = false;
        options.inSampleSize = 1;
        options.inPreferredConfig = null;
        options.inJustDecodeBounds = false;
        options.inDensity = 0;
        options.inTargetDensity = 0;
        if (Build.VERSION.SDK_INT >= 26) {
            options.inPreferredColorSpace = null;
            options.outColorSpace = null;
            options.outConfig = null;
        }
        options.outWidth = 0;
        options.outHeight = 0;
        options.outMimeType = null;
        options.inBitmap = null;
        options.inMutable = true;
    }

    public static int k(double d2) {
        return (int) (d2 + 0.5d);
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    public final c.c.a.m.v.w<Bitmap> a(s sVar, int i, int i2, c.c.a.m.p pVar, b bVar) {
        Queue<BitmapFactory.Options> queue;
        BitmapFactory.Options poll;
        BitmapFactory.Options options;
        byte[] bArr = (byte[]) this.k.d(65536, byte[].class);
        synchronized (m.class) {
            queue = f3983h;
            synchronized (queue) {
                poll = queue.poll();
            }
            if (poll == null) {
                poll = new BitmapFactory.Options();
                j(poll);
            }
            options = poll;
        }
        options.inTempStorage = bArr;
        c.c.a.m.b bVar2 = (c.c.a.m.b) pVar.c(f3976a);
        c.c.a.m.q qVar = (c.c.a.m.q) pVar.c(f3977b);
        l lVar = (l) pVar.c(l.f3974f);
        boolean booleanValue = ((Boolean) pVar.c(f3978c)).booleanValue();
        c.c.a.m.o<Boolean> oVar = f3979d;
        try {
            e b2 = e.b(c(sVar, options, lVar, bVar2, qVar, pVar.c(oVar) != null && ((Boolean) pVar.c(oVar)).booleanValue(), i, i2, booleanValue, bVar), this.i);
            j(options);
            synchronized (queue) {
                queue.offer(options);
            }
            this.k.put(bArr);
            return b2;
        } catch (Throwable th) {
            j(options);
            Queue<BitmapFactory.Options> queue2 = f3983h;
            synchronized (queue2) {
                queue2.offer(options);
                this.k.put(bArr);
                throw th;
            }
        }
    }

    public c.c.a.m.v.w<Bitmap> b(InputStream inputStream, int i, int i2, c.c.a.m.p pVar, b bVar) {
        return a(new s.a(inputStream, this.l, this.k), i, i2, pVar, bVar);
    }

    /* JADX WARN: Removed duplicated region for block: B:102:0x02ab  */
    /* JADX WARN: Removed duplicated region for block: B:104:0x02b4  */
    /* JADX WARN: Removed duplicated region for block: B:105:0x02b7  */
    /* JADX WARN: Removed duplicated region for block: B:124:0x0301 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:129:0x030f  */
    /* JADX WARN: Removed duplicated region for block: B:135:0x0319  */
    /* JADX WARN: Removed duplicated region for block: B:136:0x031f  */
    /* JADX WARN: Removed duplicated region for block: B:139:0x0349  */
    /* JADX WARN: Removed duplicated region for block: B:140:0x0386  */
    /* JADX WARN: Removed duplicated region for block: B:144:0x038d A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:147:0x0393  */
    /* JADX WARN: Removed duplicated region for block: B:151:0x039d  */
    /* JADX WARN: Removed duplicated region for block: B:153:0x03a0  */
    /* JADX WARN: Removed duplicated region for block: B:157:0x03ac  */
    /* JADX WARN: Removed duplicated region for block: B:169:0x03cd  */
    /* JADX WARN: Removed duplicated region for block: B:173:0x03ed  */
    /* JADX WARN: Removed duplicated region for block: B:175:0x0473  */
    /* JADX WARN: Removed duplicated region for block: B:195:0x050e A[ORIG_RETURN, RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:84:0x0197  */
    /* JADX WARN: Removed duplicated region for block: B:85:0x019b  */
    /* JADX WARN: Removed duplicated region for block: B:88:0x01a9  */
    /* JADX WARN: Removed duplicated region for block: B:89:0x020d  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final Bitmap c(s sVar, BitmapFactory.Options options, l lVar, c.c.a.m.b bVar, c.c.a.m.q qVar, boolean z, int i, int i2, boolean z2, b bVar2) {
        int i3;
        int i4;
        long j;
        int i5;
        int i6;
        String str;
        int i7;
        String str2;
        int i8;
        String str3;
        String str4;
        int i9;
        int i10;
        int i11;
        m mVar;
        boolean b2;
        String str5;
        boolean z3;
        boolean z4;
        int i12;
        int i13;
        String str6;
        int round;
        String str7;
        int i14;
        Bitmap d2;
        boolean z5;
        Bitmap e2;
        ColorSpace colorSpace;
        Bitmap.Config config;
        String str8;
        int i15;
        int i16;
        int min;
        int floor;
        int floor2;
        int f2;
        int k;
        int f3;
        int i17;
        String str9;
        int i18 = c.c.a.s.f.f4187b;
        long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        int[] g2 = g(sVar, options, bVar2, this.i);
        boolean z6 = false;
        int i19 = g2[0];
        int i20 = g2[1];
        String str10 = options.outMimeType;
        boolean z7 = (i19 == -1 || i20 == -1) ? false : z;
        int a2 = sVar.a();
        switch (a2) {
            case 3:
            case 4:
                i3 = BaseTransientBottomBar.ANIMATION_FADE_DURATION;
                i4 = i3;
                break;
            case 5:
            case 6:
                i3 = 90;
                i4 = i3;
                break;
            case 7:
            case 8:
                i3 = 270;
                i4 = i3;
                break;
            default:
                i4 = 0;
                break;
        }
        switch (a2) {
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                z6 = true;
                break;
        }
        if (i != Integer.MIN_VALUE) {
            j = elapsedRealtimeNanos;
            i5 = i;
        } else if (h(i4)) {
            j = elapsedRealtimeNanos;
            i5 = i20;
        } else {
            j = elapsedRealtimeNanos;
            i5 = i19;
        }
        if (i2 == Integer.MIN_VALUE) {
            i6 = h(i4) ? i19 : i20;
        } else {
            i6 = i2;
        }
        ImageHeaderParser.ImageType d3 = sVar.d();
        c.c.a.m.v.c0.d dVar = this.i;
        String str11 = ", density: ";
        boolean z8 = z6;
        boolean z9 = z7;
        if (i19 <= 0) {
            str = "]";
            i7 = i6;
            str2 = "x";
            i8 = i19;
            str3 = ", target density: ";
            str4 = "Downsampler";
            i9 = i20;
            i10 = i5;
            i11 = 3;
        } else if (i20 <= 0) {
            i7 = i6;
            i8 = i19;
            str3 = ", target density: ";
            str4 = "Downsampler";
            i9 = i20;
            i10 = i5;
            i11 = 3;
            str = "]";
            str2 = "x";
        } else {
            if (h(i4)) {
                str8 = "]";
                i16 = i20;
                i15 = i19;
            } else {
                str8 = "]";
                i15 = i20;
                i16 = i19;
            }
            float b3 = lVar.b(i16, i15, i5, i6);
            if (b3 > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                int a3 = lVar.a(i16, i15, i5, i6);
                if (a3 != 0) {
                    float f4 = i16;
                    float f5 = i15;
                    int i21 = i5;
                    int i22 = i6;
                    int k2 = i16 / k(b3 * f4);
                    int k3 = i15 / k(b3 * f5);
                    if (a3 == 1) {
                        min = Math.max(k2, k3);
                    } else {
                        min = Math.min(k2, k3);
                    }
                    int max = Math.max(1, Integer.highestOneBit(min));
                    if (a3 == 1 && max < 1.0f / b3) {
                        max <<= 1;
                    }
                    options.inSampleSize = max;
                    if (d3 == ImageHeaderParser.ImageType.JPEG) {
                        float min2 = Math.min(max, 8);
                        floor = (int) Math.ceil(f4 / min2);
                        floor2 = (int) Math.ceil(f5 / min2);
                        int i23 = max / 8;
                        if (i23 > 0) {
                            floor /= i23;
                            floor2 /= i23;
                        }
                    } else if (d3 != ImageHeaderParser.ImageType.PNG && d3 != ImageHeaderParser.ImageType.PNG_A) {
                        if (d3 != ImageHeaderParser.ImageType.WEBP && d3 != ImageHeaderParser.ImageType.WEBP_A) {
                            if (i16 % max == 0 && i15 % max == 0) {
                                int i24 = i15 / max;
                                i10 = i21;
                                i7 = i22;
                                floor = i16 / max;
                                floor2 = i24;
                                double b4 = lVar.b(floor, floor2, i10, i7);
                                int i25 = max;
                                options.inTargetDensity = k((b4 / (k / f2)) * k(f(b4) * b4));
                                f3 = f(b4);
                                options.inDensity = f3;
                                i17 = options.inTargetDensity;
                                if (!(i17 <= 0 && f3 > 0 && i17 != f3)) {
                                    options.inScaled = true;
                                } else {
                                    options.inTargetDensity = 0;
                                    options.inDensity = 0;
                                }
                                str4 = "Downsampler";
                                if (Log.isLoggable(str4, 2)) {
                                    str11 = ", density: ";
                                    str3 = ", target density: ";
                                    i9 = i20;
                                    str9 = "x";
                                    i8 = i19;
                                } else {
                                    i9 = i20;
                                    str9 = "x";
                                    i8 = i19;
                                    StringBuilder z10 = c.b.a.a.a.z("Calculate scaling, source: [", i8, str9, i9, "], degreesToRotate: ");
                                    z10.append(i4);
                                    z10.append(", target: [");
                                    z10.append(i10);
                                    z10.append(str9);
                                    z10.append(i7);
                                    z10.append("], power of two scaled: [");
                                    z10.append(floor);
                                    z10.append(str9);
                                    z10.append(floor2);
                                    z10.append("], exact scale factor: ");
                                    z10.append(b3);
                                    z10.append(", power of 2 sample size: ");
                                    z10.append(i25);
                                    z10.append(", adjusted scale factor: ");
                                    z10.append(b4);
                                    str3 = ", target density: ";
                                    z10.append(str3);
                                    z10.append(options.inTargetDensity);
                                    str11 = ", density: ";
                                    z10.append(str11);
                                    z10.append(options.inDensity);
                                    Log.v(str4, z10.toString());
                                }
                                mVar = this;
                                str2 = str9;
                                b2 = mVar.m.b(i10, i7, z9, z8);
                                if (b2) {
                                    options.inPreferredConfig = Bitmap.Config.HARDWARE;
                                    options.inMutable = false;
                                }
                                if (b2) {
                                    z4 = true;
                                    str5 = str11;
                                } else {
                                    str5 = str11;
                                    if (bVar != c.c.a.m.b.PREFER_ARGB_8888) {
                                        try {
                                            z3 = sVar.d().hasAlpha();
                                        } catch (IOException e3) {
                                            if (Log.isLoggable(str4, 3)) {
                                                Log.d(str4, "Cannot determine whether the image has alpha or not from header, format " + bVar, e3);
                                            }
                                            z3 = false;
                                        }
                                        Bitmap.Config config2 = z3 ? Bitmap.Config.ARGB_8888 : Bitmap.Config.RGB_565;
                                        options.inPreferredConfig = config2;
                                        if (config2 == Bitmap.Config.RGB_565) {
                                            options.inDither = true;
                                        }
                                        z4 = true;
                                    } else {
                                        z4 = true;
                                        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                                    }
                                }
                                i12 = Build.VERSION.SDK_INT;
                                i13 = options.inSampleSize;
                                if (i8 >= 0 || i9 < 0 || !z2) {
                                    int i26 = options.inTargetDensity;
                                    float f6 = (i26 > 0 || (i14 = options.inDensity) <= 0 || i26 == i14) ? false : false ? i26 / options.inDensity : 1.0f;
                                    float f7 = i13;
                                    str6 = str3;
                                    String str12 = str5;
                                    round = Math.round(((int) Math.ceil(i8 / f7)) * f6);
                                    int round2 = Math.round(((int) Math.ceil(i9 / f7)) * f6);
                                    if (Log.isLoggable(str4, 2)) {
                                        StringBuilder z11 = c.b.a.a.a.z("Calculated target [", round, str2, round2, "] for source [");
                                        z11.append(i8);
                                        z11.append(str2);
                                        z11.append(i9);
                                        z11.append("], sampleSize: ");
                                        z11.append(i13);
                                        z11.append(", targetDensity: ");
                                        z11.append(options.inTargetDensity);
                                        str7 = str12;
                                        z11.append(str7);
                                        z11.append(options.inDensity);
                                        z11.append(", density multiplier: ");
                                        z11.append(f6);
                                        Log.v(str4, z11.toString());
                                    } else {
                                        str7 = str12;
                                    }
                                    i7 = round2;
                                } else {
                                    str6 = str3;
                                    round = i10;
                                    str7 = str5;
                                }
                                if (round > 0 && i7 > 0) {
                                    c.c.a.m.v.c0.d dVar2 = mVar.i;
                                    if (i12 < 26) {
                                        config = options.inPreferredConfig != Bitmap.Config.HARDWARE ? options.outConfig : null;
                                    }
                                    if (config == null) {
                                        config = options.inPreferredConfig;
                                    }
                                    options.inBitmap = dVar2.c(round, i7, config);
                                }
                                if (i12 >= 28) {
                                    options.inPreferredColorSpace = ColorSpace.get(qVar == c.c.a.m.q.DISPLAY_P3 && (colorSpace = options.outColorSpace) != null && colorSpace.isWideGamut() ? ColorSpace.Named.DISPLAY_P3 : ColorSpace.Named.SRGB);
                                } else if (i12 >= 26) {
                                    options.inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB);
                                }
                                d2 = d(sVar, options, bVar2, mVar.i);
                                bVar2.a(mVar.i, d2);
                                if (Log.isLoggable(str4, 2)) {
                                    StringBuilder x = c.b.a.a.a.x("Decoded ");
                                    x.append(e(d2));
                                    x.append(" from [");
                                    x.append(i8);
                                    x.append(str2);
                                    x.append(i9);
                                    x.append("] ");
                                    x.append(str10);
                                    x.append(" with inBitmap ");
                                    x.append(e(options.inBitmap));
                                    x.append(" for [");
                                    x.append(i);
                                    x.append(str2);
                                    x.append(i2);
                                    x.append("], sample size: ");
                                    x.append(options.inSampleSize);
                                    x.append(str7);
                                    x.append(options.inDensity);
                                    x.append(str6);
                                    x.append(options.inTargetDensity);
                                    x.append(", thread: ");
                                    x.append(Thread.currentThread().getName());
                                    x.append(", duration: ");
                                    x.append(c.c.a.s.f.a(j));
                                    Log.v(str4, x.toString());
                                }
                                if (d2 != null) {
                                    d2.setDensity(mVar.j.densityDpi);
                                    c.c.a.m.v.c0.d dVar3 = mVar.i;
                                    switch (a2) {
                                        case 2:
                                        case 3:
                                        case 4:
                                        case 5:
                                        case 6:
                                        case 7:
                                        case 8:
                                            z5 = true;
                                            break;
                                        default:
                                            z5 = false;
                                            break;
                                    }
                                    if (z5) {
                                        Matrix matrix = new Matrix();
                                        switch (a2) {
                                            case 2:
                                                matrix.setScale(-1.0f, 1.0f);
                                                break;
                                            case 3:
                                                matrix.setRotate(180.0f);
                                                break;
                                            case 4:
                                                matrix.setRotate(180.0f);
                                                matrix.postScale(-1.0f, 1.0f);
                                                break;
                                            case 5:
                                                matrix.setRotate(90.0f);
                                                matrix.postScale(-1.0f, 1.0f);
                                                break;
                                            case 6:
                                                matrix.setRotate(90.0f);
                                                break;
                                            case 7:
                                                matrix.setRotate(-90.0f);
                                                matrix.postScale(-1.0f, 1.0f);
                                                break;
                                            case 8:
                                                matrix.setRotate(-90.0f);
                                                break;
                                        }
                                        RectF rectF = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, d2.getWidth(), d2.getHeight());
                                        matrix.mapRect(rectF);
                                        e2 = dVar3.e(Math.round(rectF.width()), Math.round(rectF.height()), a0.d(d2));
                                        matrix.postTranslate(-rectF.left, -rectF.top);
                                        e2.setHasAlpha(d2.hasAlpha());
                                        a0.a(d2, e2, matrix);
                                    } else {
                                        e2 = d2;
                                    }
                                    if (d2.equals(e2)) {
                                        return e2;
                                    }
                                    mVar.i.d(d2);
                                    return e2;
                                }
                                return null;
                            }
                            int[] g3 = g(sVar, options, bVar2, dVar);
                            floor = g3[0];
                            floor2 = g3[1];
                        } else {
                            float f8 = max;
                            floor = Math.round(f4 / f8);
                            floor2 = Math.round(f5 / f8);
                        }
                    } else {
                        float f9 = max;
                        floor = (int) Math.floor(f4 / f9);
                        floor2 = (int) Math.floor(f5 / f9);
                    }
                    i10 = i21;
                    i7 = i22;
                    double b42 = lVar.b(floor, floor2, i10, i7);
                    int i252 = max;
                    options.inTargetDensity = k((b42 / (k / f2)) * k(f(b42) * b42));
                    f3 = f(b42);
                    options.inDensity = f3;
                    i17 = options.inTargetDensity;
                    if (!(i17 <= 0 && f3 > 0 && i17 != f3)) {
                    }
                    str4 = "Downsampler";
                    if (Log.isLoggable(str4, 2)) {
                    }
                    mVar = this;
                    str2 = str9;
                    b2 = mVar.m.b(i10, i7, z9, z8);
                    if (b2) {
                    }
                    if (b2) {
                    }
                    i12 = Build.VERSION.SDK_INT;
                    i13 = options.inSampleSize;
                    if (i8 >= 0) {
                    }
                    int i262 = options.inTargetDensity;
                    if ((i262 > 0 || (i14 = options.inDensity) <= 0 || i262 == i14) ? false : false) {
                    }
                    float f72 = i13;
                    str6 = str3;
                    String str122 = str5;
                    round = Math.round(((int) Math.ceil(i8 / f72)) * f6);
                    int round22 = Math.round(((int) Math.ceil(i9 / f72)) * f6);
                    if (Log.isLoggable(str4, 2)) {
                    }
                    i7 = round22;
                    if (round > 0) {
                        c.c.a.m.v.c0.d dVar22 = mVar.i;
                        if (i12 < 26) {
                        }
                        if (config == null) {
                        }
                        options.inBitmap = dVar22.c(round, i7, config);
                    }
                    if (i12 >= 28) {
                    }
                    d2 = d(sVar, options, bVar2, mVar.i);
                    bVar2.a(mVar.i, d2);
                    if (Log.isLoggable(str4, 2)) {
                    }
                    if (d2 != null) {
                    }
                } else {
                    throw new IllegalArgumentException("Cannot round with null rounding");
                }
            } else {
                throw new IllegalArgumentException("Cannot scale with factor: " + b3 + " from: " + lVar + ", source: [" + i19 + "x" + i20 + "], target: [" + i5 + "x" + i6 + str8);
            }
        }
        if (Log.isLoggable(str4, i11)) {
            Log.d(str4, "Unable to determine dimensions for: " + d3 + " with target [" + i10 + str2 + i7 + str);
        }
        mVar = this;
        b2 = mVar.m.b(i10, i7, z9, z8);
        if (b2) {
        }
        if (b2) {
        }
        i12 = Build.VERSION.SDK_INT;
        i13 = options.inSampleSize;
        if (i8 >= 0) {
        }
        int i2622 = options.inTargetDensity;
        if ((i2622 > 0 || (i14 = options.inDensity) <= 0 || i2622 == i14) ? false : false) {
        }
        float f722 = i13;
        str6 = str3;
        String str1222 = str5;
        round = Math.round(((int) Math.ceil(i8 / f722)) * f6);
        int round222 = Math.round(((int) Math.ceil(i9 / f722)) * f6);
        if (Log.isLoggable(str4, 2)) {
        }
        i7 = round222;
        if (round > 0) {
        }
        if (i12 >= 28) {
        }
        d2 = d(sVar, options, bVar2, mVar.i);
        bVar2.a(mVar.i, d2);
        if (Log.isLoggable(str4, 2)) {
        }
        if (d2 != null) {
        }
    }
}