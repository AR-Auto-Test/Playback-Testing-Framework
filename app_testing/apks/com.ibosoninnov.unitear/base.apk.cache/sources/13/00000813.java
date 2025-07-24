package c.c.a.m.x.c;

import android.util.Log;
import com.bumptech.glide.load.ImageHeaderParser;
import com.google.common.primitives.UnsignedBytes;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.Objects;

/* compiled from: DefaultImageHeaderParser.java */
/* loaded from: classes.dex */
public final class k implements ImageHeaderParser {

    /* renamed from: a  reason: collision with root package name */
    public static final byte[] f3964a = "Exif\u0000\u0000".getBytes(Charset.forName("UTF-8"));

    /* renamed from: b  reason: collision with root package name */
    public static final int[] f3965b = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};

    /* compiled from: DefaultImageHeaderParser.java */
    /* loaded from: classes.dex */
    public static final class a implements c {

        /* renamed from: a  reason: collision with root package name */
        public final ByteBuffer f3966a;

        public a(ByteBuffer byteBuffer) {
            this.f3966a = byteBuffer;
            byteBuffer.order(ByteOrder.BIG_ENDIAN);
        }

        @Override // c.c.a.m.x.c.k.c
        public long c(long j) {
            int min = (int) Math.min(this.f3966a.remaining(), j);
            ByteBuffer byteBuffer = this.f3966a;
            byteBuffer.position(byteBuffer.position() + min);
            return min;
        }

        @Override // c.c.a.m.x.c.k.c
        public int d() {
            return (f() << 8) | f();
        }

        @Override // c.c.a.m.x.c.k.c
        public int e(byte[] bArr, int i) {
            int min = Math.min(i, this.f3966a.remaining());
            if (min == 0) {
                return -1;
            }
            this.f3966a.get(bArr, 0, min);
            return min;
        }

        @Override // c.c.a.m.x.c.k.c
        public short f() {
            if (this.f3966a.remaining() >= 1) {
                return (short) (this.f3966a.get() & UnsignedBytes.MAX_VALUE);
            }
            throw new c.a();
        }
    }

    /* compiled from: DefaultImageHeaderParser.java */
    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final ByteBuffer f3967a;

        public b(byte[] bArr, int i) {
            this.f3967a = (ByteBuffer) ByteBuffer.wrap(bArr).order(ByteOrder.BIG_ENDIAN).limit(i);
        }

        public short a(int i) {
            if (this.f3967a.remaining() - i >= 2) {
                return this.f3967a.getShort(i);
            }
            return (short) -1;
        }

        public int b(int i) {
            if (this.f3967a.remaining() - i >= 4) {
                return this.f3967a.getInt(i);
            }
            return -1;
        }
    }

    /* compiled from: DefaultImageHeaderParser.java */
    /* loaded from: classes.dex */
    public interface c {

        /* compiled from: DefaultImageHeaderParser.java */
        /* loaded from: classes.dex */
        public static final class a extends IOException {
            public a() {
                super("Unexpectedly reached end of a file");
            }
        }

        long c(long j);

        int d();

        int e(byte[] bArr, int i);

        short f();
    }

    /* compiled from: DefaultImageHeaderParser.java */
    /* loaded from: classes.dex */
    public static final class d implements c {

        /* renamed from: a  reason: collision with root package name */
        public final InputStream f3968a;

        public d(InputStream inputStream) {
            this.f3968a = inputStream;
        }

        @Override // c.c.a.m.x.c.k.c
        public long c(long j) {
            if (j < 0) {
                return 0L;
            }
            long j2 = j;
            while (j2 > 0) {
                long skip = this.f3968a.skip(j2);
                if (skip <= 0) {
                    if (this.f3968a.read() == -1) {
                        break;
                    }
                    skip = 1;
                }
                j2 -= skip;
            }
            return j - j2;
        }

        @Override // c.c.a.m.x.c.k.c
        public int d() {
            return (f() << 8) | f();
        }

        @Override // c.c.a.m.x.c.k.c
        public int e(byte[] bArr, int i) {
            int i2 = 0;
            int i3 = 0;
            while (i2 < i && (i3 = this.f3968a.read(bArr, i2, i - i2)) != -1) {
                i2 += i3;
            }
            if (i2 == 0 && i3 == -1) {
                throw new c.a();
            }
            return i2;
        }

        @Override // c.c.a.m.x.c.k.c
        public short f() {
            int read = this.f3968a.read();
            if (read != -1) {
                return (short) read;
            }
            throw new c.a();
        }
    }

    @Override // com.bumptech.glide.load.ImageHeaderParser
    public ImageHeaderParser.ImageType a(ByteBuffer byteBuffer) {
        Objects.requireNonNull(byteBuffer, "Argument must not be null");
        return d(new a(byteBuffer));
    }

    @Override // com.bumptech.glide.load.ImageHeaderParser
    public ImageHeaderParser.ImageType b(InputStream inputStream) {
        Objects.requireNonNull(inputStream, "Argument must not be null");
        return d(new d(inputStream));
    }

    @Override // com.bumptech.glide.load.ImageHeaderParser
    public int c(InputStream inputStream, c.c.a.m.v.c0.b bVar) {
        Objects.requireNonNull(inputStream, "Argument must not be null");
        d dVar = new d(inputStream);
        Objects.requireNonNull(bVar, "Argument must not be null");
        int i = -1;
        try {
            int d2 = dVar.d();
            if (!((d2 & 65496) == 65496 || d2 == 19789 || d2 == 18761)) {
                if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                    Log.d("DfltImageHeaderParser", "Parser doesn't handle magic number: " + d2);
                }
            } else {
                int e2 = e(dVar);
                if (e2 == -1) {
                    if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                        Log.d("DfltImageHeaderParser", "Failed to parse exif segment length, or exif segment not found");
                    }
                } else {
                    byte[] bArr = (byte[]) bVar.d(e2, byte[].class);
                    int f2 = f(dVar, bArr, e2);
                    bVar.put(bArr);
                    i = f2;
                }
            }
        } catch (c.a unused) {
        }
        return i;
    }

    public final ImageHeaderParser.ImageType d(c cVar) {
        try {
            int d2 = cVar.d();
            if (d2 == 65496) {
                return ImageHeaderParser.ImageType.JPEG;
            }
            int f2 = (d2 << 8) | cVar.f();
            if (f2 == 4671814) {
                return ImageHeaderParser.ImageType.GIF;
            }
            int f3 = (f2 << 8) | cVar.f();
            if (f3 == -1991225785) {
                cVar.c(21L);
                try {
                    return cVar.f() >= 3 ? ImageHeaderParser.ImageType.PNG_A : ImageHeaderParser.ImageType.PNG;
                } catch (c.a unused) {
                    return ImageHeaderParser.ImageType.PNG;
                }
            } else if (f3 != 1380533830) {
                return ImageHeaderParser.ImageType.UNKNOWN;
            } else {
                cVar.c(4L);
                if (((cVar.d() << 16) | cVar.d()) != 1464156752) {
                    return ImageHeaderParser.ImageType.UNKNOWN;
                }
                int d3 = (cVar.d() << 16) | cVar.d();
                if ((d3 & (-256)) != 1448097792) {
                    return ImageHeaderParser.ImageType.UNKNOWN;
                }
                int i = d3 & 255;
                if (i == 88) {
                    cVar.c(4L);
                    return (cVar.f() & 16) != 0 ? ImageHeaderParser.ImageType.WEBP_A : ImageHeaderParser.ImageType.WEBP;
                } else if (i == 76) {
                    cVar.c(4L);
                    return (cVar.f() & 8) != 0 ? ImageHeaderParser.ImageType.WEBP_A : ImageHeaderParser.ImageType.WEBP;
                } else {
                    return ImageHeaderParser.ImageType.WEBP;
                }
            }
        } catch (c.a unused2) {
            return ImageHeaderParser.ImageType.UNKNOWN;
        }
    }

    public final int e(c cVar) {
        short f2;
        int d2;
        long j;
        long c2;
        do {
            short f3 = cVar.f();
            if (f3 != 255) {
                if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                    c.b.a.a.a.L("Unknown segmentId=", f3, "DfltImageHeaderParser");
                }
                return -1;
            }
            f2 = cVar.f();
            if (f2 == 218) {
                return -1;
            }
            if (f2 == 217) {
                if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                    Log.d("DfltImageHeaderParser", "Found MARKER_EOI in exif segment");
                }
                return -1;
            }
            d2 = cVar.d() - 2;
            if (f2 == 225) {
                return d2;
            }
            j = d2;
            c2 = cVar.c(j);
        } while (c2 == j);
        if (Log.isLoggable("DfltImageHeaderParser", 3)) {
            StringBuilder z = c.b.a.a.a.z("Unable to skip enough data, type: ", f2, ", wanted to skip: ", d2, ", but actually skipped: ");
            z.append(c2);
            Log.d("DfltImageHeaderParser", z.toString());
        }
        return -1;
    }

    public final int f(c cVar, byte[] bArr, int i) {
        ByteOrder byteOrder;
        int e2 = cVar.e(bArr, i);
        if (e2 != i) {
            if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                Log.d("DfltImageHeaderParser", "Unable to read exif segment data, length: " + i + ", actually read: " + e2);
            }
            return -1;
        }
        boolean z = bArr != null && i > f3964a.length;
        if (z) {
            int i2 = 0;
            while (true) {
                byte[] bArr2 = f3964a;
                if (i2 >= bArr2.length) {
                    break;
                } else if (bArr[i2] != bArr2[i2]) {
                    z = false;
                    break;
                } else {
                    i2++;
                }
            }
        }
        if (z) {
            b bVar = new b(bArr, i);
            short a2 = bVar.a(6);
            if (a2 == 18761) {
                byteOrder = ByteOrder.LITTLE_ENDIAN;
            } else if (a2 != 19789) {
                if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                    c.b.a.a.a.L("Unknown endianness = ", a2, "DfltImageHeaderParser");
                }
                byteOrder = ByteOrder.BIG_ENDIAN;
            } else {
                byteOrder = ByteOrder.BIG_ENDIAN;
            }
            bVar.f3967a.order(byteOrder);
            int b2 = bVar.b(10) + 6;
            short a3 = bVar.a(b2);
            for (int i3 = 0; i3 < a3; i3++) {
                int i4 = (i3 * 12) + b2 + 2;
                short a4 = bVar.a(i4);
                if (a4 == 274) {
                    short a5 = bVar.a(i4 + 2);
                    if (a5 >= 1 && a5 <= 12) {
                        int b3 = bVar.b(i4 + 4);
                        if (b3 < 0) {
                            if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                                Log.d("DfltImageHeaderParser", "Negative tiff component count");
                            }
                        } else {
                            if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                                StringBuilder z2 = c.b.a.a.a.z("Got tagIndex=", i3, " tagType=", a4, " formatCode=");
                                z2.append((int) a5);
                                z2.append(" componentCount=");
                                z2.append(b3);
                                Log.d("DfltImageHeaderParser", z2.toString());
                            }
                            int i5 = b3 + f3965b[a5];
                            if (i5 > 4) {
                                if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                                    c.b.a.a.a.L("Got byte count > 4, not orientation, continuing, formatCode=", a5, "DfltImageHeaderParser");
                                }
                            } else {
                                int i6 = i4 + 8;
                                if (i6 >= 0 && i6 <= bVar.f3967a.remaining()) {
                                    if (i5 >= 0 && i5 + i6 <= bVar.f3967a.remaining()) {
                                        return bVar.a(i6);
                                    }
                                    if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                                        c.b.a.a.a.L("Illegal number of bytes for TI tag data tagType=", a4, "DfltImageHeaderParser");
                                    }
                                } else if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                                    Log.d("DfltImageHeaderParser", "Illegal tagValueOffset=" + i6 + " tagType=" + ((int) a4));
                                }
                            }
                        }
                    } else if (Log.isLoggable("DfltImageHeaderParser", 3)) {
                        c.b.a.a.a.L("Got invalid format code = ", a5, "DfltImageHeaderParser");
                    }
                }
            }
            return -1;
        }
        if (Log.isLoggable("DfltImageHeaderParser", 3)) {
            Log.d("DfltImageHeaderParser", "Missing jpeg exif preamble");
        }
        return -1;
    }
}