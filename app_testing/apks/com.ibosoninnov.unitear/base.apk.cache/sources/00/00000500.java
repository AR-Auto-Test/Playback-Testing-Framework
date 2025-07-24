package b.p.a;

import android.content.res.AssetManager;
import android.media.MediaDataSource;
import android.media.MediaMetadataRetriever;
import android.system.Os;
import android.system.OsConstants;
import android.util.Log;
import com.google.android.material.datepicker.UtcDates;
import com.google.common.base.Ascii;
import com.google.common.primitives.UnsignedBytes;
import com.google.common.primitives.UnsignedInts;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TimeZone;
import java.util.regex.Pattern;
import java.util.zip.CRC32;
import org.opencv.imgcodecs.Imgcodecs;

/* compiled from: ExifInterface.java */
/* loaded from: classes.dex */
public class a {
    public static final d[] A;
    public static final d B;
    public static final d[] C;
    public static final d[] D;
    public static final d[] E;
    public static final d[] F;
    public static final d[][] G;
    public static final d[] H;
    public static final HashMap<Integer, d>[] I;
    public static final HashMap<String, d>[] J;
    public static final HashSet<String> K;
    public static final HashMap<Integer, Integer> L;
    public static final Charset M;
    public static final byte[] N;
    public static final byte[] O;

    /* renamed from: a  reason: collision with root package name */
    public static final boolean f2370a = Log.isLoggable("ExifInterface", 3);

    /* renamed from: b  reason: collision with root package name */
    public static final List<Integer> f2371b = Arrays.asList(1, 6, 3, 8);

    /* renamed from: c  reason: collision with root package name */
    public static final List<Integer> f2372c = Arrays.asList(2, 7, 4, 5);

    /* renamed from: d  reason: collision with root package name */
    public static final int[] f2373d = {8, 8, 8};

    /* renamed from: e  reason: collision with root package name */
    public static final int[] f2374e = {8};

    /* renamed from: f  reason: collision with root package name */
    public static final byte[] f2375f = {-1, -40, -1};

    /* renamed from: g  reason: collision with root package name */
    public static final byte[] f2376g = {102, 116, 121, 112};

    /* renamed from: h  reason: collision with root package name */
    public static final byte[] f2377h = {109, 105, 102, 49};
    public static final byte[] i = {104, 101, 105, 99};
    public static final byte[] j = {79, 76, 89, 77, 80, 0};
    public static final byte[] k = {79, 76, 89, 77, 80, 85, 83, 0, 73, 73};
    public static final byte[] l = {-119, 80, 78, 71, 13, 10, Ascii.SUB, 10};
    public static final byte[] m = {101, 88, 73, 102};
    public static final byte[] n = {73, 72, 68, 82};
    public static final byte[] o = {73, 69, 78, 68};
    public static final byte[] p = {82, 73, 70, 70};
    public static final byte[] q = {87, 69, 66, 80};
    public static final byte[] r = {69, 88, 73, 70};
    public static SimpleDateFormat s;
    public static final String[] t;
    public static final int[] u;
    public static final byte[] v;
    public static final d[] w;
    public static final d[] x;
    public static final d[] y;
    public static final d[] z;
    public FileDescriptor P;
    public AssetManager.AssetInputStream Q;
    public int R;
    public boolean S;
    public final HashMap<String, c>[] T;
    public Set<Integer> U;
    public ByteOrder V;
    public boolean W;
    public int X;
    public int Y;
    public int Z;
    public int a0;
    public int b0;
    public int c0;
    public int d0;
    public int e0;

    /* compiled from: ExifInterface.java */
    /* renamed from: b.p.a.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0046a extends MediaDataSource {

        /* renamed from: b  reason: collision with root package name */
        public long f2378b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ b f2379c;

        public C0046a(a aVar, b bVar) {
            this.f2379c = bVar;
        }

        @Override // java.io.Closeable, java.lang.AutoCloseable
        public void close() {
        }

        @Override // android.media.MediaDataSource
        public long getSize() {
            return -1L;
        }

        @Override // android.media.MediaDataSource
        public int readAt(long j, byte[] bArr, int i, int i2) {
            if (i2 == 0) {
                return 0;
            }
            if (j < 0) {
                return -1;
            }
            try {
                long j2 = this.f2378b;
                if (j2 != j) {
                    if (j2 >= 0 && j >= j2 + this.f2379c.available()) {
                        return -1;
                    }
                    this.f2379c.C(j);
                    this.f2378b = j;
                }
                if (i2 > this.f2379c.available()) {
                    i2 = this.f2379c.available();
                }
                b bVar = this.f2379c;
                int read = bVar.f2382d.read(bArr, i, i2);
                bVar.f2385g += read;
                if (read >= 0) {
                    this.f2378b += read;
                    return read;
                }
            } catch (IOException unused) {
            }
            this.f2378b = -1L;
            return -1;
        }
    }

    /* compiled from: ExifInterface.java */
    /* loaded from: classes.dex */
    public static class e {

        /* renamed from: a  reason: collision with root package name */
        public final long f2393a;

        /* renamed from: b  reason: collision with root package name */
        public final long f2394b;

        public e(long j, long j2) {
            if (j2 == 0) {
                this.f2393a = 0L;
                this.f2394b = 1L;
                return;
            }
            this.f2393a = j;
            this.f2394b = j2;
        }

        public String toString() {
            return this.f2393a + "/" + this.f2394b;
        }
    }

    static {
        d[] dVarArr;
        "VP8X".getBytes(Charset.defaultCharset());
        "VP8L".getBytes(Charset.defaultCharset());
        "VP8 ".getBytes(Charset.defaultCharset());
        "ANIM".getBytes(Charset.defaultCharset());
        "ANMF".getBytes(Charset.defaultCharset());
        "XMP ".getBytes(Charset.defaultCharset());
        t = new String[]{"", "BYTE", "STRING", "USHORT", "ULONG", "URATIONAL", "SBYTE", "UNDEFINED", "SSHORT", "SLONG", "SRATIONAL", "SINGLE", "DOUBLE", "IFD"};
        u = new int[]{0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 1};
        v = new byte[]{65, 83, 67, 73, 73, 0, 0, 0};
        d[] dVarArr2 = {new d("NewSubfileType", 254, 4), new d("SubfileType", 255, 4), new d("ImageWidth", 256, 3, 4), new d("ImageLength", Imgcodecs.IMWRITE_TIFF_XDPI, 3, 4), new d("BitsPerSample", Imgcodecs.IMWRITE_TIFF_YDPI, 3), new d("Compression", Imgcodecs.IMWRITE_TIFF_COMPRESSION, 3), new d("PhotometricInterpretation", 262, 3), new d("ImageDescription", 270, 2), new d("Make", 271, 2), new d("Model", 272, 2), new d("StripOffsets", 273, 3, 4), new d("Orientation", 274, 3), new d("SamplesPerPixel", 277, 3), new d("RowsPerStrip", 278, 3, 4), new d("StripByteCounts", 279, 3, 4), new d("XResolution", 282, 5), new d("YResolution", 283, 5), new d("PlanarConfiguration", 284, 3), new d("ResolutionUnit", 296, 3), new d("TransferFunction", 301, 3), new d("Software", 305, 2), new d("DateTime", 306, 2), new d("Artist", 315, 2), new d("WhitePoint", 318, 5), new d("PrimaryChromaticities", 319, 5), new d("SubIFDPointer", 330, 4), new d("JPEGInterchangeFormat", 513, 4), new d("JPEGInterchangeFormatLength", 514, 4), new d("YCbCrCoefficients", 529, 5), new d("YCbCrSubSampling", 530, 3), new d("YCbCrPositioning", 531, 3), new d("ReferenceBlackWhite", 532, 5), new d("Copyright", 33432, 2), new d("ExifIFDPointer", 34665, 4), new d("GPSInfoIFDPointer", 34853, 4), new d("SensorTopBorder", 4, 4), new d("SensorLeftBorder", 5, 4), new d("SensorBottomBorder", 6, 4), new d("SensorRightBorder", 7, 4), new d("ISO", 23, 3), new d("JpgFromRaw", 46, 7), new d("Xmp", 700, 1)};
        w = dVarArr2;
        d[] dVarArr3 = {new d("ExposureTime", 33434, 5), new d("FNumber", 33437, 5), new d("ExposureProgram", 34850, 3), new d("SpectralSensitivity", 34852, 2), new d("PhotographicSensitivity", 34855, 3), new d("OECF", 34856, 7), new d("SensitivityType", 34864, 3), new d("StandardOutputSensitivity", 34865, 4), new d("RecommendedExposureIndex", 34866, 4), new d("ISOSpeed", 34867, 4), new d("ISOSpeedLatitudeyyy", 34868, 4), new d("ISOSpeedLatitudezzz", 34869, 4), new d("ExifVersion", 36864, 2), new d("DateTimeOriginal", 36867, 2), new d("DateTimeDigitized", 36868, 2), new d("OffsetTime", 36880, 2), new d("OffsetTimeOriginal", 36881, 2), new d("OffsetTimeDigitized", 36882, 2), new d("ComponentsConfiguration", 37121, 7), new d("CompressedBitsPerPixel", 37122, 5), new d("ShutterSpeedValue", 37377, 10), new d("ApertureValue", 37378, 5), new d("BrightnessValue", 37379, 10), new d("ExposureBiasValue", 37380, 10), new d("MaxApertureValue", 37381, 5), new d("SubjectDistance", 37382, 5), new d("MeteringMode", 37383, 3), new d("LightSource", 37384, 3), new d("Flash", 37385, 3), new d("FocalLength", 37386, 5), new d("SubjectArea", 37396, 3), new d("MakerNote", 37500, 7), new d("UserComment", 37510, 7), new d("SubSecTime", 37520, 2), new d("SubSecTimeOriginal", 37521, 2), new d("SubSecTimeDigitized", 37522, 2), new d("FlashpixVersion", 40960, 7), new d("ColorSpace", 40961, 3), new d("PixelXDimension", 40962, 3, 4), new d("PixelYDimension", 40963, 3, 4), new d("RelatedSoundFile", 40964, 2), new d("InteroperabilityIFDPointer", 40965, 4), new d("FlashEnergy", 41483, 5), new d("SpatialFrequencyResponse", 41484, 7), new d("FocalPlaneXResolution", 41486, 5), new d("FocalPlaneYResolution", 41487, 5), new d("FocalPlaneResolutionUnit", 41488, 3), new d("SubjectLocation", 41492, 3), new d("ExposureIndex", 41493, 5), new d("SensingMethod", 41495, 3), new d("FileSource", 41728, 7), new d("SceneType", 41729, 7), new d("CFAPattern", 41730, 7), new d("CustomRendered", 41985, 3), new d("ExposureMode", 41986, 3), new d("WhiteBalance", 41987, 3), new d("DigitalZoomRatio", 41988, 5), new d("FocalLengthIn35mmFilm", 41989, 3), new d("SceneCaptureType", 41990, 3), new d("GainControl", 41991, 3), new d("Contrast", 41992, 3), new d("Saturation", 41993, 3), new d("Sharpness", 41994, 3), new d("DeviceSettingDescription", 41995, 7), new d("SubjectDistanceRange", 41996, 3), new d("ImageUniqueID", 42016, 2), new d("CameraOwnerName", 42032, 2), new d("BodySerialNumber", 42033, 2), new d("LensSpecification", 42034, 5), new d("LensMake", 42035, 2), new d("LensModel", 42036, 2), new d("Gamma", 42240, 5), new d("DNGVersion", 50706, 1), new d("DefaultCropSize", 50720, 3, 4)};
        x = dVarArr3;
        d[] dVarArr4 = {new d("GPSVersionID", 0, 1), new d("GPSLatitudeRef", 1, 2), new d("GPSLatitude", 2, 5), new d("GPSLongitudeRef", 3, 2), new d("GPSLongitude", 4, 5), new d("GPSAltitudeRef", 5, 1), new d("GPSAltitude", 6, 5), new d("GPSTimeStamp", 7, 5), new d("GPSSatellites", 8, 2), new d("GPSStatus", 9, 2), new d("GPSMeasureMode", 10, 2), new d("GPSDOP", 11, 5), new d("GPSSpeedRef", 12, 2), new d("GPSSpeed", 13, 5), new d("GPSTrackRef", 14, 2), new d("GPSTrack", 15, 5), new d("GPSImgDirectionRef", 16, 2), new d("GPSImgDirection", 17, 5), new d("GPSMapDatum", 18, 2), new d("GPSDestLatitudeRef", 19, 2), new d("GPSDestLatitude", 20, 5), new d("GPSDestLongitudeRef", 21, 2), new d("GPSDestLongitude", 22, 5), new d("GPSDestBearingRef", 23, 2), new d("GPSDestBearing", 24, 5), new d("GPSDestDistanceRef", 25, 2), new d("GPSDestDistance", 26, 5), new d("GPSProcessingMethod", 27, 7), new d("GPSAreaInformation", 28, 7), new d("GPSDateStamp", 29, 2), new d("GPSDifferential", 30, 3), new d("GPSHPositioningError", 31, 5)};
        y = dVarArr4;
        d[] dVarArr5 = {new d("InteroperabilityIndex", 1, 2)};
        z = dVarArr5;
        d[] dVarArr6 = {new d("NewSubfileType", 254, 4), new d("SubfileType", 255, 4), new d("ThumbnailImageWidth", 256, 3, 4), new d("ThumbnailImageLength", Imgcodecs.IMWRITE_TIFF_XDPI, 3, 4), new d("BitsPerSample", Imgcodecs.IMWRITE_TIFF_YDPI, 3), new d("Compression", Imgcodecs.IMWRITE_TIFF_COMPRESSION, 3), new d("PhotometricInterpretation", 262, 3), new d("ImageDescription", 270, 2), new d("Make", 271, 2), new d("Model", 272, 2), new d("StripOffsets", 273, 3, 4), new d("ThumbnailOrientation", 274, 3), new d("SamplesPerPixel", 277, 3), new d("RowsPerStrip", 278, 3, 4), new d("StripByteCounts", 279, 3, 4), new d("XResolution", 282, 5), new d("YResolution", 283, 5), new d("PlanarConfiguration", 284, 3), new d("ResolutionUnit", 296, 3), new d("TransferFunction", 301, 3), new d("Software", 305, 2), new d("DateTime", 306, 2), new d("Artist", 315, 2), new d("WhitePoint", 318, 5), new d("PrimaryChromaticities", 319, 5), new d("SubIFDPointer", 330, 4), new d("JPEGInterchangeFormat", 513, 4), new d("JPEGInterchangeFormatLength", 514, 4), new d("YCbCrCoefficients", 529, 5), new d("YCbCrSubSampling", 530, 3), new d("YCbCrPositioning", 531, 3), new d("ReferenceBlackWhite", 532, 5), new d("Copyright", 33432, 2), new d("ExifIFDPointer", 34665, 4), new d("GPSInfoIFDPointer", 34853, 4), new d("DNGVersion", 50706, 1), new d("DefaultCropSize", 50720, 3, 4)};
        A = dVarArr6;
        B = new d("StripOffsets", 273, 3);
        d[] dVarArr7 = {new d("ThumbnailImage", 256, 7), new d("CameraSettingsIFDPointer", 8224, 4), new d("ImageProcessingIFDPointer", 8256, 4)};
        C = dVarArr7;
        d[] dVarArr8 = {new d("PreviewImageStart", Imgcodecs.IMWRITE_TIFF_XDPI, 4), new d("PreviewImageLength", Imgcodecs.IMWRITE_TIFF_YDPI, 4)};
        D = dVarArr8;
        d[] dVarArr9 = {new d("AspectFrame", 4371, 3)};
        E = dVarArr9;
        d[] dVarArr10 = {new d("ColorSpace", 55, 3)};
        F = dVarArr10;
        d[][] dVarArr11 = {dVarArr2, dVarArr3, dVarArr4, dVarArr5, dVarArr6, dVarArr2, dVarArr7, dVarArr8, dVarArr9, dVarArr10};
        G = dVarArr11;
        H = new d[]{new d("SubIFDPointer", 330, 4), new d("ExifIFDPointer", 34665, 4), new d("GPSInfoIFDPointer", 34853, 4), new d("InteroperabilityIFDPointer", 40965, 4), new d("CameraSettingsIFDPointer", 8224, 1), new d("ImageProcessingIFDPointer", 8256, 1)};
        I = new HashMap[dVarArr11.length];
        J = new HashMap[dVarArr11.length];
        K = new HashSet<>(Arrays.asList("FNumber", "DigitalZoomRatio", "ExposureTime", "SubjectDistance", "GPSTimeStamp"));
        L = new HashMap<>();
        Charset forName = Charset.forName("US-ASCII");
        M = forName;
        N = "Exif\u0000\u0000".getBytes(forName);
        O = "http://ns.adobe.com/xap/1.0/\u0000".getBytes(forName);
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy:MM:dd HH:mm:ss");
        s = simpleDateFormat;
        simpleDateFormat.setTimeZone(TimeZone.getTimeZone(UtcDates.UTC));
        int i2 = 0;
        while (true) {
            d[][] dVarArr12 = G;
            if (i2 < dVarArr12.length) {
                I[i2] = new HashMap<>();
                J[i2] = new HashMap<>();
                for (d dVar : dVarArr12[i2]) {
                    I[i2].put(Integer.valueOf(dVar.f2389a), dVar);
                    J[i2].put(dVar.f2390b, dVar);
                }
                i2++;
            } else {
                HashMap<Integer, Integer> hashMap = L;
                d[] dVarArr13 = H;
                hashMap.put(Integer.valueOf(dVarArr13[0].f2389a), 5);
                hashMap.put(Integer.valueOf(dVarArr13[1].f2389a), 1);
                hashMap.put(Integer.valueOf(dVarArr13[2].f2389a), 2);
                hashMap.put(Integer.valueOf(dVarArr13[3].f2389a), 3);
                hashMap.put(Integer.valueOf(dVarArr13[4].f2389a), 7);
                hashMap.put(Integer.valueOf(dVarArr13[5].f2389a), 8);
                Pattern.compile(".*[1-9].*");
                Pattern.compile("^([0-9][0-9]):([0-9][0-9]):([0-9][0-9])$");
                return;
            }
        }
    }

    public a(InputStream inputStream) {
        boolean z2;
        d[][] dVarArr = G;
        this.T = new HashMap[dVarArr.length];
        this.U = new HashSet(dVarArr.length);
        this.V = ByteOrder.BIG_ENDIAN;
        Objects.requireNonNull(inputStream, "inputStream cannot be null");
        if (inputStream instanceof AssetManager.AssetInputStream) {
            this.Q = (AssetManager.AssetInputStream) inputStream;
            this.P = null;
        } else {
            if (inputStream instanceof FileInputStream) {
                FileInputStream fileInputStream = (FileInputStream) inputStream;
                try {
                    Os.lseek(fileInputStream.getFD(), 0L, OsConstants.SEEK_CUR);
                    z2 = true;
                } catch (Exception unused) {
                    if (f2370a) {
                        Log.d("ExifInterface", "The file descriptor for the given input is not seekable");
                    }
                    z2 = false;
                }
                if (z2) {
                    this.Q = null;
                    this.P = fileInputStream.getFD();
                }
            }
            this.Q = null;
            this.P = null;
        }
        for (int i2 = 0; i2 < G.length; i2++) {
            try {
                try {
                    this.T[i2] = new HashMap<>();
                } catch (IOException e2) {
                    boolean z3 = f2370a;
                    if (z3) {
                        Log.w("ExifInterface", "Invalid image: ExifInterface got an unsupported image format file(ExifInterface supports JPEG and some RAW image formats only) or a corrupted JPEG file to ExifInterface.", e2);
                    }
                    a();
                    if (!z3) {
                        return;
                    }
                }
            } finally {
                a();
                if (f2370a) {
                    s();
                }
            }
        }
        if (!this.S) {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(inputStream, 5000);
            this.R = h(bufferedInputStream);
            inputStream = bufferedInputStream;
        }
        b bVar = new b(inputStream);
        if (!this.S) {
            switch (this.R) {
                case 0:
                case 1:
                case 2:
                case 3:
                case 5:
                case 6:
                case 8:
                case 11:
                    l(bVar);
                    break;
                case 4:
                    g(bVar, 0, 0);
                    break;
                case 7:
                    i(bVar);
                    break;
                case 9:
                    k(bVar);
                    break;
                case 10:
                    m(bVar);
                    break;
                case 12:
                    f(bVar);
                    break;
                case 13:
                    j(bVar);
                    break;
                case 14:
                    o(bVar);
                    break;
            }
        } else {
            n(bVar);
        }
        w(bVar);
    }

    public static String b(byte[] bArr) {
        StringBuilder sb = new StringBuilder(bArr.length * 2);
        for (int i2 = 0; i2 < bArr.length; i2++) {
            sb.append(String.format("%02x", Byte.valueOf(bArr[i2])));
        }
        return sb.toString();
    }

    public static long[] c(Object obj) {
        if (obj instanceof int[]) {
            int[] iArr = (int[]) obj;
            long[] jArr = new long[iArr.length];
            for (int i2 = 0; i2 < iArr.length; i2++) {
                jArr[i2] = iArr[i2];
            }
            return jArr;
        } else if (obj instanceof long[]) {
            return (long[]) obj;
        } else {
            return null;
        }
    }

    public static boolean x(byte[] bArr, byte[] bArr2) {
        if (bArr2 != null && bArr.length >= bArr2.length) {
            for (int i2 = 0; i2 < bArr2.length; i2++) {
                if (bArr[i2] != bArr2[i2]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    public final void A() {
        y(0, 5);
        y(0, 4);
        y(5, 4);
        c cVar = this.T[1].get("PixelXDimension");
        c cVar2 = this.T[1].get("PixelYDimension");
        if (cVar != null && cVar2 != null) {
            this.T[0].put("ImageWidth", cVar);
            this.T[0].put("ImageLength", cVar2);
        }
        if (this.T[4].isEmpty() && q(this.T[5])) {
            HashMap<String, c>[] hashMapArr = this.T;
            hashMapArr[4] = hashMapArr[5];
            hashMapArr[5] = new HashMap<>();
        }
        if (q(this.T[4])) {
            return;
        }
        Log.d("ExifInterface", "No image meets the size requirements of a thumbnail image.");
    }

    public final void a() {
        String d2 = d("DateTimeOriginal");
        if (d2 != null && d("DateTime") == null) {
            this.T[0].put("DateTime", c.a(d2));
        }
        if (d("ImageWidth") == null) {
            this.T[0].put("ImageWidth", c.b(0L, this.V));
        }
        if (d("ImageLength") == null) {
            this.T[0].put("ImageLength", c.b(0L, this.V));
        }
        if (d("Orientation") == null) {
            this.T[0].put("Orientation", c.b(0L, this.V));
        }
        if (d("LightSource") == null) {
            this.T[1].put("LightSource", c.b(0L, this.V));
        }
    }

    public String d(String str) {
        c e2 = e(str);
        if (e2 != null) {
            if (!K.contains(str)) {
                return e2.g(this.V);
            }
            if (str.equals("GPSTimeStamp")) {
                int i2 = e2.f2386a;
                if (i2 != 5 && i2 != 10) {
                    StringBuilder x2 = c.b.a.a.a.x("GPS Timestamp format is not rational. format=");
                    x2.append(e2.f2386a);
                    Log.w("ExifInterface", x2.toString());
                    return null;
                }
                e[] eVarArr = (e[]) e2.h(this.V);
                if (eVarArr != null && eVarArr.length == 3) {
                    return String.format("%02d:%02d:%02d", Integer.valueOf((int) (((float) eVarArr[0].f2393a) / ((float) eVarArr[0].f2394b))), Integer.valueOf((int) (((float) eVarArr[1].f2393a) / ((float) eVarArr[1].f2394b))), Integer.valueOf((int) (((float) eVarArr[2].f2393a) / ((float) eVarArr[2].f2394b))));
                }
                StringBuilder x3 = c.b.a.a.a.x("Invalid GPS Timestamp array. array=");
                x3.append(Arrays.toString(eVarArr));
                Log.w("ExifInterface", x3.toString());
                return null;
            }
            try {
                return Double.toString(e2.e(this.V));
            } catch (NumberFormatException unused) {
            }
        }
        return null;
    }

    public final c e(String str) {
        if ("ISOSpeedRatings".equals(str)) {
            if (f2370a) {
                Log.d("ExifInterface", "getExifAttribute: Replacing TAG_ISO_SPEED_RATINGS with TAG_PHOTOGRAPHIC_SENSITIVITY.");
            }
            str = "PhotographicSensitivity";
        }
        for (int i2 = 0; i2 < G.length; i2++) {
            c cVar = this.T[i2].get(str);
            if (cVar != null) {
                return cVar;
            }
        }
        return null;
    }

    public final void f(b bVar) {
        String str;
        String str2;
        MediaMetadataRetriever mediaMetadataRetriever = new MediaMetadataRetriever();
        try {
            mediaMetadataRetriever.setDataSource(new C0046a(this, bVar));
            String extractMetadata = mediaMetadataRetriever.extractMetadata(33);
            String extractMetadata2 = mediaMetadataRetriever.extractMetadata(34);
            String extractMetadata3 = mediaMetadataRetriever.extractMetadata(26);
            String extractMetadata4 = mediaMetadataRetriever.extractMetadata(17);
            String str3 = null;
            if ("yes".equals(extractMetadata3)) {
                str3 = mediaMetadataRetriever.extractMetadata(29);
                str = mediaMetadataRetriever.extractMetadata(30);
                str2 = mediaMetadataRetriever.extractMetadata(31);
            } else if ("yes".equals(extractMetadata4)) {
                str3 = mediaMetadataRetriever.extractMetadata(18);
                str = mediaMetadataRetriever.extractMetadata(19);
                str2 = mediaMetadataRetriever.extractMetadata(24);
            } else {
                str = null;
                str2 = null;
            }
            if (str3 != null) {
                this.T[0].put("ImageWidth", c.d(Integer.parseInt(str3), this.V));
            }
            if (str != null) {
                this.T[0].put("ImageLength", c.d(Integer.parseInt(str), this.V));
            }
            if (str2 != null) {
                int i2 = 1;
                int parseInt = Integer.parseInt(str2);
                if (parseInt == 90) {
                    i2 = 6;
                } else if (parseInt == 180) {
                    i2 = 3;
                } else if (parseInt == 270) {
                    i2 = 8;
                }
                this.T[0].put("Orientation", c.d(i2, this.V));
            }
            if (extractMetadata != null && extractMetadata2 != null) {
                int parseInt2 = Integer.parseInt(extractMetadata);
                int parseInt3 = Integer.parseInt(extractMetadata2);
                if (parseInt3 > 6) {
                    bVar.C(parseInt2);
                    byte[] bArr = new byte[6];
                    if (bVar.read(bArr) == 6) {
                        int i3 = parseInt2 + 6;
                        int i4 = parseInt3 - 6;
                        if (Arrays.equals(bArr, N)) {
                            byte[] bArr2 = new byte[i4];
                            if (bVar.read(bArr2) == i4) {
                                this.a0 = i3;
                                u(bArr2, 0);
                            } else {
                                throw new IOException("Can't read exif");
                            }
                        } else {
                            throw new IOException("Invalid identifier");
                        }
                    } else {
                        throw new IOException("Can't read identifier");
                    }
                } else {
                    throw new IOException("Invalid exif length");
                }
            }
            if (f2370a) {
                Log.d("ExifInterface", "Heif meta: " + str3 + "x" + str + ", rotation " + str2);
            }
        } finally {
            mediaMetadataRetriever.release();
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:66:0x0185, code lost:
        r18.f2383e = r17.V;
     */
    /* JADX WARN: Code restructure failed: missing block: B:67:0x0189, code lost:
        return;
     */
    /* JADX WARN: Removed duplicated region for block: B:34:0x00b7 A[FALL_THROUGH] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void g(b bVar, int i2, int i3) {
        if (f2370a) {
            Log.d("ExifInterface", "getJpegAttributes starting with: " + bVar);
        }
        bVar.f2383e = ByteOrder.BIG_ENDIAN;
        bVar.C(i2);
        byte readByte = bVar.readByte();
        byte b2 = -1;
        if (readByte == -1) {
            int i4 = 1;
            int i5 = i2 + 1;
            if (bVar.readByte() != -40) {
                StringBuilder x2 = c.b.a.a.a.x("Invalid marker: ");
                x2.append(Integer.toHexString(readByte & UnsignedBytes.MAX_VALUE));
                throw new IOException(x2.toString());
            }
            int i6 = i5 + 1;
            while (true) {
                byte readByte2 = bVar.readByte();
                if (readByte2 == b2) {
                    int i7 = i6 + i4;
                    byte readByte3 = bVar.readByte();
                    boolean z2 = f2370a;
                    if (z2) {
                        StringBuilder x3 = c.b.a.a.a.x("Found JPEG segment indicator: ");
                        x3.append(Integer.toHexString(readByte3 & UnsignedBytes.MAX_VALUE));
                        Log.d("ExifInterface", x3.toString());
                    }
                    int i8 = i7 + i4;
                    if (readByte3 != -39 && readByte3 != -38) {
                        int readUnsignedShort = bVar.readUnsignedShort() - 2;
                        int i9 = i8 + 2;
                        if (z2) {
                            StringBuilder x4 = c.b.a.a.a.x("JPEG segment: ");
                            x4.append(Integer.toHexString(readByte3 & UnsignedBytes.MAX_VALUE));
                            x4.append(" (length: ");
                            x4.append(readUnsignedShort + 2);
                            x4.append(")");
                            Log.d("ExifInterface", x4.toString());
                        }
                        if (readUnsignedShort < 0) {
                            throw new IOException("Invalid length");
                        }
                        if (readByte3 == -31) {
                            byte[] bArr = new byte[readUnsignedShort];
                            bVar.readFully(bArr);
                            int i10 = i9 + readUnsignedShort;
                            byte[] bArr2 = N;
                            if (x(bArr, bArr2)) {
                                byte[] copyOfRange = Arrays.copyOfRange(bArr, bArr2.length, readUnsignedShort);
                                this.a0 = i9 + bArr2.length;
                                u(copyOfRange, i3);
                            } else {
                                byte[] bArr3 = O;
                                if (x(bArr, bArr3)) {
                                    int length = i9 + bArr3.length;
                                    byte[] copyOfRange2 = Arrays.copyOfRange(bArr, bArr3.length, readUnsignedShort);
                                    if (d("Xmp") == null) {
                                        this.T[0].put("Xmp", new c(1, copyOfRange2.length, length, copyOfRange2));
                                    }
                                }
                            }
                            readUnsignedShort = 0;
                            i9 = i10;
                        } else if (readByte3 != -2) {
                            switch (readByte3) {
                                default:
                                    switch (readByte3) {
                                        default:
                                            switch (readByte3) {
                                                default:
                                                    switch (readByte3) {
                                                    }
                                                case -55:
                                                case -54:
                                                case -53:
                                                    if (bVar.skipBytes(i4) == i4) {
                                                        this.T[i3].put("ImageLength", c.b(bVar.readUnsignedShort(), this.V));
                                                        this.T[i3].put("ImageWidth", c.b(bVar.readUnsignedShort(), this.V));
                                                        readUnsignedShort -= 5;
                                                        break;
                                                    } else {
                                                        throw new IOException("Invalid SOFx");
                                                    }
                                            }
                                        case -59:
                                        case -58:
                                        case -57:
                                            break;
                                    }
                                case -64:
                                case -63:
                                case -62:
                                case -61:
                                    break;
                            }
                        } else {
                            byte[] bArr4 = new byte[readUnsignedShort];
                            if (bVar.read(bArr4) == readUnsignedShort) {
                                if (d("UserComment") == null) {
                                    this.T[i4].put("UserComment", c.a(new String(bArr4, M)));
                                }
                                readUnsignedShort = 0;
                            } else {
                                throw new IOException("Invalid exif");
                            }
                        }
                        if (readUnsignedShort >= 0) {
                            if (bVar.skipBytes(readUnsignedShort) != readUnsignedShort) {
                                throw new IOException("Invalid JPEG segment");
                            }
                            i6 = i9 + readUnsignedShort;
                            b2 = -1;
                            i4 = 1;
                        } else {
                            throw new IOException("Invalid length");
                        }
                    }
                } else {
                    StringBuilder x5 = c.b.a.a.a.x("Invalid marker:");
                    x5.append(Integer.toHexString(readByte2 & UnsignedBytes.MAX_VALUE));
                    throw new IOException(x5.toString());
                }
            }
        } else {
            StringBuilder x6 = c.b.a.a.a.x("Invalid marker: ");
            x6.append(Integer.toHexString(readByte & UnsignedBytes.MAX_VALUE));
            throw new IOException(x6.toString());
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:70:0x00cf, code lost:
        if (r8 != null) goto L29;
     */
    /* JADX WARN: Removed duplicated region for block: B:120:0x0143 A[RETURN] */
    /* JADX WARN: Removed duplicated region for block: B:122:0x0146  */
    /* JADX WARN: Removed duplicated region for block: B:156:0x018f  */
    /* JADX WARN: Removed duplicated region for block: B:163:0x0111 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:98:0x010f A[RETURN] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final int h(BufferedInputStream bufferedInputStream) {
        boolean z2;
        boolean z3;
        b bVar;
        b bVar2;
        boolean z4;
        b bVar3;
        b bVar4;
        boolean z5;
        b bVar5;
        b bVar6;
        boolean z6;
        boolean z7;
        boolean z8;
        b bVar7;
        bufferedInputStream.mark(5000);
        byte[] bArr = new byte[5000];
        bufferedInputStream.read(bArr);
        bufferedInputStream.reset();
        int i2 = 0;
        while (true) {
            byte[] bArr2 = f2375f;
            if (i2 >= bArr2.length) {
                z2 = true;
                break;
            } else if (bArr[i2] != bArr2[i2]) {
                z2 = false;
                break;
            } else {
                i2++;
            }
        }
        if (z2) {
            return 4;
        }
        byte[] bytes = "FUJIFILMCCD-RAW".getBytes(Charset.defaultCharset());
        int i3 = 0;
        while (true) {
            if (i3 >= bytes.length) {
                z3 = true;
                break;
            } else if (bArr[i3] != bytes[i3]) {
                z3 = false;
                break;
            } else {
                i3++;
            }
        }
        if (z3) {
            return 9;
        }
        try {
            bVar2 = new b(bArr);
        } catch (Exception e2) {
            e = e2;
            bVar2 = null;
        } catch (Throwable th) {
            th = th;
            bVar = null;
            bVar2 = bVar;
            if (bVar2 != null) {
            }
            throw th;
        }
        try {
            long readInt = bVar2.readInt();
            byte[] bArr3 = new byte[4];
            bVar2.read(bArr3);
            if (Arrays.equals(bArr3, f2376g)) {
                long j2 = 16;
                if (readInt == 1) {
                    readInt = bVar2.readLong();
                    if (readInt < 16) {
                    }
                } else {
                    j2 = 8;
                }
                long j3 = 5000;
                if (readInt > j3) {
                    readInt = j3;
                }
                long j4 = readInt - j2;
                if (j4 >= 8) {
                    byte[] bArr4 = new byte[4];
                    boolean z9 = false;
                    boolean z10 = false;
                    for (long j5 = 0; j5 < j4 / 4 && bVar2.read(bArr4) == 4; j5++) {
                        if (j5 != 1) {
                            if (Arrays.equals(bArr4, f2377h)) {
                                z9 = true;
                            } else if (Arrays.equals(bArr4, i)) {
                                z10 = true;
                            }
                            if (z9 && z10) {
                                bVar2.close();
                                z4 = true;
                                break;
                            }
                        }
                    }
                }
            }
        } catch (Exception e3) {
            e = e3;
            try {
                if (f2370a) {
                    Log.d("ExifInterface", "Exception parsing HEIF file type box.", e);
                }
            } catch (Throwable th2) {
                th = th2;
                bVar = bVar2;
                bVar2 = bVar;
                if (bVar2 != null) {
                    bVar2.close();
                }
                throw th;
            }
        } catch (Throwable th3) {
            th = th3;
            if (bVar2 != null) {
            }
            throw th;
        }
        bVar2.close();
        z4 = false;
        if (z4) {
            return 12;
        }
        try {
            bVar4 = new b(bArr);
        } catch (Exception unused) {
            bVar4 = null;
        } catch (Throwable th4) {
            th = th4;
            bVar3 = null;
        }
        try {
            ByteOrder t2 = t(bVar4);
            this.V = t2;
            bVar4.f2383e = t2;
            short readShort = bVar4.readShort();
            z5 = readShort == 20306 || readShort == 21330;
            bVar4.close();
        } catch (Exception unused2) {
            if (bVar4 != null) {
                bVar4.close();
            }
            z5 = false;
            if (z5) {
            }
        } catch (Throwable th5) {
            th = th5;
            bVar3 = bVar4;
            if (bVar3 != null) {
                bVar3.close();
            }
            throw th;
        }
        if (z5) {
            try {
                bVar7 = new b(bArr);
            } catch (Exception unused3) {
                bVar6 = null;
            } catch (Throwable th6) {
                th = th6;
                bVar5 = null;
            }
            try {
                ByteOrder t3 = t(bVar7);
                this.V = t3;
                bVar7.f2383e = t3;
                z6 = bVar7.readShort() == 85;
                bVar7.close();
            } catch (Exception unused4) {
                bVar6 = bVar7;
                if (bVar6 != null) {
                    bVar6.close();
                }
                z6 = false;
                if (z6) {
                }
            } catch (Throwable th7) {
                th = th7;
                bVar5 = bVar7;
                if (bVar5 != null) {
                    bVar5.close();
                }
                throw th;
            }
            if (z6) {
                int i4 = 0;
                while (true) {
                    byte[] bArr5 = l;
                    if (i4 >= bArr5.length) {
                        z7 = true;
                        break;
                    } else if (bArr[i4] != bArr5[i4]) {
                        z7 = false;
                        break;
                    } else {
                        i4++;
                    }
                }
                if (z7) {
                    return 13;
                }
                int i5 = 0;
                while (true) {
                    byte[] bArr6 = p;
                    if (i5 >= bArr6.length) {
                        int i6 = 0;
                        while (true) {
                            byte[] bArr7 = q;
                            if (i6 >= bArr7.length) {
                                z8 = true;
                                break;
                            } else if (bArr[p.length + i6 + 4] != bArr7[i6]) {
                                break;
                            } else {
                                i6++;
                            }
                        }
                    } else if (bArr[i5] != bArr6[i5]) {
                        break;
                    } else {
                        i5++;
                    }
                }
                z8 = false;
                return z8 ? 14 : 0;
            }
            return 10;
        }
        return 7;
    }

    public final void i(b bVar) {
        l(bVar);
        c cVar = this.T[1].get("MakerNote");
        if (cVar != null) {
            b bVar2 = new b(cVar.f2388c);
            bVar2.f2383e = this.V;
            byte[] bArr = j;
            byte[] bArr2 = new byte[bArr.length];
            bVar2.readFully(bArr2);
            bVar2.C(0L);
            byte[] bArr3 = k;
            byte[] bArr4 = new byte[bArr3.length];
            bVar2.readFully(bArr4);
            if (Arrays.equals(bArr2, bArr)) {
                bVar2.C(8L);
            } else if (Arrays.equals(bArr4, bArr3)) {
                bVar2.C(12L);
            }
            v(bVar2, 6);
            c cVar2 = this.T[7].get("PreviewImageStart");
            c cVar3 = this.T[7].get("PreviewImageLength");
            if (cVar2 != null && cVar3 != null) {
                this.T[5].put("JPEGInterchangeFormat", cVar2);
                this.T[5].put("JPEGInterchangeFormatLength", cVar3);
            }
            c cVar4 = this.T[8].get("AspectFrame");
            if (cVar4 != null) {
                int[] iArr = (int[]) cVar4.h(this.V);
                if (iArr != null && iArr.length == 4) {
                    if (iArr[2] <= iArr[0] || iArr[3] <= iArr[1]) {
                        return;
                    }
                    int i2 = (iArr[2] - iArr[0]) + 1;
                    int i3 = (iArr[3] - iArr[1]) + 1;
                    if (i2 < i3) {
                        int i4 = i2 + i3;
                        i3 = i4 - i3;
                        i2 = i4 - i3;
                    }
                    c d2 = c.d(i2, this.V);
                    c d3 = c.d(i3, this.V);
                    this.T[0].put("ImageWidth", d2);
                    this.T[0].put("ImageLength", d3);
                    return;
                }
                StringBuilder x2 = c.b.a.a.a.x("Invalid aspect frame values. frame=");
                x2.append(Arrays.toString(iArr));
                Log.w("ExifInterface", x2.toString());
            }
        }
    }

    public final void j(b bVar) {
        if (f2370a) {
            Log.d("ExifInterface", "getPngAttributes starting with: " + bVar);
        }
        bVar.f2383e = ByteOrder.BIG_ENDIAN;
        byte[] bArr = l;
        bVar.skipBytes(bArr.length);
        int length = bArr.length + 0;
        while (true) {
            try {
                int readInt = bVar.readInt();
                int i2 = length + 4;
                byte[] bArr2 = new byte[4];
                if (bVar.read(bArr2) == 4) {
                    int i3 = i2 + 4;
                    if (i3 == 16 && !Arrays.equals(bArr2, n)) {
                        throw new IOException("Encountered invalid PNG file--IHDR chunk should appearas the first chunk");
                    }
                    if (Arrays.equals(bArr2, o)) {
                        return;
                    }
                    if (Arrays.equals(bArr2, m)) {
                        byte[] bArr3 = new byte[readInt];
                        if (bVar.read(bArr3) == readInt) {
                            int readInt2 = bVar.readInt();
                            CRC32 crc32 = new CRC32();
                            crc32.update(bArr2);
                            crc32.update(bArr3);
                            if (((int) crc32.getValue()) == readInt2) {
                                this.a0 = i3;
                                u(bArr3, 0);
                                A();
                                return;
                            }
                            throw new IOException("Encountered invalid CRC value for PNG-EXIF chunk.\n recorded CRC value: " + readInt2 + ", calculated CRC value: " + crc32.getValue());
                        }
                        throw new IOException("Failed to read given length for given PNG chunk type: " + b(bArr2));
                    }
                    int i4 = readInt + 4;
                    bVar.skipBytes(i4);
                    length = i3 + i4;
                } else {
                    throw new IOException("Encountered invalid length while parsing PNG chunktype");
                }
            } catch (EOFException unused) {
                throw new IOException("Encountered corrupt PNG file.");
            }
        }
    }

    public final void k(b bVar) {
        bVar.skipBytes(84);
        byte[] bArr = new byte[4];
        byte[] bArr2 = new byte[4];
        bVar.read(bArr);
        bVar.skipBytes(4);
        bVar.read(bArr2);
        int i2 = ByteBuffer.wrap(bArr).getInt();
        int i3 = ByteBuffer.wrap(bArr2).getInt();
        g(bVar, i2, 5);
        bVar.C(i3);
        bVar.f2383e = ByteOrder.BIG_ENDIAN;
        int readInt = bVar.readInt();
        if (f2370a) {
            c.b.a.a.a.L("numberOfDirectoryEntry: ", readInt, "ExifInterface");
        }
        for (int i4 = 0; i4 < readInt; i4++) {
            int readUnsignedShort = bVar.readUnsignedShort();
            int readUnsignedShort2 = bVar.readUnsignedShort();
            if (readUnsignedShort == B.f2389a) {
                short readShort = bVar.readShort();
                short readShort2 = bVar.readShort();
                c d2 = c.d(readShort, this.V);
                c d3 = c.d(readShort2, this.V);
                this.T[0].put("ImageLength", d2);
                this.T[0].put("ImageWidth", d3);
                if (f2370a) {
                    Log.d("ExifInterface", "Updated to length: " + ((int) readShort) + ", width: " + ((int) readShort2));
                    return;
                }
                return;
            }
            bVar.skipBytes(readUnsignedShort2);
        }
    }

    public final void l(b bVar) {
        c cVar;
        r(bVar, bVar.available());
        v(bVar, 0);
        z(bVar, 0);
        z(bVar, 5);
        z(bVar, 4);
        A();
        if (this.R != 8 || (cVar = this.T[1].get("MakerNote")) == null) {
            return;
        }
        b bVar2 = new b(cVar.f2388c);
        bVar2.f2383e = this.V;
        bVar2.C(6L);
        v(bVar2, 9);
        c cVar2 = this.T[9].get("ColorSpace");
        if (cVar2 != null) {
            this.T[1].put("ColorSpace", cVar2);
        }
    }

    public final void m(b bVar) {
        l(bVar);
        if (this.T[0].get("JpgFromRaw") != null) {
            g(bVar, this.e0, 5);
        }
        c cVar = this.T[0].get("ISO");
        c cVar2 = this.T[1].get("PhotographicSensitivity");
        if (cVar == null || cVar2 != null) {
            return;
        }
        this.T[1].put("PhotographicSensitivity", cVar);
    }

    public final void n(b bVar) {
        byte[] bArr = N;
        bVar.skipBytes(bArr.length);
        byte[] bArr2 = new byte[bVar.available()];
        bVar.readFully(bArr2);
        this.a0 = bArr.length;
        u(bArr2, 0);
    }

    public final void o(b bVar) {
        if (f2370a) {
            Log.d("ExifInterface", "getWebpAttributes starting with: " + bVar);
        }
        bVar.f2383e = ByteOrder.LITTLE_ENDIAN;
        bVar.skipBytes(p.length);
        int readInt = bVar.readInt() + 8;
        int skipBytes = bVar.skipBytes(q.length) + 8;
        while (true) {
            try {
                byte[] bArr = new byte[4];
                if (bVar.read(bArr) == 4) {
                    int readInt2 = bVar.readInt();
                    int i2 = skipBytes + 4 + 4;
                    if (Arrays.equals(r, bArr)) {
                        byte[] bArr2 = new byte[readInt2];
                        if (bVar.read(bArr2) == readInt2) {
                            this.a0 = i2;
                            u(bArr2, 0);
                            this.a0 = i2;
                            return;
                        }
                        throw new IOException("Failed to read given length for given PNG chunk type: " + b(bArr));
                    }
                    if (readInt2 % 2 == 1) {
                        readInt2++;
                    }
                    int i3 = i2 + readInt2;
                    if (i3 == readInt) {
                        return;
                    }
                    if (i3 <= readInt) {
                        int skipBytes2 = bVar.skipBytes(readInt2);
                        if (skipBytes2 != readInt2) {
                            throw new IOException("Encountered WebP file with invalid chunk size");
                        }
                        skipBytes = i2 + skipBytes2;
                    } else {
                        throw new IOException("Encountered WebP file with invalid chunk size");
                    }
                } else {
                    throw new IOException("Encountered invalid length while parsing WebP chunktype");
                }
            } catch (EOFException unused) {
                throw new IOException("Encountered corrupt WebP file.");
            }
        }
    }

    public final void p(b bVar, HashMap hashMap) {
        c cVar = (c) hashMap.get("JPEGInterchangeFormat");
        c cVar2 = (c) hashMap.get("JPEGInterchangeFormatLength");
        if (cVar == null || cVar2 == null) {
            return;
        }
        int f2 = cVar.f(this.V);
        int f3 = cVar2.f(this.V);
        if (this.R == 7) {
            f2 += this.b0;
        }
        int min = Math.min(f3, bVar.f2384f - f2);
        if (f2 > 0 && min > 0) {
            int i2 = this.a0 + f2;
            this.X = i2;
            this.Y = min;
            if (this.Q == null && this.P == null) {
                bVar.C(i2);
                bVar.readFully(new byte[min]);
            }
        }
        if (f2370a) {
            Log.d("ExifInterface", "Setting thumbnail attributes with offset: " + f2 + ", length: " + min);
        }
    }

    public final boolean q(HashMap hashMap) {
        c cVar = (c) hashMap.get("ImageLength");
        c cVar2 = (c) hashMap.get("ImageWidth");
        if (cVar == null || cVar2 == null) {
            return false;
        }
        return cVar.f(this.V) <= 512 && cVar2.f(this.V) <= 512;
    }

    public final void r(b bVar, int i2) {
        ByteOrder t2 = t(bVar);
        this.V = t2;
        bVar.f2383e = t2;
        int readUnsignedShort = bVar.readUnsignedShort();
        int i3 = this.R;
        if (i3 != 7 && i3 != 10 && readUnsignedShort != 42) {
            StringBuilder x2 = c.b.a.a.a.x("Invalid start code: ");
            x2.append(Integer.toHexString(readUnsignedShort));
            throw new IOException(x2.toString());
        }
        int readInt = bVar.readInt();
        if (readInt >= 8 && readInt < i2) {
            int i4 = readInt - 8;
            if (i4 > 0 && bVar.skipBytes(i4) != i4) {
                throw new IOException(c.b.a.a.a.j("Couldn't jump to first Ifd: ", i4));
            }
            return;
        }
        throw new IOException(c.b.a.a.a.j("Invalid first Ifd offset: ", readInt));
    }

    public final void s() {
        for (int i2 = 0; i2 < this.T.length; i2++) {
            StringBuilder y2 = c.b.a.a.a.y("The size of tag group[", i2, "]: ");
            y2.append(this.T[i2].size());
            Log.d("ExifInterface", y2.toString());
            for (Map.Entry<String, c> entry : this.T[i2].entrySet()) {
                c value = entry.getValue();
                StringBuilder x2 = c.b.a.a.a.x("tagName: ");
                x2.append(entry.getKey());
                x2.append(", tagType: ");
                x2.append(value.toString());
                x2.append(", tagValue: '");
                x2.append(value.g(this.V));
                x2.append("'");
                Log.d("ExifInterface", x2.toString());
            }
        }
    }

    public final ByteOrder t(b bVar) {
        short readShort = bVar.readShort();
        if (readShort == 18761) {
            if (f2370a) {
                Log.d("ExifInterface", "readExifSegment: Byte Align II");
            }
            return ByteOrder.LITTLE_ENDIAN;
        } else if (readShort == 19789) {
            if (f2370a) {
                Log.d("ExifInterface", "readExifSegment: Byte Align MM");
            }
            return ByteOrder.BIG_ENDIAN;
        } else {
            StringBuilder x2 = c.b.a.a.a.x("Invalid byte order: ");
            x2.append(Integer.toHexString(readShort));
            throw new IOException(x2.toString());
        }
    }

    public final void u(byte[] bArr, int i2) {
        b bVar = new b(bArr);
        r(bVar, bArr.length);
        v(bVar, i2);
    }

    /* JADX WARN: Removed duplicated region for block: B:133:0x0245  */
    /* JADX WARN: Removed duplicated region for block: B:144:0x02a9  */
    /* JADX WARN: Removed duplicated region for block: B:61:0x00de  */
    /* JADX WARN: Removed duplicated region for block: B:64:0x0101  */
    /* JADX WARN: Removed duplicated region for block: B:81:0x0136  */
    /* JADX WARN: Removed duplicated region for block: B:82:0x013b  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void v(b bVar, int i2) {
        short s2;
        short s3;
        int[] iArr;
        boolean z2;
        long j2;
        boolean z3;
        int i3;
        int i4;
        int i5;
        d dVar;
        String str;
        int readUnsignedShort;
        long j3;
        this.U.add(Integer.valueOf(bVar.f2385g));
        if (bVar.f2385g + 2 > bVar.f2384f) {
            return;
        }
        short readShort = bVar.readShort();
        String str2 = "ExifInterface";
        if (f2370a) {
            c.b.a.a.a.L("numberOfDirectoryEntry: ", readShort, "ExifInterface");
        }
        if ((readShort * 12) + bVar.f2385g > bVar.f2384f || readShort <= 0) {
            return;
        }
        short s4 = 0;
        char c2 = 0;
        while (s4 < readShort) {
            int readUnsignedShort2 = bVar.readUnsignedShort();
            int readUnsignedShort3 = bVar.readUnsignedShort();
            int readInt = bVar.readInt();
            long j4 = bVar.f2385g + 4;
            d dVar2 = I[i2].get(Integer.valueOf(readUnsignedShort2));
            boolean z4 = f2370a;
            if (z4) {
                Object[] objArr = new Object[5];
                objArr[c2] = Integer.valueOf(i2);
                objArr[1] = Integer.valueOf(readUnsignedShort2);
                objArr[2] = dVar2 != null ? dVar2.f2390b : null;
                objArr[3] = Integer.valueOf(readUnsignedShort3);
                objArr[4] = Integer.valueOf(readInt);
                Log.d(str2, String.format("ifdType: %d, tagNumber: %d, tagName: %s, dataFormat: %d, numberOfComponents: %d", objArr));
            }
            if (dVar2 == null) {
                if (z4) {
                    c.b.a.a.a.L("Skip the tag entry since tag number is not defined: ", readUnsignedShort2, str2);
                }
                s2 = readShort;
            } else {
                if (readUnsignedShort3 > 0) {
                    if (readUnsignedShort3 < u.length) {
                        int i6 = dVar2.f2391c;
                        if (i6 == 7 || readUnsignedShort3 == 7 || i6 == readUnsignedShort3 || (i3 = dVar2.f2392d) == readUnsignedShort3) {
                            s2 = readShort;
                        } else {
                            s2 = readShort;
                            if (((i6 != 4 && i3 != 4) || readUnsignedShort3 != 3) && (((i6 != 9 && i3 != 9) || readUnsignedShort3 != 8) && ((i6 != 12 && i3 != 12) || readUnsignedShort3 != 11))) {
                                z2 = false;
                                if (!z2) {
                                    short s5 = s4;
                                    if (readUnsignedShort3 == 7) {
                                        readUnsignedShort3 = i6;
                                    }
                                    s3 = s5;
                                    j2 = readInt * iArr[readUnsignedShort3];
                                    if (j2 < 0 || j2 > 2147483647L) {
                                        if (z4) {
                                            c.b.a.a.a.L("Skip the tag entry since the number of components is invalid: ", readInt, str2);
                                        }
                                        z3 = false;
                                    } else {
                                        z3 = true;
                                    }
                                    if (!z3) {
                                        bVar.C(j4);
                                    } else {
                                        if (j2 > 4) {
                                            int readInt2 = bVar.readInt();
                                            if (z4) {
                                                c.b.a.a.a.L("seek to data offset: ", readInt2, str2);
                                            }
                                            int i7 = this.R;
                                            i4 = readUnsignedShort3;
                                            if (i7 == 7) {
                                                if ("MakerNote".equals(dVar2.f2390b)) {
                                                    this.b0 = readInt2;
                                                } else if (i2 == 6 && "ThumbnailImage".equals(dVar2.f2390b)) {
                                                    this.c0 = readInt2;
                                                    this.d0 = readInt;
                                                    c d2 = c.d(6, this.V);
                                                    j3 = j4;
                                                    c b2 = c.b(this.c0, this.V);
                                                    i5 = readInt;
                                                    c b3 = c.b(this.d0, this.V);
                                                    this.T[4].put("Compression", d2);
                                                    this.T[4].put("JPEGInterchangeFormat", b2);
                                                    this.T[4].put("JPEGInterchangeFormatLength", b3);
                                                }
                                                j3 = j4;
                                                i5 = readInt;
                                            } else {
                                                j3 = j4;
                                                i5 = readInt;
                                                if (i7 == 10 && "JpgFromRaw".equals(dVar2.f2390b)) {
                                                    this.e0 = readInt2;
                                                }
                                            }
                                            long j5 = readInt2;
                                            dVar = dVar2;
                                            if (j5 + j2 <= bVar.f2384f) {
                                                bVar.C(j5);
                                                j4 = j3;
                                            } else {
                                                if (z4) {
                                                    c.b.a.a.a.L("Skip the tag entry since data offset is invalid: ", readInt2, str2);
                                                }
                                                bVar.C(j3);
                                            }
                                        } else {
                                            i4 = readUnsignedShort3;
                                            i5 = readInt;
                                            dVar = dVar2;
                                        }
                                        Integer num = L.get(Integer.valueOf(readUnsignedShort2));
                                        if (z4) {
                                            Log.d(str2, "nextIfdType: " + num + " byteCount: " + j2);
                                        }
                                        if (num != null) {
                                            long j6 = -1;
                                            int i8 = i4;
                                            if (i8 == 3) {
                                                readUnsignedShort = bVar.readUnsignedShort();
                                            } else {
                                                if (i8 == 4) {
                                                    j6 = bVar.B();
                                                } else if (i8 == 8) {
                                                    readUnsignedShort = bVar.readShort();
                                                } else if (i8 == 9 || i8 == 13) {
                                                    readUnsignedShort = bVar.readInt();
                                                }
                                                if (z4) {
                                                    Log.d(str2, String.format("Offset: %d, tagName: %s", Long.valueOf(j6), dVar.f2390b));
                                                }
                                                if (j6 > 0 || j6 >= bVar.f2384f) {
                                                    if (z4) {
                                                        Log.d(str2, "Skip jump into the IFD since its offset is invalid: " + j6);
                                                    }
                                                } else if (!this.U.contains(Integer.valueOf((int) j6))) {
                                                    bVar.C(j6);
                                                    v(bVar, num.intValue());
                                                } else if (z4) {
                                                    Log.d(str2, "Skip jump into the IFD since it has already been read: IfdType " + num + " (at " + j6 + ")");
                                                }
                                                bVar.C(j4);
                                            }
                                            j6 = readUnsignedShort;
                                            if (z4) {
                                            }
                                            if (j6 > 0) {
                                            }
                                            if (z4) {
                                            }
                                            bVar.C(j4);
                                        } else {
                                            d dVar3 = dVar;
                                            byte[] bArr = new byte[(int) j2];
                                            bVar.readFully(bArr);
                                            long j7 = j4;
                                            str = str2;
                                            c cVar = new c(i4, i5, bVar.f2385g + this.a0, bArr);
                                            this.T[i2].put(dVar3.f2390b, cVar);
                                            if ("DNGVersion".equals(dVar3.f2390b)) {
                                                this.R = 3;
                                            }
                                            if ((("Make".equals(dVar3.f2390b) || "Model".equals(dVar3.f2390b)) && cVar.g(this.V).contains("PENTAX")) || ("Compression".equals(dVar3.f2390b) && cVar.f(this.V) == 65535)) {
                                                this.R = 8;
                                            }
                                            if (bVar.f2385g != j7) {
                                                bVar.C(j7);
                                            }
                                            s4 = (short) (s3 + 1);
                                            str2 = str;
                                            c2 = 0;
                                            readShort = s2;
                                        }
                                    }
                                    str = str2;
                                    s4 = (short) (s3 + 1);
                                    str2 = str;
                                    c2 = 0;
                                    readShort = s2;
                                } else if (z4) {
                                    StringBuilder x2 = c.b.a.a.a.x("Skip the tag entry since data format (");
                                    x2.append(t[readUnsignedShort3]);
                                    x2.append(") is unexpected for tag: ");
                                    x2.append(dVar2.f2390b);
                                    Log.d(str2, x2.toString());
                                }
                            }
                        }
                        z2 = true;
                        if (!z2) {
                        }
                    }
                }
                s2 = readShort;
                s3 = s4;
                if (z4) {
                    c.b.a.a.a.L("Skip the tag entry since data format is invalid: ", readUnsignedShort3, str2);
                }
                z3 = false;
                j2 = 0;
                if (!z3) {
                }
                str = str2;
                s4 = (short) (s3 + 1);
                str2 = str;
                c2 = 0;
                readShort = s2;
            }
            s3 = s4;
            z3 = false;
            j2 = 0;
            if (!z3) {
            }
            str = str2;
            s4 = (short) (s3 + 1);
            str2 = str;
            c2 = 0;
            readShort = s2;
        }
        String str3 = str2;
        if (bVar.f2385g + 4 <= bVar.f2384f) {
            int readInt3 = bVar.readInt();
            boolean z5 = f2370a;
            if (z5) {
                Log.d(str3, String.format("nextIfdOffset: %d", Integer.valueOf(readInt3)));
            }
            long j8 = readInt3;
            if (j8 <= 0 || readInt3 >= bVar.f2384f) {
                if (z5) {
                    c.b.a.a.a.L("Stop reading file since a wrong offset may cause an infinite loop: ", readInt3, str3);
                }
            } else if (this.U.contains(Integer.valueOf(readInt3))) {
                if (z5) {
                    c.b.a.a.a.L("Stop reading file since re-reading an IFD may cause an infinite loop: ", readInt3, str3);
                }
            } else {
                bVar.C(j8);
                if (this.T[4].isEmpty()) {
                    v(bVar, 4);
                } else if (this.T[5].isEmpty()) {
                    v(bVar, 5);
                }
            }
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:33:0x007e  */
    /* JADX WARN: Removed duplicated region for block: B:75:? A[RETURN, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void w(b bVar) {
        boolean z2;
        c cVar;
        int f2;
        HashMap<String, c> hashMap = this.T[4];
        c cVar2 = hashMap.get("Compression");
        if (cVar2 != null) {
            int f3 = cVar2.f(this.V);
            this.Z = f3;
            if (f3 != 1) {
                if (f3 == 6) {
                    p(bVar, hashMap);
                    return;
                } else if (f3 != 7) {
                    return;
                }
            }
            c cVar3 = hashMap.get("BitsPerSample");
            if (cVar3 != null) {
                int[] iArr = (int[]) cVar3.h(this.V);
                int[] iArr2 = f2373d;
                if (Arrays.equals(iArr2, iArr) || (this.R == 3 && (cVar = hashMap.get("PhotometricInterpretation")) != null && (((f2 = cVar.f(this.V)) == 1 && Arrays.equals(iArr, f2374e)) || (f2 == 6 && Arrays.equals(iArr, iArr2))))) {
                    z2 = true;
                    if (z2) {
                        return;
                    }
                    c cVar4 = hashMap.get("StripOffsets");
                    c cVar5 = hashMap.get("StripByteCounts");
                    if (cVar4 == null || cVar5 == null) {
                        return;
                    }
                    long[] c2 = c(cVar4.h(this.V));
                    long[] c3 = c(cVar5.h(this.V));
                    if (c2 != null && c2.length != 0) {
                        if (c3 != null && c3.length != 0) {
                            if (c2.length != c3.length) {
                                Log.w("ExifInterface", "stripOffsets and stripByteCounts should have same length.");
                                return;
                            }
                            long j2 = 0;
                            for (long j3 : c3) {
                                j2 += j3;
                            }
                            int i2 = (int) j2;
                            byte[] bArr = new byte[i2];
                            this.W = true;
                            int i3 = 0;
                            int i4 = 0;
                            for (int i5 = 0; i5 < c2.length; i5++) {
                                int i6 = (int) c2[i5];
                                int i7 = (int) c3[i5];
                                if (i5 < c2.length - 1 && i6 + i7 != c2[i5 + 1]) {
                                    this.W = false;
                                }
                                int i8 = i6 - i3;
                                if (i8 < 0) {
                                    Log.d("ExifInterface", "Invalid strip offset value");
                                }
                                bVar.C(i8);
                                int i9 = i3 + i8;
                                byte[] bArr2 = new byte[i7];
                                bVar.read(bArr2);
                                i3 = i9 + i7;
                                System.arraycopy(bArr2, 0, bArr, i4, i7);
                                i4 += i7;
                            }
                            if (this.W) {
                                this.X = ((int) c2[0]) + this.a0;
                                this.Y = i2;
                                return;
                            }
                            return;
                        }
                        Log.w("ExifInterface", "stripByteCounts should not be null or have zero length.");
                        return;
                    }
                    Log.w("ExifInterface", "stripOffsets should not be null or have zero length.");
                    return;
                }
            }
            if (f2370a) {
                Log.d("ExifInterface", "Unsupported data type value");
            }
            z2 = false;
            if (z2) {
            }
        } else {
            this.Z = 6;
            p(bVar, hashMap);
        }
    }

    public final void y(int i2, int i3) {
        if (!this.T[i2].isEmpty() && !this.T[i3].isEmpty()) {
            c cVar = this.T[i2].get("ImageLength");
            c cVar2 = this.T[i2].get("ImageWidth");
            c cVar3 = this.T[i3].get("ImageLength");
            c cVar4 = this.T[i3].get("ImageWidth");
            if (cVar == null || cVar2 == null) {
                if (f2370a) {
                    Log.d("ExifInterface", "First image does not contain valid size information");
                }
            } else if (cVar3 != null && cVar4 != null) {
                int f2 = cVar.f(this.V);
                int f3 = cVar2.f(this.V);
                int f4 = cVar3.f(this.V);
                int f5 = cVar4.f(this.V);
                if (f2 >= f4 || f3 >= f5) {
                    return;
                }
                HashMap<String, c>[] hashMapArr = this.T;
                HashMap<String, c> hashMap = hashMapArr[i2];
                hashMapArr[i2] = hashMapArr[i3];
                hashMapArr[i3] = hashMap;
            } else if (f2370a) {
                Log.d("ExifInterface", "Second image does not contain valid size information");
            }
        } else if (f2370a) {
            Log.d("ExifInterface", "Cannot perform swap since only one image data exists");
        }
    }

    public final void z(b bVar, int i2) {
        c cVar;
        c d2;
        c d3;
        c cVar2 = this.T[i2].get("DefaultCropSize");
        c cVar3 = this.T[i2].get("SensorTopBorder");
        c cVar4 = this.T[i2].get("SensorLeftBorder");
        c cVar5 = this.T[i2].get("SensorBottomBorder");
        c cVar6 = this.T[i2].get("SensorRightBorder");
        if (cVar2 != null) {
            if (cVar2.f2386a == 5) {
                e[] eVarArr = (e[]) cVar2.h(this.V);
                if (eVarArr != null && eVarArr.length == 2) {
                    d2 = c.c(eVarArr[0], this.V);
                    d3 = c.c(eVarArr[1], this.V);
                } else {
                    StringBuilder x2 = c.b.a.a.a.x("Invalid crop size values. cropSize=");
                    x2.append(Arrays.toString(eVarArr));
                    Log.w("ExifInterface", x2.toString());
                    return;
                }
            } else {
                int[] iArr = (int[]) cVar2.h(this.V);
                if (iArr != null && iArr.length == 2) {
                    d2 = c.d(iArr[0], this.V);
                    d3 = c.d(iArr[1], this.V);
                } else {
                    StringBuilder x3 = c.b.a.a.a.x("Invalid crop size values. cropSize=");
                    x3.append(Arrays.toString(iArr));
                    Log.w("ExifInterface", x3.toString());
                    return;
                }
            }
            this.T[i2].put("ImageWidth", d2);
            this.T[i2].put("ImageLength", d3);
        } else if (cVar3 != null && cVar4 != null && cVar5 != null && cVar6 != null) {
            int f2 = cVar3.f(this.V);
            int f3 = cVar5.f(this.V);
            int f4 = cVar6.f(this.V);
            int f5 = cVar4.f(this.V);
            if (f3 <= f2 || f4 <= f5) {
                return;
            }
            c d4 = c.d(f3 - f2, this.V);
            c d5 = c.d(f4 - f5, this.V);
            this.T[i2].put("ImageLength", d4);
            this.T[i2].put("ImageWidth", d5);
        } else {
            c cVar7 = this.T[i2].get("ImageLength");
            c cVar8 = this.T[i2].get("ImageWidth");
            if ((cVar7 == null || cVar8 == null) && (cVar = this.T[i2].get("JPEGInterchangeFormat")) != null) {
                g(bVar, cVar.f(this.V), i2);
            }
        }
    }

    /* compiled from: ExifInterface.java */
    /* loaded from: classes.dex */
    public static class b extends InputStream implements DataInput {

        /* renamed from: b  reason: collision with root package name */
        public static final ByteOrder f2380b = ByteOrder.LITTLE_ENDIAN;

        /* renamed from: c  reason: collision with root package name */
        public static final ByteOrder f2381c = ByteOrder.BIG_ENDIAN;

        /* renamed from: d  reason: collision with root package name */
        public DataInputStream f2382d;

        /* renamed from: e  reason: collision with root package name */
        public ByteOrder f2383e;

        /* renamed from: f  reason: collision with root package name */
        public final int f2384f;

        /* renamed from: g  reason: collision with root package name */
        public int f2385g;

        public b(InputStream inputStream) {
            ByteOrder byteOrder = ByteOrder.BIG_ENDIAN;
            this.f2383e = byteOrder;
            DataInputStream dataInputStream = new DataInputStream(inputStream);
            this.f2382d = dataInputStream;
            int available = dataInputStream.available();
            this.f2384f = available;
            this.f2385g = 0;
            this.f2382d.mark(available);
            this.f2383e = byteOrder;
        }

        public long B() {
            return readInt() & UnsignedInts.INT_MASK;
        }

        public void C(long j) {
            int i = this.f2385g;
            if (i > j) {
                this.f2385g = 0;
                this.f2382d.reset();
                this.f2382d.mark(this.f2384f);
            } else {
                j -= i;
            }
            int i2 = (int) j;
            if (skipBytes(i2) != i2) {
                throw new IOException("Couldn't seek up to the byteCount");
            }
        }

        @Override // java.io.InputStream
        public int available() {
            return this.f2382d.available();
        }

        @Override // java.io.InputStream
        public int read() {
            this.f2385g++;
            return this.f2382d.read();
        }

        @Override // java.io.DataInput
        public boolean readBoolean() {
            this.f2385g++;
            return this.f2382d.readBoolean();
        }

        @Override // java.io.DataInput
        public byte readByte() {
            int i = this.f2385g + 1;
            this.f2385g = i;
            if (i <= this.f2384f) {
                int read = this.f2382d.read();
                if (read >= 0) {
                    return (byte) read;
                }
                throw new EOFException();
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public char readChar() {
            this.f2385g += 2;
            return this.f2382d.readChar();
        }

        @Override // java.io.DataInput
        public double readDouble() {
            return Double.longBitsToDouble(readLong());
        }

        @Override // java.io.DataInput
        public float readFloat() {
            return Float.intBitsToFloat(readInt());
        }

        @Override // java.io.DataInput
        public void readFully(byte[] bArr, int i, int i2) {
            int i3 = this.f2385g + i2;
            this.f2385g = i3;
            if (i3 <= this.f2384f) {
                if (this.f2382d.read(bArr, i, i2) != i2) {
                    throw new IOException("Couldn't read up to the length of buffer");
                }
                return;
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public int readInt() {
            int i = this.f2385g + 4;
            this.f2385g = i;
            if (i <= this.f2384f) {
                int read = this.f2382d.read();
                int read2 = this.f2382d.read();
                int read3 = this.f2382d.read();
                int read4 = this.f2382d.read();
                if ((read | read2 | read3 | read4) >= 0) {
                    ByteOrder byteOrder = this.f2383e;
                    if (byteOrder == f2380b) {
                        return (read4 << 24) + (read3 << 16) + (read2 << 8) + read;
                    }
                    if (byteOrder == f2381c) {
                        return (read << 24) + (read2 << 16) + (read3 << 8) + read4;
                    }
                    StringBuilder x = c.b.a.a.a.x("Invalid byte order: ");
                    x.append(this.f2383e);
                    throw new IOException(x.toString());
                }
                throw new EOFException();
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public String readLine() {
            Log.d("ExifInterface", "Currently unsupported");
            return null;
        }

        @Override // java.io.DataInput
        public long readLong() {
            int i = this.f2385g + 8;
            this.f2385g = i;
            if (i <= this.f2384f) {
                int read = this.f2382d.read();
                int read2 = this.f2382d.read();
                int read3 = this.f2382d.read();
                int read4 = this.f2382d.read();
                int read5 = this.f2382d.read();
                int read6 = this.f2382d.read();
                int read7 = this.f2382d.read();
                int read8 = this.f2382d.read();
                if ((read | read2 | read3 | read4 | read5 | read6 | read7 | read8) >= 0) {
                    ByteOrder byteOrder = this.f2383e;
                    if (byteOrder == f2380b) {
                        return (read8 << 56) + (read7 << 48) + (read6 << 40) + (read5 << 32) + (read4 << 24) + (read3 << 16) + (read2 << 8) + read;
                    }
                    if (byteOrder == f2381c) {
                        return (read << 56) + (read2 << 48) + (read3 << 40) + (read4 << 32) + (read5 << 24) + (read6 << 16) + (read7 << 8) + read8;
                    }
                    StringBuilder x = c.b.a.a.a.x("Invalid byte order: ");
                    x.append(this.f2383e);
                    throw new IOException(x.toString());
                }
                throw new EOFException();
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public short readShort() {
            int i = this.f2385g + 2;
            this.f2385g = i;
            if (i <= this.f2384f) {
                int read = this.f2382d.read();
                int read2 = this.f2382d.read();
                if ((read | read2) >= 0) {
                    ByteOrder byteOrder = this.f2383e;
                    if (byteOrder == f2380b) {
                        return (short) ((read2 << 8) + read);
                    }
                    if (byteOrder == f2381c) {
                        return (short) ((read << 8) + read2);
                    }
                    StringBuilder x = c.b.a.a.a.x("Invalid byte order: ");
                    x.append(this.f2383e);
                    throw new IOException(x.toString());
                }
                throw new EOFException();
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public String readUTF() {
            this.f2385g += 2;
            return this.f2382d.readUTF();
        }

        @Override // java.io.DataInput
        public int readUnsignedByte() {
            this.f2385g++;
            return this.f2382d.readUnsignedByte();
        }

        @Override // java.io.DataInput
        public int readUnsignedShort() {
            int i = this.f2385g + 2;
            this.f2385g = i;
            if (i <= this.f2384f) {
                int read = this.f2382d.read();
                int read2 = this.f2382d.read();
                if ((read | read2) >= 0) {
                    ByteOrder byteOrder = this.f2383e;
                    if (byteOrder == f2380b) {
                        return (read2 << 8) + read;
                    }
                    if (byteOrder == f2381c) {
                        return (read << 8) + read2;
                    }
                    StringBuilder x = c.b.a.a.a.x("Invalid byte order: ");
                    x.append(this.f2383e);
                    throw new IOException(x.toString());
                }
                throw new EOFException();
            }
            throw new EOFException();
        }

        @Override // java.io.DataInput
        public int skipBytes(int i) {
            int min = Math.min(i, this.f2384f - this.f2385g);
            int i2 = 0;
            while (i2 < min) {
                i2 += this.f2382d.skipBytes(min - i2);
            }
            this.f2385g += i2;
            return i2;
        }

        @Override // java.io.InputStream
        public int read(byte[] bArr, int i, int i2) {
            int read = this.f2382d.read(bArr, i, i2);
            this.f2385g += read;
            return read;
        }

        @Override // java.io.DataInput
        public void readFully(byte[] bArr) {
            int length = this.f2385g + bArr.length;
            this.f2385g = length;
            if (length <= this.f2384f) {
                if (this.f2382d.read(bArr, 0, bArr.length) != bArr.length) {
                    throw new IOException("Couldn't read up to the length of buffer");
                }
                return;
            }
            throw new EOFException();
        }

        public b(byte[] bArr) {
            this(new ByteArrayInputStream(bArr));
        }
    }

    /* compiled from: ExifInterface.java */
    /* loaded from: classes.dex */
    public static class c {

        /* renamed from: a  reason: collision with root package name */
        public final int f2386a;

        /* renamed from: b  reason: collision with root package name */
        public final int f2387b;

        /* renamed from: c  reason: collision with root package name */
        public final byte[] f2388c;

        public c(int i, int i2, long j, byte[] bArr) {
            this.f2386a = i;
            this.f2387b = i2;
            this.f2388c = bArr;
        }

        public static c a(String str) {
            byte[] bytes = (str + (char) 0).getBytes(a.M);
            return new c(2, bytes.length, bytes);
        }

        public static c b(long j, ByteOrder byteOrder) {
            long[] jArr = {j};
            ByteBuffer wrap = ByteBuffer.wrap(new byte[a.u[4] * 1]);
            wrap.order(byteOrder);
            for (int i = 0; i < 1; i++) {
                wrap.putInt((int) jArr[i]);
            }
            return new c(4, 1, wrap.array());
        }

        public static c c(e eVar, ByteOrder byteOrder) {
            e[] eVarArr = {eVar};
            ByteBuffer wrap = ByteBuffer.wrap(new byte[a.u[5] * 1]);
            wrap.order(byteOrder);
            for (int i = 0; i < 1; i++) {
                e eVar2 = eVarArr[i];
                wrap.putInt((int) eVar2.f2393a);
                wrap.putInt((int) eVar2.f2394b);
            }
            return new c(5, 1, wrap.array());
        }

        public static c d(int i, ByteOrder byteOrder) {
            int[] iArr = {i};
            ByteBuffer wrap = ByteBuffer.wrap(new byte[a.u[3] * 1]);
            wrap.order(byteOrder);
            for (int i2 = 0; i2 < 1; i2++) {
                wrap.putShort((short) iArr[i2]);
            }
            return new c(3, 1, wrap.array());
        }

        public double e(ByteOrder byteOrder) {
            Object h2 = h(byteOrder);
            if (h2 != null) {
                if (h2 instanceof String) {
                    return Double.parseDouble((String) h2);
                }
                if (h2 instanceof long[]) {
                    long[] jArr = (long[]) h2;
                    if (jArr.length == 1) {
                        return jArr[0];
                    }
                    throw new NumberFormatException("There are more than one component");
                } else if (h2 instanceof int[]) {
                    int[] iArr = (int[]) h2;
                    if (iArr.length == 1) {
                        return iArr[0];
                    }
                    throw new NumberFormatException("There are more than one component");
                } else if (h2 instanceof double[]) {
                    double[] dArr = (double[]) h2;
                    if (dArr.length == 1) {
                        return dArr[0];
                    }
                    throw new NumberFormatException("There are more than one component");
                } else if (h2 instanceof e[]) {
                    e[] eVarArr = (e[]) h2;
                    if (eVarArr.length == 1) {
                        e eVar = eVarArr[0];
                        return eVar.f2393a / eVar.f2394b;
                    }
                    throw new NumberFormatException("There are more than one component");
                } else {
                    throw new NumberFormatException("Couldn't find a double value");
                }
            }
            throw new NumberFormatException("NULL can't be converted to a double value");
        }

        public int f(ByteOrder byteOrder) {
            Object h2 = h(byteOrder);
            if (h2 != null) {
                if (h2 instanceof String) {
                    return Integer.parseInt((String) h2);
                }
                if (h2 instanceof long[]) {
                    long[] jArr = (long[]) h2;
                    if (jArr.length == 1) {
                        return (int) jArr[0];
                    }
                    throw new NumberFormatException("There are more than one component");
                } else if (h2 instanceof int[]) {
                    int[] iArr = (int[]) h2;
                    if (iArr.length == 1) {
                        return iArr[0];
                    }
                    throw new NumberFormatException("There are more than one component");
                } else {
                    throw new NumberFormatException("Couldn't find a integer value");
                }
            }
            throw new NumberFormatException("NULL can't be converted to a integer value");
        }

        public String g(ByteOrder byteOrder) {
            Object h2 = h(byteOrder);
            if (h2 == null) {
                return null;
            }
            if (h2 instanceof String) {
                return (String) h2;
            }
            StringBuilder sb = new StringBuilder();
            int i = 0;
            if (h2 instanceof long[]) {
                long[] jArr = (long[]) h2;
                while (i < jArr.length) {
                    sb.append(jArr[i]);
                    i++;
                    if (i != jArr.length) {
                        sb.append(",");
                    }
                }
                return sb.toString();
            } else if (h2 instanceof int[]) {
                int[] iArr = (int[]) h2;
                while (i < iArr.length) {
                    sb.append(iArr[i]);
                    i++;
                    if (i != iArr.length) {
                        sb.append(",");
                    }
                }
                return sb.toString();
            } else if (h2 instanceof double[]) {
                double[] dArr = (double[]) h2;
                while (i < dArr.length) {
                    sb.append(dArr[i]);
                    i++;
                    if (i != dArr.length) {
                        sb.append(",");
                    }
                }
                return sb.toString();
            } else if (h2 instanceof e[]) {
                e[] eVarArr = (e[]) h2;
                while (i < eVarArr.length) {
                    sb.append(eVarArr[i].f2393a);
                    sb.append('/');
                    sb.append(eVarArr[i].f2394b);
                    i++;
                    if (i != eVarArr.length) {
                        sb.append(",");
                    }
                }
                return sb.toString();
            } else {
                return null;
            }
        }

        /* JADX WARN: Not initialized variable reg: 3, insn: 0x019f: MOVE  (r2 I:??[OBJECT, ARRAY]) = (r3 I:??[OBJECT, ARRAY]), block:B:152:0x019f */
        /* JADX WARN: Removed duplicated region for block: B:174:0x01a2 A[EXC_TOP_SPLITTER, SYNTHETIC] */
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public Object h(ByteOrder byteOrder) {
            b bVar;
            InputStream inputStream;
            byte b2;
            byte[] bArr;
            InputStream inputStream2 = null;
            try {
                try {
                    bVar = new b(this.f2388c);
                } catch (IOException e2) {
                    e = e2;
                    bVar = null;
                } catch (Throwable th) {
                    th = th;
                    if (inputStream2 != null) {
                    }
                    throw th;
                }
                try {
                    bVar.f2383e = byteOrder;
                    boolean z = true;
                    int i = 0;
                    switch (this.f2386a) {
                        case 1:
                        case 6:
                            byte[] bArr2 = this.f2388c;
                            if (bArr2.length == 1 && bArr2[0] >= 0 && bArr2[0] <= 1) {
                                String str = new String(new char[]{(char) (this.f2388c[0] + 48)});
                                try {
                                    bVar.close();
                                } catch (IOException e3) {
                                    Log.e("ExifInterface", "IOException occurred while closing InputStream", e3);
                                }
                                return str;
                            }
                            String str2 = new String(this.f2388c, a.M);
                            try {
                                bVar.close();
                            } catch (IOException e4) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e4);
                            }
                            return str2;
                        case 2:
                        case 7:
                            if (this.f2387b >= a.v.length) {
                                int i2 = 0;
                                while (true) {
                                    bArr = a.v;
                                    if (i2 < bArr.length) {
                                        if (this.f2388c[i2] != bArr[i2]) {
                                            z = false;
                                        } else {
                                            i2++;
                                        }
                                    }
                                }
                                if (z) {
                                    i = bArr.length;
                                }
                            }
                            StringBuilder sb = new StringBuilder();
                            while (i < this.f2387b && (b2 = this.f2388c[i]) != 0) {
                                if (b2 >= 32) {
                                    sb.append((char) b2);
                                } else {
                                    sb.append('?');
                                }
                                i++;
                            }
                            String sb2 = sb.toString();
                            try {
                                bVar.close();
                            } catch (IOException e5) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e5);
                            }
                            return sb2;
                        case 3:
                            int[] iArr = new int[this.f2387b];
                            while (i < this.f2387b) {
                                iArr[i] = bVar.readUnsignedShort();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e6) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e6);
                            }
                            return iArr;
                        case 4:
                            long[] jArr = new long[this.f2387b];
                            while (i < this.f2387b) {
                                jArr[i] = bVar.B();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e7) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e7);
                            }
                            return jArr;
                        case 5:
                            e[] eVarArr = new e[this.f2387b];
                            while (i < this.f2387b) {
                                eVarArr[i] = new e(bVar.B(), bVar.B());
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e8) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e8);
                            }
                            return eVarArr;
                        case 8:
                            int[] iArr2 = new int[this.f2387b];
                            while (i < this.f2387b) {
                                iArr2[i] = bVar.readShort();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e9) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e9);
                            }
                            return iArr2;
                        case 9:
                            int[] iArr3 = new int[this.f2387b];
                            while (i < this.f2387b) {
                                iArr3[i] = bVar.readInt();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e10) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e10);
                            }
                            return iArr3;
                        case 10:
                            e[] eVarArr2 = new e[this.f2387b];
                            while (i < this.f2387b) {
                                eVarArr2[i] = new e(bVar.readInt(), bVar.readInt());
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e11) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e11);
                            }
                            return eVarArr2;
                        case 11:
                            double[] dArr = new double[this.f2387b];
                            while (i < this.f2387b) {
                                dArr[i] = bVar.readFloat();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e12) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e12);
                            }
                            return dArr;
                        case 12:
                            double[] dArr2 = new double[this.f2387b];
                            while (i < this.f2387b) {
                                dArr2[i] = bVar.readDouble();
                                i++;
                            }
                            try {
                                bVar.close();
                            } catch (IOException e13) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e13);
                            }
                            return dArr2;
                        default:
                            try {
                                bVar.close();
                            } catch (IOException e14) {
                                Log.e("ExifInterface", "IOException occurred while closing InputStream", e14);
                            }
                            return null;
                    }
                } catch (IOException e15) {
                    e = e15;
                    Log.w("ExifInterface", "IOException occurred during reading a value", e);
                    if (bVar != null) {
                        try {
                            bVar.close();
                        } catch (IOException e16) {
                            Log.e("ExifInterface", "IOException occurred while closing InputStream", e16);
                        }
                    }
                    return null;
                }
            } catch (Throwable th2) {
                th = th2;
                inputStream2 = inputStream;
                if (inputStream2 != null) {
                    try {
                        inputStream2.close();
                    } catch (IOException e17) {
                        Log.e("ExifInterface", "IOException occurred while closing InputStream", e17);
                    }
                }
                throw th;
            }
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("(");
            x.append(a.t[this.f2386a]);
            x.append(", data length:");
            return c.b.a.a.a.s(x, this.f2388c.length, ")");
        }

        public c(int i, int i2, byte[] bArr) {
            this.f2386a = i;
            this.f2387b = i2;
            this.f2388c = bArr;
        }
    }

    /* compiled from: ExifInterface.java */
    /* loaded from: classes.dex */
    public static class d {

        /* renamed from: a  reason: collision with root package name */
        public final int f2389a;

        /* renamed from: b  reason: collision with root package name */
        public final String f2390b;

        /* renamed from: c  reason: collision with root package name */
        public final int f2391c;

        /* renamed from: d  reason: collision with root package name */
        public final int f2392d;

        public d(String str, int i, int i2) {
            this.f2390b = str;
            this.f2389a = i;
            this.f2391c = i2;
            this.f2392d = -1;
        }

        public d(String str, int i, int i2, int i3) {
            this.f2390b = str;
            this.f2389a = i;
            this.f2391c = i2;
            this.f2392d = i3;
        }
    }
}