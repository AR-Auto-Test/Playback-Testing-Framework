package com.google.android.play.core.internal;

import android.util.Pair;
import c.b.a.a.a;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.security.DigestException;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.PSSParameterSpec;
import java.security.spec.X509EncodedKeySpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.opencv.imgcodecs.Imgcodecs;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzi {
    /* JADX DEBUG: Another duplicated slice has different insns count: {[]}, finally: {[INVOKE] complete} */
    public static X509Certificate[][] zza(String str) {
        RandomAccessFile randomAccessFile = new RandomAccessFile(str, "r");
        try {
            Pair zzc = zzj.zzc(randomAccessFile);
            if (zzc != null) {
                ByteBuffer byteBuffer = (ByteBuffer) zzc.first;
                long longValue = ((Long) zzc.second).longValue();
                long j = (-20) + longValue;
                if (j >= 0) {
                    randomAccessFile.seek(j);
                    if (randomAccessFile.readInt() == 1347094023) {
                        throw new zzf("ZIP64 APK not supported");
                    }
                }
                long zza = zzj.zza(byteBuffer);
                if (zza < longValue) {
                    if (zzj.zzb(byteBuffer) + zza == longValue) {
                        if (zza >= 32) {
                            ByteBuffer allocate = ByteBuffer.allocate(24);
                            ByteOrder byteOrder = ByteOrder.LITTLE_ENDIAN;
                            allocate.order(byteOrder);
                            randomAccessFile.seek(zza - allocate.capacity());
                            randomAccessFile.readFully(allocate.array(), allocate.arrayOffset(), allocate.capacity());
                            if (allocate.getLong(8) == 2334950737559900225L && allocate.getLong(16) == 3617552046287187010L) {
                                int i = 0;
                                long j2 = allocate.getLong(0);
                                if (j2 < allocate.capacity() || j2 > 2147483639) {
                                    StringBuilder sb = new StringBuilder(57);
                                    sb.append("APK Signing Block size out of range: ");
                                    sb.append(j2);
                                    throw new zzf(sb.toString());
                                }
                                int i2 = (int) (8 + j2);
                                long j3 = zza - i2;
                                if (j3 >= 0) {
                                    ByteBuffer allocate2 = ByteBuffer.allocate(i2);
                                    allocate2.order(byteOrder);
                                    randomAccessFile.seek(j3);
                                    randomAccessFile.readFully(allocate2.array(), allocate2.arrayOffset(), allocate2.capacity());
                                    long j4 = allocate2.getLong(0);
                                    if (j4 == j2) {
                                        Pair create = Pair.create(allocate2, Long.valueOf(j3));
                                        ByteBuffer byteBuffer2 = (ByteBuffer) create.first;
                                        long longValue2 = ((Long) create.second).longValue();
                                        if (byteBuffer2.order() == byteOrder) {
                                            int capacity = byteBuffer2.capacity() - 24;
                                            if (capacity >= 8) {
                                                int capacity2 = byteBuffer2.capacity();
                                                if (capacity <= byteBuffer2.capacity()) {
                                                    int limit = byteBuffer2.limit();
                                                    int position = byteBuffer2.position();
                                                    byteBuffer2.position(0);
                                                    byteBuffer2.limit(capacity);
                                                    byteBuffer2.position(8);
                                                    ByteBuffer slice = byteBuffer2.slice();
                                                    slice.order(byteBuffer2.order());
                                                    byteBuffer2.position(0);
                                                    byteBuffer2.limit(limit);
                                                    byteBuffer2.position(position);
                                                    while (slice.hasRemaining()) {
                                                        i++;
                                                        if (slice.remaining() >= 8) {
                                                            long j5 = slice.getLong();
                                                            if (j5 >= 4 && j5 <= 2147483647L) {
                                                                int i3 = (int) j5;
                                                                int position2 = slice.position() + i3;
                                                                if (i3 <= slice.remaining()) {
                                                                    if (slice.getInt() == 1896449818) {
                                                                        X509Certificate[][] zzl = zzl(randomAccessFile.getChannel(), new zze(zze(slice, i3 - 4), longValue2, zza, longValue, byteBuffer, null));
                                                                        randomAccessFile.close();
                                                                        return zzl;
                                                                    }
                                                                    slice.position(position2);
                                                                } else {
                                                                    int remaining = slice.remaining();
                                                                    StringBuilder sb2 = new StringBuilder(91);
                                                                    sb2.append("APK Signing Block entry #");
                                                                    sb2.append(i);
                                                                    sb2.append(" size out of range: ");
                                                                    sb2.append(i3);
                                                                    sb2.append(", available: ");
                                                                    sb2.append(remaining);
                                                                    throw new zzf(sb2.toString());
                                                                }
                                                            } else {
                                                                StringBuilder sb3 = new StringBuilder(76);
                                                                sb3.append("APK Signing Block entry #");
                                                                sb3.append(i);
                                                                sb3.append(" size out of range: ");
                                                                sb3.append(j5);
                                                                throw new zzf(sb3.toString());
                                                            }
                                                        } else {
                                                            StringBuilder sb4 = new StringBuilder(70);
                                                            sb4.append("Insufficient data to read size of APK Signing Block entry #");
                                                            sb4.append(i);
                                                            throw new zzf(sb4.toString());
                                                        }
                                                    }
                                                    throw new zzf("No APK Signature Scheme v2 block in APK Signing Block");
                                                }
                                                StringBuilder sb5 = new StringBuilder(41);
                                                sb5.append("end > capacity: ");
                                                sb5.append(capacity);
                                                sb5.append(" > ");
                                                sb5.append(capacity2);
                                                throw new IllegalArgumentException(sb5.toString());
                                            }
                                            StringBuilder sb6 = new StringBuilder(38);
                                            sb6.append("end < start: ");
                                            sb6.append(capacity);
                                            sb6.append(" < ");
                                            sb6.append(8);
                                            throw new IllegalArgumentException(sb6.toString());
                                        }
                                        throw new IllegalArgumentException("ByteBuffer byte order must be little endian");
                                    }
                                    StringBuilder sb7 = new StringBuilder(103);
                                    sb7.append("APK Signing Block sizes in header and footer do not match: ");
                                    sb7.append(j4);
                                    sb7.append(" vs ");
                                    sb7.append(j2);
                                    throw new zzf(sb7.toString());
                                }
                                StringBuilder sb8 = new StringBuilder(59);
                                sb8.append("APK Signing Block offset out of range: ");
                                sb8.append(j3);
                                throw new zzf(sb8.toString());
                            }
                            throw new zzf("No APK Signing Block before ZIP Central Directory");
                        }
                        StringBuilder sb9 = new StringBuilder(87);
                        sb9.append("APK too small for APK Signing Block. ZIP Central Directory offset: ");
                        sb9.append(zza);
                        throw new zzf(sb9.toString());
                    }
                    throw new zzf("ZIP Central Directory is not immediately followed by End of Central Directory");
                }
                StringBuilder sb10 = new StringBuilder(122);
                sb10.append("ZIP Central Directory offset out of range: ");
                sb10.append(zza);
                sb10.append(". ZIP End of Central Directory offset: ");
                sb10.append(longValue);
                throw new zzf(sb10.toString());
            }
            long length = randomAccessFile.length();
            StringBuilder sb11 = new StringBuilder(102);
            sb11.append("Not an APK file: ZIP End of Central Directory record not found in file with ");
            sb11.append(length);
            sb11.append(" bytes");
            throw new zzf(sb11.toString());
        } finally {
            try {
                randomAccessFile.close();
            } catch (IOException unused) {
            }
        }
    }

    private static int zzb(int i) {
        if (i != 1) {
            if (i == 2) {
                return 64;
            }
            throw new IllegalArgumentException(a.g(44, "Unknown content digest algorthm: ", i));
        }
        return 32;
    }

    private static int zzc(int i) {
        if (i != 513) {
            if (i != 514) {
                if (i != 769) {
                    switch (i) {
                        case Imgcodecs.IMWRITE_TIFF_XDPI /* 257 */:
                        case Imgcodecs.IMWRITE_TIFF_COMPRESSION /* 259 */:
                            return 1;
                        case Imgcodecs.IMWRITE_TIFF_YDPI /* 258 */:
                        case 260:
                            return 2;
                        default:
                            String valueOf = String.valueOf(Long.toHexString(i));
                            throw new IllegalArgumentException(valueOf.length() != 0 ? "Unknown signature algorithm: 0x".concat(valueOf) : new String("Unknown signature algorithm: 0x"));
                    }
                }
                return 1;
            }
            return 2;
        }
        return 1;
    }

    private static String zzd(int i) {
        if (i != 1) {
            if (i == 2) {
                return "SHA-512";
            }
            throw new IllegalArgumentException(a.g(44, "Unknown content digest algorthm: ", i));
        }
        return "SHA-256";
    }

    private static ByteBuffer zze(ByteBuffer byteBuffer, int i) {
        int limit = byteBuffer.limit();
        int position = byteBuffer.position();
        int i2 = i + position;
        if (i2 >= position && i2 <= limit) {
            byteBuffer.limit(i2);
            try {
                ByteBuffer slice = byteBuffer.slice();
                slice.order(byteBuffer.order());
                byteBuffer.position(i2);
                return slice;
            } finally {
                byteBuffer.limit(limit);
            }
        }
        throw new BufferUnderflowException();
    }

    private static ByteBuffer zzf(ByteBuffer byteBuffer) {
        if (byteBuffer.remaining() >= 4) {
            int i = byteBuffer.getInt();
            if (i >= 0) {
                if (i <= byteBuffer.remaining()) {
                    return zze(byteBuffer, i);
                }
                throw new IOException(a.h(101, "Length-prefixed field longer than remaining buffer. Field length: ", i, ", remaining: ", byteBuffer.remaining()));
            }
            throw new IllegalArgumentException("Negative length");
        }
        throw new IOException(a.g(93, "Remaining buffer too short to contain length of length-prefixed field. Remaining: ", byteBuffer.remaining()));
    }

    private static void zzg(int i, byte[] bArr, int i2) {
        bArr[1] = (byte) (i & 255);
        bArr[2] = (byte) ((i >>> 8) & 255);
        bArr[3] = (byte) ((i >>> 16) & 255);
        bArr[4] = (byte) (i >> 24);
    }

    private static void zzh(Map map, FileChannel fileChannel, long j, long j2, long j3, ByteBuffer byteBuffer) {
        if (!map.isEmpty()) {
            zzd zzdVar = new zzd(fileChannel, 0L, j);
            zzd zzdVar2 = new zzd(fileChannel, j2, j3 - j2);
            ByteBuffer duplicate = byteBuffer.duplicate();
            duplicate.order(ByteOrder.LITTLE_ENDIAN);
            zzj.zzd(duplicate, j);
            zzb zzbVar = new zzb(duplicate);
            int size = map.size();
            int[] iArr = new int[size];
            int i = 0;
            for (Integer num : map.keySet()) {
                iArr[i] = num.intValue();
                i++;
            }
            try {
                byte[][] zzk = zzk(iArr, new zzc[]{zzdVar, zzdVar2, zzbVar});
                for (int i2 = 0; i2 < size; i2++) {
                    int i3 = iArr[i2];
                    if (!MessageDigest.isEqual((byte[]) map.get(Integer.valueOf(i3)), zzk[i2])) {
                        throw new SecurityException(zzd(i3).concat(" digest of contents did not verify"));
                    }
                }
                return;
            } catch (DigestException e2) {
                throw new SecurityException("Failed to compute digest(s) of contents", e2);
            }
        }
        throw new SecurityException("No digests provided");
    }

    private static byte[] zzi(ByteBuffer byteBuffer) {
        int i = byteBuffer.getInt();
        if (i >= 0) {
            if (i <= byteBuffer.remaining()) {
                byte[] bArr = new byte[i];
                byteBuffer.get(bArr);
                return bArr;
            }
            throw new IOException(a.h(90, "Underflow while reading length-prefixed value. Length: ", i, ", available: ", byteBuffer.remaining()));
        }
        throw new IOException("Negative length");
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:0x0048, code lost:
        r11 = zzc(r10);
        r12 = zzc(r7);
     */
    /* JADX WARN: Code restructure failed: missing block: B:16:0x0050, code lost:
        if (r11 == 1) goto L26;
     */
    /* JADX WARN: Code restructure failed: missing block: B:17:0x0052, code lost:
        if (r12 == 1) goto L19;
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private static X509Certificate[] zzj(ByteBuffer byteBuffer, Map map, CertificateFactory certificateFactory) {
        String str;
        Pair create;
        ByteBuffer zzf = zzf(byteBuffer);
        ByteBuffer zzf2 = zzf(byteBuffer);
        byte[] zzi = zzi(byteBuffer);
        ArrayList arrayList = new ArrayList();
        byte[] bArr = null;
        int i = 0;
        int i2 = -1;
        byte[] bArr2 = null;
        while (zzf2.hasRemaining()) {
            i++;
            try {
                ByteBuffer zzf3 = zzf(zzf2);
                if (zzf3.remaining() >= 8) {
                    int i3 = zzf3.getInt();
                    arrayList.add(Integer.valueOf(i3));
                    if (i3 != 513 && i3 != 514 && i3 != 769) {
                        switch (i3) {
                            case Imgcodecs.IMWRITE_TIFF_XDPI /* 257 */:
                            case Imgcodecs.IMWRITE_TIFF_YDPI /* 258 */:
                            case Imgcodecs.IMWRITE_TIFF_COMPRESSION /* 259 */:
                            case 260:
                                break;
                            default:
                                continue;
                        }
                    }
                    bArr2 = zzi(zzf3);
                    i2 = i3;
                } else {
                    throw new SecurityException("Signature record too short");
                }
            } catch (IOException | BufferUnderflowException e2) {
                throw new SecurityException(a.g(45, "Failed to parse signature record #", i), e2);
            }
        }
        if (i2 == -1) {
            if (i == 0) {
                throw new SecurityException("No signatures found");
            }
            throw new SecurityException("No supported signatures found");
        }
        if (i2 == 513 || i2 == 514) {
            str = "EC";
        } else if (i2 != 769) {
            switch (i2) {
                case Imgcodecs.IMWRITE_TIFF_XDPI /* 257 */:
                case Imgcodecs.IMWRITE_TIFF_YDPI /* 258 */:
                case Imgcodecs.IMWRITE_TIFF_COMPRESSION /* 259 */:
                case 260:
                    str = "RSA";
                    break;
                default:
                    String valueOf = String.valueOf(Long.toHexString(i2));
                    throw new IllegalArgumentException(valueOf.length() != 0 ? "Unknown signature algorithm: 0x".concat(valueOf) : new String("Unknown signature algorithm: 0x"));
            }
        } else {
            str = "DSA";
        }
        if (i2 == 513) {
            create = Pair.create("SHA256withECDSA", null);
        } else if (i2 == 514) {
            create = Pair.create("SHA512withECDSA", null);
        } else if (i2 != 769) {
            switch (i2) {
                case Imgcodecs.IMWRITE_TIFF_XDPI /* 257 */:
                    create = Pair.create("SHA256withRSA/PSS", new PSSParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA256, 32, 1));
                    break;
                case Imgcodecs.IMWRITE_TIFF_YDPI /* 258 */:
                    create = Pair.create("SHA512withRSA/PSS", new PSSParameterSpec("SHA-512", "MGF1", MGF1ParameterSpec.SHA512, 64, 1));
                    break;
                case Imgcodecs.IMWRITE_TIFF_COMPRESSION /* 259 */:
                    create = Pair.create("SHA256withRSA", null);
                    break;
                case 260:
                    create = Pair.create("SHA512withRSA", null);
                    break;
                default:
                    String valueOf2 = String.valueOf(Long.toHexString(i2));
                    throw new IllegalArgumentException(valueOf2.length() != 0 ? "Unknown signature algorithm: 0x".concat(valueOf2) : new String("Unknown signature algorithm: 0x"));
            }
        } else {
            create = Pair.create("SHA256withDSA", null);
        }
        String str2 = (String) create.first;
        AlgorithmParameterSpec algorithmParameterSpec = (AlgorithmParameterSpec) create.second;
        try {
            PublicKey generatePublic = KeyFactory.getInstance(str).generatePublic(new X509EncodedKeySpec(zzi));
            Signature signature = Signature.getInstance(str2);
            signature.initVerify(generatePublic);
            if (algorithmParameterSpec != null) {
                signature.setParameter(algorithmParameterSpec);
            }
            signature.update(zzf);
            if (signature.verify(bArr2)) {
                zzf.clear();
                ByteBuffer zzf4 = zzf(zzf);
                ArrayList arrayList2 = new ArrayList();
                int i4 = 0;
                while (zzf4.hasRemaining()) {
                    i4++;
                    try {
                        ByteBuffer zzf5 = zzf(zzf4);
                        if (zzf5.remaining() >= 8) {
                            int i5 = zzf5.getInt();
                            arrayList2.add(Integer.valueOf(i5));
                            if (i5 == i2) {
                                bArr = zzi(zzf5);
                            }
                        } else {
                            throw new IOException("Record too short");
                        }
                    } catch (IOException | BufferUnderflowException e3) {
                        throw new IOException(a.g(42, "Failed to parse digest record #", i4), e3);
                    }
                }
                if (arrayList.equals(arrayList2)) {
                    int zzc = zzc(i2);
                    byte[] bArr3 = (byte[]) map.put(Integer.valueOf(zzc), bArr);
                    if (bArr3 != null && !MessageDigest.isEqual(bArr3, bArr)) {
                        throw new SecurityException(zzd(zzc).concat(" contents digest does not match the digest specified by a preceding signer"));
                    }
                    ByteBuffer zzf6 = zzf(zzf);
                    ArrayList arrayList3 = new ArrayList();
                    int i6 = 0;
                    while (zzf6.hasRemaining()) {
                        i6++;
                        byte[] zzi2 = zzi(zzf6);
                        try {
                            arrayList3.add(new zzg((X509Certificate) certificateFactory.generateCertificate(new ByteArrayInputStream(zzi2)), zzi2));
                        } catch (CertificateException e4) {
                            throw new SecurityException(a.g(41, "Failed to decode certificate #", i6), e4);
                        }
                    }
                    if (!arrayList3.isEmpty()) {
                        if (Arrays.equals(zzi, ((X509Certificate) arrayList3.get(0)).getPublicKey().getEncoded())) {
                            return (X509Certificate[]) arrayList3.toArray(new X509Certificate[arrayList3.size()]);
                        }
                        throw new SecurityException("Public key mismatch between certificate and signature record");
                    }
                    throw new SecurityException("No certificates listed");
                }
                throw new SecurityException("Signature algorithms don't match between digests and signatures records");
            }
            throw new SecurityException(String.valueOf(str2).concat(" signature did not verify"));
        } catch (InvalidAlgorithmParameterException | InvalidKeyException | NoSuchAlgorithmException | SignatureException | InvalidKeySpecException e5) {
            StringBuilder sb = new StringBuilder(String.valueOf(str2).length() + 27);
            sb.append("Failed to verify ");
            sb.append(str2);
            sb.append(" signature");
            throw new SecurityException(sb.toString(), e5);
        }
    }

    private static byte[][] zzk(int[] iArr, zzc[] zzcVarArr) {
        int i;
        int length;
        long j = 0;
        int i2 = 0;
        long j2 = 0;
        int i3 = 0;
        while (true) {
            i = 3;
            if (i3 >= 3) {
                break;
            }
            j2 += (zzcVarArr[i3].zza() + 1048575) / 1048576;
            i3++;
        }
        if (j2 < 2097151) {
            int i4 = (int) j2;
            byte[][] bArr = new byte[iArr.length];
            int i5 = 0;
            while (true) {
                length = iArr.length;
                if (i5 >= length) {
                    break;
                }
                byte[] bArr2 = new byte[(zzb(iArr[i5]) * i4) + 5];
                bArr2[0] = 90;
                zzg(i4, bArr2, 1);
                bArr[i5] = bArr2;
                i5++;
            }
            byte[] bArr3 = new byte[5];
            bArr3[0] = -91;
            MessageDigest[] messageDigestArr = new MessageDigest[length];
            for (int i6 = 0; i6 < iArr.length; i6++) {
                String zzd = zzd(iArr[i6]);
                try {
                    messageDigestArr[i6] = MessageDigest.getInstance(zzd);
                } catch (NoSuchAlgorithmException e2) {
                    throw new RuntimeException(zzd.concat(" digest not supported"), e2);
                }
            }
            long j3 = 1048576;
            int i7 = 0;
            int i8 = 0;
            while (i2 < i) {
                zzc zzcVar = zzcVarArr[i2];
                int i9 = i7;
                int i10 = i8;
                long zza = zzcVar.zza();
                long j4 = j3;
                long j5 = j;
                while (zza > j) {
                    int min = (int) Math.min(zza, j4);
                    zzg(min, bArr3, 1);
                    for (int i11 = 0; i11 < length; i11++) {
                        messageDigestArr[i11].update(bArr3);
                    }
                    try {
                        zzcVar.zzb(messageDigestArr, j5, min);
                        int i12 = 0;
                        while (i12 < iArr.length) {
                            int i13 = iArr[i12];
                            byte[] bArr4 = bArr[i12];
                            int zzb = zzb(i13);
                            byte[] bArr5 = bArr3;
                            MessageDigest messageDigest = messageDigestArr[i12];
                            MessageDigest[] messageDigestArr2 = messageDigestArr;
                            int digest = messageDigest.digest(bArr4, (i10 * zzb) + 5, zzb);
                            if (digest != zzb) {
                                String algorithm = messageDigest.getAlgorithm();
                                StringBuilder sb = new StringBuilder(String.valueOf(algorithm).length() + 46);
                                sb.append("Unexpected output size of ");
                                sb.append(algorithm);
                                sb.append(" digest: ");
                                sb.append(digest);
                                throw new RuntimeException(sb.toString());
                            }
                            i12++;
                            bArr3 = bArr5;
                            messageDigestArr = messageDigestArr2;
                        }
                        long j6 = min;
                        j5 += j6;
                        zza -= j6;
                        i10++;
                        j = 0;
                        j4 = 1048576;
                    } catch (IOException e3) {
                        throw new DigestException(a.h(59, "Failed to digest chunk #", i10, " of section #", i9), e3);
                    }
                }
                i8 = i10;
                i7 = i9 + 1;
                i2++;
                j = 0;
                j3 = 1048576;
                i = 3;
                messageDigestArr = messageDigestArr;
            }
            byte[][] bArr6 = new byte[iArr.length];
            for (int i14 = 0; i14 < iArr.length; i14++) {
                int i15 = iArr[i14];
                byte[] bArr7 = bArr[i14];
                String zzd2 = zzd(i15);
                try {
                    bArr6[i14] = MessageDigest.getInstance(zzd2).digest(bArr7);
                } catch (NoSuchAlgorithmException e4) {
                    throw new RuntimeException(zzd2.concat(" digest not supported"), e4);
                }
            }
            return bArr6;
        }
        StringBuilder sb2 = new StringBuilder(37);
        sb2.append("Too many chunks: ");
        sb2.append(j2);
        throw new DigestException(sb2.toString());
    }

    private static X509Certificate[][] zzl(FileChannel fileChannel, zze zzeVar) {
        ByteBuffer byteBuffer;
        long j;
        long j2;
        long j3;
        ByteBuffer byteBuffer2;
        HashMap hashMap = new HashMap();
        ArrayList arrayList = new ArrayList();
        try {
            CertificateFactory certificateFactory = CertificateFactory.getInstance("X.509");
            try {
                byteBuffer = zzeVar.zza;
                ByteBuffer zzf = zzf(byteBuffer);
                int i = 0;
                while (zzf.hasRemaining()) {
                    i++;
                    try {
                        arrayList.add(zzj(zzf(zzf), hashMap, certificateFactory));
                    } catch (IOException | SecurityException | BufferUnderflowException e2) {
                        StringBuilder sb = new StringBuilder(48);
                        sb.append("Failed to parse/verify signer #");
                        sb.append(i);
                        sb.append(" block");
                        throw new SecurityException(sb.toString(), e2);
                    }
                }
                if (i > 0) {
                    if (!hashMap.isEmpty()) {
                        j = zzeVar.zzb;
                        j2 = zzeVar.zzc;
                        j3 = zzeVar.zzd;
                        byteBuffer2 = zzeVar.zze;
                        zzh(hashMap, fileChannel, j, j2, j3, byteBuffer2);
                        return (X509Certificate[][]) arrayList.toArray(new X509Certificate[arrayList.size()]);
                    }
                    throw new SecurityException("No content digests found");
                }
                throw new SecurityException("No signers found");
            } catch (IOException e3) {
                throw new SecurityException("Failed to read list of signers", e3);
            }
        } catch (CertificateException e4) {
            throw new RuntimeException("Failed to obtain X.509 CertificateFactory", e4);
        }
    }
}