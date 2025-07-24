package com.google.android.play.core.internal;

import c.b.a.a.a;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import org.opencv.calib3d.Calib3d;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzcl {
    public static long zza(zzcm zzcmVar, InputStream inputStream, OutputStream outputStream, long j) {
        byte[] bArr = new byte[Calib3d.CALIB_RATIONAL_MODEL];
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(inputStream, 4096));
        int readInt = dataInputStream.readInt();
        if (readInt != -771763713) {
            String valueOf = String.valueOf(String.format("%x", Integer.valueOf(readInt)));
            throw new zzck(valueOf.length() != 0 ? "Unexpected magic=".concat(valueOf) : new String("Unexpected magic="));
        }
        int read = dataInputStream.read();
        if (read != 4) {
            throw new zzck(a.g(30, "Unexpected version=", read));
        }
        long j2 = 0;
        while (true) {
            long j3 = j - j2;
            try {
                int read2 = dataInputStream.read();
                if (read2 == -1) {
                    throw new IOException("Patch file overrun");
                }
                if (read2 == 0) {
                    return j2;
                }
                switch (read2) {
                    case 247:
                        read2 = dataInputStream.readUnsignedShort();
                        zzc(bArr, dataInputStream, outputStream, read2, j3);
                        break;
                    case 248:
                        read2 = dataInputStream.readInt();
                        zzc(bArr, dataInputStream, outputStream, read2, j3);
                        break;
                    case 249:
                        long readUnsignedShort = dataInputStream.readUnsignedShort();
                        read2 = dataInputStream.read();
                        if (read2 != -1) {
                            zzb(bArr, zzcmVar, outputStream, readUnsignedShort, read2, j3);
                            break;
                        } else {
                            throw new IOException("Unexpected end of patch");
                        }
                    case BaseTransientBottomBar.ANIMATION_DURATION /* 250 */:
                        read2 = dataInputStream.readUnsignedShort();
                        zzb(bArr, zzcmVar, outputStream, dataInputStream.readUnsignedShort(), read2, j3);
                        break;
                    case 251:
                        read2 = dataInputStream.readInt();
                        zzb(bArr, zzcmVar, outputStream, dataInputStream.readUnsignedShort(), read2, j3);
                        break;
                    case 252:
                        long readInt2 = dataInputStream.readInt();
                        read2 = dataInputStream.read();
                        if (read2 != -1) {
                            zzb(bArr, zzcmVar, outputStream, readInt2, read2, j3);
                            break;
                        } else {
                            throw new IOException("Unexpected end of patch");
                        }
                    case 253:
                        read2 = dataInputStream.readUnsignedShort();
                        zzb(bArr, zzcmVar, outputStream, dataInputStream.readInt(), read2, j3);
                        break;
                    case 254:
                        read2 = dataInputStream.readInt();
                        zzb(bArr, zzcmVar, outputStream, dataInputStream.readInt(), read2, j3);
                        break;
                    case 255:
                        long readLong = dataInputStream.readLong();
                        read2 = dataInputStream.readInt();
                        zzb(bArr, zzcmVar, outputStream, readLong, read2, j3);
                        break;
                    default:
                        zzc(bArr, dataInputStream, outputStream, read2, j3);
                        break;
                }
                j2 += read2;
            } finally {
                outputStream.flush();
            }
        }
    }

    private static void zzb(byte[] bArr, zzcm zzcmVar, OutputStream outputStream, long j, int i, long j2) {
        if (i < 0) {
            throw new IOException("copyLength negative");
        }
        if (j < 0) {
            throw new IOException("inputOffset negative");
        }
        long j3 = i;
        if (j3 <= j2) {
            try {
                InputStream zzc = new zzcn(zzcmVar, j, j3).zzc();
                while (i > 0) {
                    int min = Math.min(i, (int) Calib3d.CALIB_RATIONAL_MODEL);
                    int i2 = 0;
                    while (i2 < min) {
                        int read = zzc.read(bArr, i2, min - i2);
                        if (read == -1) {
                            throw new IOException("truncated input stream");
                        }
                        i2 += read;
                    }
                    outputStream.write(bArr, 0, min);
                    i -= min;
                }
                zzc.close();
                return;
            } catch (EOFException e2) {
                throw new IOException("patch underrun", e2);
            }
        }
        throw new IOException("Output length overrun");
    }

    private static void zzc(byte[] bArr, DataInputStream dataInputStream, OutputStream outputStream, int i, long j) {
        if (i < 0) {
            throw new IOException("copyLength negative");
        }
        if (i > j) {
            throw new IOException("Output length overrun");
        }
        while (i > 0) {
            try {
                int min = Math.min(i, (int) Calib3d.CALIB_RATIONAL_MODEL);
                dataInputStream.readFully(bArr, 0, min);
                outputStream.write(bArr, 0, min);
                i -= min;
            } catch (EOFException unused) {
                throw new IOException("patch underrun");
            }
        }
    }
}