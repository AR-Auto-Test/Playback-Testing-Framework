package com.google.android.play.core.assetpacks;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Objects;

/* compiled from: com.google.android.play:core@@1.10.3 */
/* loaded from: classes.dex */
public final class zzdr extends InputStream {
    private final Enumeration zza;
    private InputStream zzb;

    public zzdr(Enumeration enumeration) {
        this.zza = enumeration;
        zza();
    }

    @Override // java.io.InputStream, java.io.Closeable, java.lang.AutoCloseable
    public final void close() {
        super.close();
        InputStream inputStream = this.zzb;
        if (inputStream != null) {
            inputStream.close();
            this.zzb = null;
        }
    }

    @Override // java.io.InputStream
    public final int read() {
        while (true) {
            InputStream inputStream = this.zzb;
            if (inputStream == null) {
                return -1;
            }
            int read = inputStream.read();
            if (read != -1) {
                return read;
            }
            zza();
        }
    }

    public final void zza() {
        InputStream inputStream = this.zzb;
        if (inputStream != null) {
            inputStream.close();
        }
        this.zzb = this.zza.hasMoreElements() ? new FileInputStream((File) this.zza.nextElement()) : null;
    }

    @Override // java.io.InputStream
    public final int read(byte[] bArr, int i, int i2) {
        if (this.zzb == null) {
            return -1;
        }
        Objects.requireNonNull(bArr);
        if (i < 0 || i2 < 0 || i2 > bArr.length - i) {
            throw new IndexOutOfBoundsException();
        }
        if (i2 != 0) {
            do {
                int read = this.zzb.read(bArr, i, i2);
                if (read > 0) {
                    return read;
                }
                zza();
            } while (this.zzb != null);
            return -1;
        }
        return 0;
    }
}