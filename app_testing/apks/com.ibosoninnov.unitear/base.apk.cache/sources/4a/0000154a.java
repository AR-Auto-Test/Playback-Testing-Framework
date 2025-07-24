package com.google.android.gms.measurement.internal;

import com.google.android.gms.common.internal.Preconditions;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.List;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-measurement-impl@@21.2.0 */
/* loaded from: classes.dex */
public final class zzia implements Runnable {
    public final /* synthetic */ zzib zza;
    private final URL zzb;
    private final String zzc;
    private final zzfp zzd;

    public zzia(zzib zzibVar, String str, URL url, byte[] bArr, Map map, zzfp zzfpVar, byte[] bArr2) {
        this.zza = zzibVar;
        Preconditions.checkNotEmpty(str);
        Preconditions.checkNotNull(url);
        Preconditions.checkNotNull(zzfpVar);
        this.zzb = url;
        this.zzd = zzfpVar;
        this.zzc = str;
    }

    private final void zzb(final int i, final Exception exc, final byte[] bArr, final Map map) {
        this.zza.zzt.zzaz().zzp(new Runnable() { // from class: com.google.android.gms.measurement.internal.zzhz
            @Override // java.lang.Runnable
            public final void run() {
                zzia.this.zza(i, exc, bArr, map);
            }
        });
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:25:0x006c */
    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:27:0x006e */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:43:0x008c  */
    /* JADX WARN: Removed duplicated region for block: B:50:0x009b  */
    /* JADX WARN: Type inference failed for: r10v0, types: [com.google.android.gms.measurement.internal.zzia] */
    /* JADX WARN: Type inference failed for: r4v0 */
    /* JADX WARN: Type inference failed for: r4v1 */
    /* JADX WARN: Type inference failed for: r4v10 */
    /* JADX WARN: Type inference failed for: r4v11 */
    /* JADX WARN: Type inference failed for: r4v12 */
    /* JADX WARN: Type inference failed for: r4v14 */
    /* JADX WARN: Type inference failed for: r4v2 */
    /* JADX WARN: Type inference failed for: r4v3 */
    /* JADX WARN: Type inference failed for: r4v4, types: [java.util.Map] */
    /* JADX WARN: Type inference failed for: r4v5, types: [java.util.Map] */
    /* JADX WARN: Type inference failed for: r4v8 */
    /* JADX WARN: Type inference failed for: r4v9 */
    @Override // java.lang.Runnable
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void run() {
        HttpURLConnection httpURLConnection;
        ?? r4;
        ?? r42;
        int i;
        Throwable th;
        IOException e2;
        InputStream inputStream;
        ByteArrayOutputStream byteArrayOutputStream;
        this.zza.zzax();
        try {
            zzib zzibVar = this.zza;
            URLConnection openConnection = this.zzb.openConnection();
            if (openConnection instanceof HttpURLConnection) {
                httpURLConnection = (HttpURLConnection) openConnection;
                httpURLConnection.setDefaultUseCaches(false);
                zzibVar.zzt.zzf();
                r4 = 60000;
                r42 = 60000;
                httpURLConnection.setConnectTimeout(60000);
                zzibVar.zzt.zzf();
                httpURLConnection.setReadTimeout(61000);
                httpURLConnection.setInstanceFollowRedirects(false);
                httpURLConnection.setDoInput(true);
                try {
                    i = httpURLConnection.getResponseCode();
                } catch (IOException e3) {
                    e = e3;
                    r42 = 0;
                    IOException iOException = e;
                    i = 0;
                    e2 = iOException;
                    if (httpURLConnection != null) {
                    }
                    zzb(i, e2, null, r42);
                } catch (Throwable th2) {
                    th = th2;
                    r4 = 0;
                    Throwable th3 = th;
                    i = 0;
                    th = th3;
                    if (httpURLConnection != null) {
                    }
                    zzb(i, null, null, r4);
                    throw th;
                }
                try {
                    try {
                        Map<String, List<String>> headerFields = httpURLConnection.getHeaderFields();
                        try {
                            byteArrayOutputStream = new ByteArrayOutputStream();
                            inputStream = httpURLConnection.getInputStream();
                        } catch (Throwable th4) {
                            th = th4;
                            inputStream = null;
                        }
                        try {
                            byte[] bArr = new byte[1024];
                            while (true) {
                                int read = inputStream.read(bArr);
                                if (read > 0) {
                                    byteArrayOutputStream.write(bArr, 0, read);
                                } else {
                                    byte[] byteArray = byteArrayOutputStream.toByteArray();
                                    inputStream.close();
                                    httpURLConnection.disconnect();
                                    zzb(i, null, byteArray, headerFields);
                                    return;
                                }
                            }
                        } catch (Throwable th5) {
                            th = th5;
                            if (inputStream != null) {
                                inputStream.close();
                            }
                            throw th;
                        }
                    } catch (IOException e4) {
                        e2 = e4;
                        if (httpURLConnection != null) {
                        }
                        zzb(i, e2, null, r42);
                    } catch (Throwable th6) {
                        th = th6;
                        if (httpURLConnection != null) {
                        }
                        zzb(i, null, null, r4);
                        throw th;
                    }
                } catch (IOException e5) {
                    e2 = e5;
                    r42 = 0;
                    if (httpURLConnection != null) {
                        httpURLConnection.disconnect();
                    }
                    zzb(i, e2, null, r42);
                } catch (Throwable th7) {
                    th = th7;
                    r4 = 0;
                    if (httpURLConnection != null) {
                        httpURLConnection.disconnect();
                    }
                    zzb(i, null, null, r4);
                    throw th;
                }
            } else {
                throw new IOException("Failed to obtain HTTP connection");
            }
        } catch (IOException e6) {
            e = e6;
            httpURLConnection = null;
            r42 = 0;
        } catch (Throwable th8) {
            th = th8;
            httpURLConnection = null;
            r4 = 0;
        }
    }

    public final /* synthetic */ void zza(int i, Exception exc, byte[] bArr, Map map) {
        zzfp zzfpVar = this.zzd;
        zzfpVar.zza.zzC(this.zzc, i, exc, bArr, map);
    }
}