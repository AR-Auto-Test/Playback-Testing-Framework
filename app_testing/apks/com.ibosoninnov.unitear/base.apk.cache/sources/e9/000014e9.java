package com.google.android.gms.measurement.internal;

import com.google.android.gms.common.internal.Preconditions;
import com.google.common.net.HttpHeaders;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.List;
import java.util.Map;

/* compiled from: com.google.android.gms:play-services-measurement@@21.2.0 */
/* loaded from: classes.dex */
public final class zzem implements Runnable {
    public final /* synthetic */ zzen zza;
    private final URL zzb;
    private final byte[] zzc;
    private final zzej zzd;
    private final String zze;
    private final Map zzf;

    public zzem(zzen zzenVar, String str, URL url, byte[] bArr, Map map, zzej zzejVar) {
        this.zza = zzenVar;
        Preconditions.checkNotEmpty(str);
        Preconditions.checkNotNull(url);
        Preconditions.checkNotNull(zzejVar);
        this.zzb = url;
        this.zzc = bArr;
        this.zzd = zzejVar;
        this.zze = str;
        this.zzf = map;
    }

    /* JADX WARN: Not initialized variable reg: 11, insn: 0x00fc: MOVE  (r12 I:??[OBJECT, ARRAY]) = (r11 I:??[OBJECT, ARRAY]), block:B:42:0x00fa */
    /* JADX WARN: Not initialized variable reg: 11, insn: 0x0102: MOVE  (r12 I:??[OBJECT, ARRAY]) = (r11 I:??[OBJECT, ARRAY]), block:B:44:0x00ff */
    /* JADX WARN: Removed duplicated region for block: B:63:0x0144  */
    /* JADX WARN: Removed duplicated region for block: B:75:0x0184  */
    /* JADX WARN: Removed duplicated region for block: B:79:0x0168 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:81:0x0128 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    @Override // java.lang.Runnable
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final void run() {
        Throwable th;
        int i;
        HttpURLConnection httpURLConnection;
        Map map;
        IOException e2;
        int i2;
        Map map2;
        zzel zzelVar;
        zzfo zzfoVar;
        IOException iOException;
        zzen zzenVar;
        URLConnection openConnection;
        int responseCode;
        Map map3;
        Map map4;
        Map<String, List<String>> headerFields;
        InputStream inputStream;
        ByteArrayOutputStream byteArrayOutputStream;
        this.zza.zzax();
        OutputStream outputStream = null;
        try {
            zzenVar = this.zza;
            openConnection = this.zzb.openConnection();
        } catch (IOException e3) {
            e2 = e3;
            i2 = 0;
            httpURLConnection = null;
            map2 = null;
        } catch (Throwable th2) {
            th = th2;
            i = 0;
            httpURLConnection = null;
            map = null;
        }
        if (openConnection instanceof HttpURLConnection) {
            httpURLConnection = (HttpURLConnection) openConnection;
            httpURLConnection.setDefaultUseCaches(false);
            zzenVar.zzt.zzf();
            httpURLConnection.setConnectTimeout(60000);
            zzenVar.zzt.zzf();
            httpURLConnection.setReadTimeout(61000);
            httpURLConnection.setInstanceFollowRedirects(false);
            httpURLConnection.setDoInput(true);
            try {
                Map map5 = this.zzf;
                if (map5 != null) {
                    for (Map.Entry entry : map5.entrySet()) {
                        httpURLConnection.addRequestProperty((String) entry.getKey(), (String) entry.getValue());
                    }
                }
                if (this.zzc != null) {
                    byte[] zzy = this.zza.zzf.zzu().zzy(this.zzc);
                    zzef zzj = this.zza.zzt.zzay().zzj();
                    int length = zzy.length;
                    zzj.zzb("Uploading data. size", Integer.valueOf(length));
                    httpURLConnection.setDoOutput(true);
                    httpURLConnection.addRequestProperty(HttpHeaders.CONTENT_ENCODING, "gzip");
                    httpURLConnection.setFixedLengthStreamingMode(length);
                    httpURLConnection.connect();
                    OutputStream outputStream2 = httpURLConnection.getOutputStream();
                    try {
                        outputStream2.write(zzy);
                        outputStream2.close();
                    } catch (IOException e4) {
                        e2 = e4;
                        i2 = 0;
                        map2 = null;
                        outputStream = outputStream2;
                        iOException = e2;
                        if (outputStream != null) {
                            try {
                                outputStream.close();
                            } catch (IOException e5) {
                                this.zza.zzt.zzay().zzd().zzc("Error closing HTTP compressed POST connection output stream. appId", zzeh.zzn(this.zze), e5);
                            }
                        }
                        if (httpURLConnection != null) {
                            httpURLConnection.disconnect();
                        }
                        zzfoVar = this.zza.zzt.zzaz();
                        zzelVar = new zzel(this.zze, this.zzd, i2, iOException, null, map2, null);
                        zzfoVar.zzp(zzelVar);
                        return;
                    } catch (Throwable th3) {
                        th = th3;
                        i = 0;
                        map = null;
                        outputStream = outputStream2;
                        if (outputStream != null) {
                            try {
                                outputStream.close();
                            } catch (IOException e6) {
                                this.zza.zzt.zzay().zzd().zzc("Error closing HTTP compressed POST connection output stream. appId", zzeh.zzn(this.zze), e6);
                            }
                        }
                        if (httpURLConnection != null) {
                            httpURLConnection.disconnect();
                        }
                        this.zza.zzt.zzaz().zzp(new zzel(this.zze, this.zzd, i, null, null, map, null));
                        throw th;
                    }
                }
                responseCode = httpURLConnection.getResponseCode();
            } catch (IOException e7) {
                i2 = 0;
                map2 = null;
                iOException = e7;
            } catch (Throwable th4) {
                i = 0;
                map = null;
                th = th4;
            }
            try {
                try {
                    headerFields = httpURLConnection.getHeaderFields();
                    try {
                        byteArrayOutputStream = new ByteArrayOutputStream();
                        inputStream = httpURLConnection.getInputStream();
                    } catch (Throwable th5) {
                        th = th5;
                        inputStream = null;
                    }
                } catch (IOException e8) {
                    iOException = e8;
                    i2 = responseCode;
                    map2 = map4;
                    if (outputStream != null) {
                    }
                    if (httpURLConnection != null) {
                    }
                    zzfoVar = this.zza.zzt.zzaz();
                    zzelVar = new zzel(this.zze, this.zzd, i2, iOException, null, map2, null);
                    zzfoVar.zzp(zzelVar);
                    return;
                } catch (Throwable th6) {
                    th = th6;
                    i = responseCode;
                    map = map3;
                    if (outputStream != null) {
                    }
                    if (httpURLConnection != null) {
                    }
                    this.zza.zzt.zzaz().zzp(new zzel(this.zze, this.zzd, i, null, null, map, null));
                    throw th;
                }
            } catch (IOException e9) {
                map2 = null;
                iOException = e9;
                i2 = responseCode;
                if (outputStream != null) {
                }
                if (httpURLConnection != null) {
                }
                zzfoVar = this.zza.zzt.zzaz();
                zzelVar = new zzel(this.zze, this.zzd, i2, iOException, null, map2, null);
                zzfoVar.zzp(zzelVar);
                return;
            } catch (Throwable th7) {
                th = th7;
                map = null;
                i = responseCode;
                if (outputStream != null) {
                }
                if (httpURLConnection != null) {
                }
                this.zza.zzt.zzaz().zzp(new zzel(this.zze, this.zzd, i, null, null, map, null));
                throw th;
            }
            try {
                byte[] bArr = new byte[1024];
                while (true) {
                    int read = inputStream.read(bArr);
                    if (read <= 0) {
                        break;
                    }
                    byteArrayOutputStream.write(bArr, 0, read);
                }
                byte[] byteArray = byteArrayOutputStream.toByteArray();
                inputStream.close();
                httpURLConnection.disconnect();
                zzfoVar = this.zza.zzt.zzaz();
                zzelVar = new zzel(this.zze, this.zzd, responseCode, null, byteArray, headerFields, null);
                zzfoVar.zzp(zzelVar);
                return;
            } catch (Throwable th8) {
                th = th8;
                if (inputStream != null) {
                    inputStream.close();
                }
                throw th;
            }
        }
        throw new IOException("Failed to obtain HTTP connection");
    }
}