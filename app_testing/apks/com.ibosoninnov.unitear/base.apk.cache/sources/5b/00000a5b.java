package c.e.b.p000if;

import android.content.Context;
import android.os.AsyncTask;
import android.os.PowerManager;
import c.e.b.gf.b;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/* compiled from: DownloadTask.java */
/* renamed from: c.e.b.if.j  reason: invalid package */
/* loaded from: classes2.dex */
public class j extends AsyncTask<String, Integer, String> {

    /* renamed from: a  reason: collision with root package name */
    public Context f4879a;

    /* renamed from: b  reason: collision with root package name */
    public b f4880b;

    /* renamed from: c  reason: collision with root package name */
    public String f4881c;

    /* renamed from: d  reason: collision with root package name */
    public PowerManager.WakeLock f4882d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f4883e;

    /* renamed from: f  reason: collision with root package name */
    public String f4884f = "";

    public j(Context context, String str, boolean z) {
        this.f4883e = false;
        this.f4879a = context;
        this.f4881c = str;
        this.f4883e = z;
        this.f4880b = (b) context;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    /* JADX WARN: Code restructure failed: missing block: B:30:0x00ed, code lost:
        r7.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x00f0, code lost:
        r8.close();
        r7.close();
     */
    /* JADX WARN: Removed duplicated region for block: B:100:0x017c A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:77:0x016f A[Catch: IOException -> 0x0172, TRY_LEAVE, TryCatch #2 {IOException -> 0x0172, blocks: (B:75:0x016a, B:77:0x016f), top: B:96:0x016a }] */
    /* JADX WARN: Removed duplicated region for block: B:79:0x0174  */
    /* JADX WARN: Removed duplicated region for block: B:86:0x0181 A[Catch: IOException -> 0x0184, TRY_LEAVE, TryCatch #11 {IOException -> 0x0184, blocks: (B:84:0x017c, B:86:0x0181), top: B:100:0x017c }] */
    /* JADX WARN: Removed duplicated region for block: B:88:0x0186  */
    /* JADX WARN: Removed duplicated region for block: B:96:0x016a A[EXC_TOP_SPLITTER, SYNTHETIC] */
    @Override // android.os.AsyncTask
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public String doInBackground(String[] strArr) {
        FileOutputStream fileOutputStream;
        InputStream inputStream;
        HttpURLConnection httpURLConnection;
        HttpURLConnection httpURLConnection2;
        try {
            HttpURLConnection httpURLConnection3 = (HttpURLConnection) new URL(strArr[0]).openConnection();
            try {
                httpURLConnection3.setReadTimeout(20000);
                httpURLConnection3.setConnectTimeout(20000);
                httpURLConnection3.connect();
                if (httpURLConnection3.getResponseCode() != 200) {
                    String str = "Server returned HTTP " + httpURLConnection3.getResponseCode() + " " + httpURLConnection3.getResponseMessage();
                    httpURLConnection3.disconnect();
                    return str;
                }
                int contentLength = httpURLConnection3.getContentLength();
                if (contentLength > 0) {
                    this.f4884f = String.format("%.1f", Double.valueOf(contentLength / 1048576)) + " MB";
                }
                inputStream = httpURLConnection3.getInputStream();
                try {
                    File file = new File(this.f4879a.getCacheDir() + "/assets/models/");
                    if (!file.exists()) {
                        try {
                            file.mkdirs();
                        } catch (Exception e2) {
                            e = e2;
                            httpURLConnection = httpURLConnection3;
                            fileOutputStream = null;
                            try {
                                String exc = e.toString();
                                if (fileOutputStream != null) {
                                }
                                if (inputStream != null) {
                                }
                                if (httpURLConnection != null) {
                                }
                                return exc;
                            } catch (Throwable th) {
                                th = th;
                                if (fileOutputStream != null) {
                                    try {
                                        fileOutputStream.close();
                                    } catch (IOException unused) {
                                        if (httpURLConnection != null) {
                                            httpURLConnection.disconnect();
                                        }
                                        throw th;
                                    }
                                }
                                if (inputStream != null) {
                                    inputStream.close();
                                }
                                if (httpURLConnection != null) {
                                }
                                throw th;
                            }
                        } catch (Throwable th2) {
                            th = th2;
                            httpURLConnection = httpURLConnection3;
                            fileOutputStream = null;
                            if (fileOutputStream != null) {
                            }
                            if (inputStream != null) {
                            }
                            if (httpURLConnection != null) {
                            }
                            throw th;
                        }
                    }
                    FileOutputStream fileOutputStream2 = new FileOutputStream(this.f4879a.getCacheDir() + "/assets/models/" + this.f4881c + ".glb");
                    try {
                        byte[] bArr = new byte[4096];
                        long j = 0;
                        while (true) {
                            int read = inputStream.read(bArr);
                            if (read != -1) {
                                if (isCancelled()) {
                                    break;
                                }
                                j += read;
                                if (contentLength > 0) {
                                    Integer[] numArr = new Integer[1];
                                    httpURLConnection2 = httpURLConnection3;
                                    try {
                                        numArr[0] = Integer.valueOf((int) ((100 * j) / contentLength));
                                        publishProgress(numArr);
                                    } catch (Exception e3) {
                                        e = e3;
                                        httpURLConnection = httpURLConnection2;
                                        fileOutputStream = fileOutputStream2;
                                        String exc2 = e.toString();
                                        if (fileOutputStream != null) {
                                            try {
                                                fileOutputStream.close();
                                            } catch (IOException unused2) {
                                                if (httpURLConnection != null) {
                                                    httpURLConnection.disconnect();
                                                }
                                                return exc2;
                                            }
                                        }
                                        if (inputStream != null) {
                                            inputStream.close();
                                        }
                                        if (httpURLConnection != null) {
                                        }
                                        return exc2;
                                    } catch (Throwable th3) {
                                        th = th3;
                                        httpURLConnection = httpURLConnection2;
                                        fileOutputStream = fileOutputStream2;
                                        if (fileOutputStream != null) {
                                        }
                                        if (inputStream != null) {
                                        }
                                        if (httpURLConnection != null) {
                                        }
                                        throw th;
                                    }
                                } else {
                                    httpURLConnection2 = httpURLConnection3;
                                }
                                fileOutputStream2.write(bArr, 0, read);
                                httpURLConnection3 = httpURLConnection2;
                            } else {
                                HttpURLConnection httpURLConnection4 = httpURLConnection3;
                                try {
                                    fileOutputStream2.close();
                                    inputStream.close();
                                } catch (IOException unused3) {
                                }
                                httpURLConnection4.disconnect();
                                break;
                            }
                        }
                        return null;
                    } catch (Exception e4) {
                        e = e4;
                        httpURLConnection2 = httpURLConnection3;
                    } catch (Throwable th4) {
                        th = th4;
                        httpURLConnection2 = httpURLConnection3;
                    }
                } catch (Exception e5) {
                    e = e5;
                    httpURLConnection = httpURLConnection3;
                } catch (Throwable th5) {
                    th = th5;
                    httpURLConnection = httpURLConnection3;
                }
                httpURLConnection3.disconnect();
                return null;
            } catch (Exception e6) {
                e = e6;
                httpURLConnection = httpURLConnection3;
                fileOutputStream = null;
                inputStream = null;
            } catch (Throwable th6) {
                th = th6;
                httpURLConnection = httpURLConnection3;
                fileOutputStream = null;
                inputStream = null;
            }
        } catch (Exception e7) {
            e = e7;
            fileOutputStream = null;
            inputStream = null;
            httpURLConnection = null;
        } catch (Throwable th7) {
            th = th7;
            fileOutputStream = null;
            inputStream = null;
            httpURLConnection = null;
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // android.os.AsyncTask
    public void onPostExecute(String str) {
        this.f4882d.release();
        this.f4880b.f(this.f4881c, 101, this.f4883e, this.f4884f);
    }

    @Override // android.os.AsyncTask
    public void onPreExecute() {
        super.onPreExecute();
        PowerManager.WakeLock newWakeLock = ((PowerManager) this.f4879a.getSystemService("power")).newWakeLock(1, j.class.getName());
        this.f4882d = newWakeLock;
        newWakeLock.acquire();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    @Override // android.os.AsyncTask
    public void onProgressUpdate(Integer[] numArr) {
        Integer[] numArr2 = numArr;
        super.onProgressUpdate(numArr2);
        this.f4880b.f(this.f4881c, numArr2[0].intValue(), this.f4883e, this.f4884f);
    }
}