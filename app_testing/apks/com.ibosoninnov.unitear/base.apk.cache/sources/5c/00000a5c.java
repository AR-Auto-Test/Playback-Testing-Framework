package c.e.b.p000if;

import android.content.Context;
import android.os.AsyncTask;
import android.os.PowerManager;
import android.util.Log;
import c.b.a.a.a;
import c.e.b.gf.c;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/* compiled from: GLBDownloadTask.java */
/* renamed from: c.e.b.if.k  reason: invalid package */
/* loaded from: classes2.dex */
public class k extends AsyncTask<String, Integer, String> {

    /* renamed from: a  reason: collision with root package name */
    public Context f4885a;

    /* renamed from: b  reason: collision with root package name */
    public c f4886b;

    /* renamed from: c  reason: collision with root package name */
    public String f4887c;

    /* renamed from: d  reason: collision with root package name */
    public PowerManager.WakeLock f4888d;

    /* renamed from: e  reason: collision with root package name */
    public String f4889e = "";

    /* renamed from: f  reason: collision with root package name */
    public String f4890f = "";

    public k(String str, Context context, c cVar) {
        this.f4885a = context;
        this.f4886b = cVar;
        this.f4887c = str;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    /* JADX WARN: Code restructure failed: missing block: B:25:0x0102, code lost:
        r8.close();
        android.util.Log.d("GLBDownloadTask", "doInBackground isCancelled");
     */
    /* JADX WARN: Code restructure failed: missing block: B:26:0x010a, code lost:
        r2.close();
        r8.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:28:0x0111, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:42:0x0138, code lost:
        r16 = r8;
     */
    /* JADX WARN: Code restructure failed: missing block: B:43:0x013a, code lost:
        r2.close();
        r16.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:45:0x0141, code lost:
        r0 = e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:46:0x0142, code lost:
        r2 = c.b.a.a.a.x("doInBackground ");
        r2.append(r0.toString());
        android.util.Log.e("GLBDownloadTask", r2.toString());
     */
    /* JADX WARN: Removed duplicated region for block: B:80:0x01c4  */
    /* JADX WARN: Removed duplicated region for block: B:90:0x01d5 A[Catch: IOException -> 0x01d1, TRY_LEAVE, TryCatch #2 {IOException -> 0x01d1, blocks: (B:86:0x01cd, B:90:0x01d5), top: B:96:0x01cd }] */
    /* JADX WARN: Removed duplicated region for block: B:94:0x01ed  */
    /* JADX WARN: Removed duplicated region for block: B:96:0x01cd A[EXC_TOP_SPLITTER, SYNTHETIC] */
    @Override // android.os.AsyncTask
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public String doInBackground(String[] strArr) {
        FileOutputStream fileOutputStream;
        HttpURLConnection httpURLConnection;
        InputStream inputStream;
        FileOutputStream fileOutputStream2;
        InputStream inputStream2;
        try {
            httpURLConnection = (HttpURLConnection) new URL(strArr[0]).openConnection();
        } catch (Exception e2) {
            e = e2;
            fileOutputStream = null;
            inputStream = null;
            httpURLConnection = null;
        } catch (Throwable th) {
            th = th;
            fileOutputStream = null;
            httpURLConnection = null;
        }
        try {
            httpURLConnection.setReadTimeout(20000);
            httpURLConnection.setConnectTimeout(20000);
            httpURLConnection.connect();
            if (httpURLConnection.getResponseCode() != 200) {
                Log.d("GLBDownloadTask", "doInBackground " + httpURLConnection.getResponseMessage());
                String str = "Server returned HTTP " + httpURLConnection.getResponseCode() + " " + httpURLConnection.getResponseMessage();
                httpURLConnection.disconnect();
                return str;
            }
            int contentLength = httpURLConnection.getContentLength();
            int i = 1;
            if (contentLength > 0) {
                this.f4889e = String.format("%.1f", Double.valueOf(contentLength / 1048576)) + " MB";
            }
            InputStream inputStream3 = httpURLConnection.getInputStream();
            try {
                File file = new File(this.f4885a.getCacheDir() + "/assets/models/");
                if (!file.exists()) {
                    file.mkdirs();
                }
                this.f4890f = this.f4885a.getCacheDir() + "/assets/models/" + this.f4887c + ".glb";
                fileOutputStream = new FileOutputStream(this.f4890f);
                try {
                    byte[] bArr = new byte[4096];
                    long j = 0;
                    while (true) {
                        int read = inputStream3.read(bArr);
                        if (read == -1) {
                            break;
                        } else if (isCancelled()) {
                            break;
                        } else {
                            j += read;
                            if (contentLength > 0) {
                                Integer[] numArr = new Integer[i];
                                inputStream2 = inputStream3;
                                try {
                                    numArr[0] = Integer.valueOf((int) ((100 * j) / contentLength));
                                    publishProgress(numArr);
                                } catch (Exception e3) {
                                    e = e3;
                                    inputStream = inputStream2;
                                    try {
                                        Log.e("GLBDownloadTask", "doInBackground " + e.toString());
                                        String exc = e.toString();
                                        if (fileOutputStream != null) {
                                            try {
                                                fileOutputStream.close();
                                            } catch (IOException e4) {
                                                StringBuilder x = a.x("doInBackground ");
                                                x.append(e4.toString());
                                                Log.e("GLBDownloadTask", x.toString());
                                                if (httpURLConnection != null) {
                                                    httpURLConnection.disconnect();
                                                }
                                                return exc;
                                            }
                                        }
                                        if (inputStream != null) {
                                            inputStream.close();
                                        }
                                        if (httpURLConnection != null) {
                                        }
                                        return exc;
                                    } catch (Throwable th2) {
                                        th = th2;
                                        fileOutputStream2 = fileOutputStream;
                                        Throwable th3 = th;
                                        if (fileOutputStream2 != null) {
                                            try {
                                                fileOutputStream2.close();
                                            } catch (IOException e5) {
                                                StringBuilder x2 = a.x("doInBackground ");
                                                x2.append(e5.toString());
                                                Log.e("GLBDownloadTask", x2.toString());
                                                if (httpURLConnection != null) {
                                                    httpURLConnection.disconnect();
                                                }
                                                throw th3;
                                            }
                                        }
                                        if (inputStream != null) {
                                            inputStream.close();
                                        }
                                        if (httpURLConnection != null) {
                                        }
                                        throw th3;
                                    }
                                } catch (Throwable th4) {
                                    th = th4;
                                    inputStream = inputStream2;
                                    fileOutputStream2 = fileOutputStream;
                                    Throwable th32 = th;
                                    if (fileOutputStream2 != null) {
                                    }
                                    if (inputStream != null) {
                                    }
                                    if (httpURLConnection != null) {
                                    }
                                    throw th32;
                                }
                            } else {
                                inputStream2 = inputStream3;
                            }
                            fileOutputStream.write(bArr, 0, read);
                            inputStream3 = inputStream2;
                            i = 1;
                        }
                    }
                    httpURLConnection.disconnect();
                    return null;
                } catch (Exception e6) {
                    e = e6;
                    inputStream2 = inputStream3;
                } catch (Throwable th5) {
                    th = th5;
                    inputStream2 = inputStream3;
                }
            } catch (Exception e7) {
                e = e7;
                inputStream2 = inputStream3;
                fileOutputStream = null;
            } catch (Throwable th6) {
                th = th6;
                inputStream2 = inputStream3;
                fileOutputStream = null;
            }
        } catch (Exception e8) {
            e = e8;
            fileOutputStream = null;
            inputStream = null;
        } catch (Throwable th7) {
            th = th7;
            fileOutputStream = null;
            inputStream = null;
            fileOutputStream2 = fileOutputStream;
            Throwable th322 = th;
            if (fileOutputStream2 != null) {
            }
            if (inputStream != null) {
            }
            if (httpURLConnection != null) {
            }
            throw th322;
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // android.os.AsyncTask
    public void onPostExecute(String str) {
        Log.d("GLBDownloadTask", "onPostExecute : " + str);
        this.f4888d.release();
        this.f4886b.b(this.f4887c, this.f4890f);
    }

    @Override // android.os.AsyncTask
    public void onPreExecute() {
        super.onPreExecute();
        PowerManager.WakeLock newWakeLock = ((PowerManager) this.f4885a.getSystemService("power")).newWakeLock(1, k.class.getName());
        this.f4888d = newWakeLock;
        newWakeLock.acquire();
        Log.d("GLBDownloadTask", "onPreExecute");
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object[]] */
    @Override // android.os.AsyncTask
    public void onProgressUpdate(Integer[] numArr) {
        Integer[] numArr2 = numArr;
        super.onProgressUpdate(numArr2);
        this.f4886b.a(this.f4887c, numArr2[0].intValue(), this.f4889e);
    }
}