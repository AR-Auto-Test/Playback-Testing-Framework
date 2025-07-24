package c.e.b;

import android.os.AsyncTask;
import android.util.Log;
import com.google.common.net.HttpHeaders;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;

/* compiled from: UploadImageFormData.java */
/* loaded from: classes2.dex */
public class bf extends AsyncTask<Object, String, String> {

    /* renamed from: a  reason: collision with root package name */
    public String f4581a;

    /* renamed from: b  reason: collision with root package name */
    public String f4582b;

    /* renamed from: c  reason: collision with root package name */
    public a f4583c;

    /* compiled from: UploadImageFormData.java */
    /* loaded from: classes2.dex */
    public interface a {
    }

    public bf(a aVar, String str, String str2) {
        this.f4583c = aVar;
        this.f4581a = str;
        this.f4582b = str2;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // android.os.AsyncTask
    public String doInBackground(Object[] objArr) {
        try {
            String str = (String) objArr[0];
            Log.d("UploadImageFormData", this.f4581a);
            HttpURLConnection httpURLConnection = (HttpURLConnection) new URL(this.f4581a).openConnection();
            httpURLConnection.setDoInput(true);
            httpURLConnection.setDoOutput(true);
            httpURLConnection.setUseCaches(false);
            httpURLConnection.setRequestProperty(HttpHeaders.CONTENT_TYPE, "multipart/form-data;boundary=------WebKitFormBoundary7MA4YWxkTrZu0gW");
            httpURLConnection.setRequestMethod("POST");
            httpURLConnection.setRequestProperty(HttpHeaders.CONNECTION, "Keep-Alive");
            httpURLConnection.setRequestProperty(HttpHeaders.AUTHORIZATION, this.f4582b);
            httpURLConnection.setConnectTimeout(20000);
            httpURLConnection.setReadTimeout(20000);
            DataOutputStream dataOutputStream = new DataOutputStream(httpURLConnection.getOutputStream());
            dataOutputStream.writeBytes("Content-Disposition: form-data; name=image; filename=" + str + "\r\n\r\n");
            FileInputStream fileInputStream = new FileInputStream(str);
            int min = Math.min(fileInputStream.available(), 1048576);
            byte[] bArr = new byte[min];
            int read = fileInputStream.read(bArr, 0, min);
            while (read > 0) {
                dataOutputStream.write(bArr, 0, min);
                min = Math.min(fileInputStream.available(), 1048576);
                read = fileInputStream.read(bArr, 0, min);
            }
            dataOutputStream.writeBytes("\r\n--------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n");
            int responseCode = httpURLConnection.getResponseCode();
            Log.d("UploadImageFormData", "Response Code " + responseCode + " " + httpURLConnection.getResponseMessage());
            String str2 = null;
            if (responseCode == 200) {
                StringBuilder sb = new StringBuilder();
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new BufferedInputStream(httpURLConnection.getInputStream())));
                while (true) {
                    String readLine = bufferedReader.readLine();
                    if (readLine == null) {
                        break;
                    }
                    sb.append(readLine);
                }
                str2 = sb.toString();
            } else {
                a aVar = this.f4583c;
                ((gc) aVar).b(responseCode + " " + httpURLConnection.getResponseMessage());
            }
            fileInputStream.close();
            dataOutputStream.flush();
            dataOutputStream.close();
            if (str2 != null) {
                Log.d("UploadImageFormData", str2);
                ((gc) this.f4583c).a(str2);
                return "";
            }
            return "";
        } catch (SocketTimeoutException e2) {
            Log.e("UploadImageFormData", e2.toString());
            ((gc) this.f4583c).b("timeout");
            return "";
        } catch (Exception e3) {
            Log.e("UploadImageFormData", e3.toString());
            ((gc) this.f4583c).b(e3.getLocalizedMessage());
            return "";
        }
    }
}