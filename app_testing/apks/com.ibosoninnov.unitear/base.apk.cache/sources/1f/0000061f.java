package c.a.a.a0;

import android.content.Context;
import c.a.a.d;
import c.a.a.e;
import c.a.a.p;
import com.google.firebase.crashlytics.internal.settings.DefaultSettingsSpiCall;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.zip.ZipInputStream;

/* compiled from: NetworkFetcher.java */
/* loaded from: classes.dex */
public class c {

    /* renamed from: a  reason: collision with root package name */
    public final Context f2951a;

    /* renamed from: b  reason: collision with root package name */
    public final String f2952b;

    /* renamed from: c  reason: collision with root package name */
    public final b f2953c;

    public c(Context context, String str, String str2) {
        Context applicationContext = context.getApplicationContext();
        this.f2951a = applicationContext;
        this.f2952b = str;
        if (str2 == null) {
            this.f2953c = null;
        } else {
            this.f2953c = new b(applicationContext);
        }
    }

    public final p<d> a() {
        StringBuilder x = c.b.a.a.a.x("Fetching ");
        x.append(this.f2952b);
        c.a.a.c0.c.a(x.toString());
        HttpURLConnection httpURLConnection = (HttpURLConnection) new URL(this.f2952b).openConnection();
        httpURLConnection.setRequestMethod("GET");
        try {
            httpURLConnection.connect();
            if (httpURLConnection.getErrorStream() == null && httpURLConnection.getResponseCode() == 200) {
                p<d> c2 = c(httpURLConnection);
                StringBuilder sb = new StringBuilder();
                sb.append("Completed fetch from network. Success: ");
                sb.append(c2.f3122a != null);
                c.a.a.c0.c.a(sb.toString());
                return c2;
            }
            String b2 = b(httpURLConnection);
            return new p<>(new IllegalArgumentException("Unable to fetch " + this.f2952b + ". Failed with " + httpURLConnection.getResponseCode() + "\n" + b2));
        } catch (Exception e2) {
            return new p<>(e2);
        } finally {
            httpURLConnection.disconnect();
        }
    }

    public final String b(HttpURLConnection httpURLConnection) {
        httpURLConnection.getResponseCode();
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(httpURLConnection.getErrorStream()));
        StringBuilder sb = new StringBuilder();
        while (true) {
            try {
                try {
                    String readLine = bufferedReader.readLine();
                    if (readLine != null) {
                        sb.append(readLine);
                        sb.append('\n');
                    } else {
                        try {
                            break;
                        } catch (Exception unused) {
                        }
                    }
                } catch (Throwable th) {
                    try {
                        bufferedReader.close();
                    } catch (Exception unused2) {
                    }
                    throw th;
                }
            } catch (Exception e2) {
                throw e2;
            }
        }
        bufferedReader.close();
        return sb.toString();
    }

    public final p<d> c(HttpURLConnection httpURLConnection) {
        a aVar;
        p<d> b2;
        String contentType = httpURLConnection.getContentType();
        if (contentType == null) {
            contentType = DefaultSettingsSpiCall.ACCEPT_JSON_VALUE;
        }
        if (contentType.contains("application/zip")) {
            c.a.a.c0.c.a("Handling zip response.");
            aVar = a.ZIP;
            b bVar = this.f2953c;
            if (bVar == null) {
                b2 = e.d(new ZipInputStream(httpURLConnection.getInputStream()), null);
            } else {
                b2 = e.d(new ZipInputStream(new FileInputStream(bVar.c(this.f2952b, httpURLConnection.getInputStream(), aVar))), this.f2952b);
            }
        } else {
            c.a.a.c0.c.a("Received json response.");
            aVar = a.JSON;
            b bVar2 = this.f2953c;
            if (bVar2 == null) {
                b2 = e.b(httpURLConnection.getInputStream(), null);
            } else {
                b2 = e.b(new FileInputStream(new File(bVar2.c(this.f2952b, httpURLConnection.getInputStream(), aVar).getAbsolutePath())), this.f2952b);
            }
        }
        b bVar3 = this.f2953c;
        if (bVar3 != null && b2.f3122a != null) {
            File file = new File(bVar3.b(), b.a(this.f2952b, aVar, true));
            File file2 = new File(file.getAbsolutePath().replace(".temp", ""));
            boolean renameTo = file.renameTo(file2);
            c.a.a.c0.c.a("Copying temp file to real file (" + file2 + ")");
            if (!renameTo) {
                StringBuilder x = c.b.a.a.a.x("Unable to rename cache file ");
                x.append(file.getAbsolutePath());
                x.append(" to ");
                x.append(file2.getAbsolutePath());
                x.append(".");
                c.a.a.c0.c.b(x.toString());
            }
        }
        return b2;
    }
}