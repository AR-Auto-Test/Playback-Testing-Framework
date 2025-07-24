package c.c.a.m.u;

import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import c.c.a.m.u.d;
import com.google.common.net.HttpHeaders;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Map;

/* compiled from: HttpUrlFetcher.java */
/* loaded from: classes.dex */
public class j implements d<InputStream> {

    /* renamed from: b  reason: collision with root package name */
    public static final b f3563b = new a();

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.w.g f3564c;

    /* renamed from: d  reason: collision with root package name */
    public final int f3565d;

    /* renamed from: e  reason: collision with root package name */
    public HttpURLConnection f3566e;

    /* renamed from: f  reason: collision with root package name */
    public InputStream f3567f;

    /* renamed from: g  reason: collision with root package name */
    public volatile boolean f3568g;

    /* compiled from: HttpUrlFetcher.java */
    /* loaded from: classes.dex */
    public static class a implements b {
    }

    /* compiled from: HttpUrlFetcher.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public j(c.c.a.m.w.g gVar, int i) {
        this.f3564c = gVar;
        this.f3565d = i;
    }

    public static int c(HttpURLConnection httpURLConnection) {
        try {
            return httpURLConnection.getResponseCode();
        } catch (IOException e2) {
            if (Log.isLoggable("HttpUrlFetcher", 3)) {
                Log.d("HttpUrlFetcher", "Failed to get a response code", e2);
                return -1;
            }
            return -1;
        }
    }

    @Override // c.c.a.m.u.d
    public Class<InputStream> a() {
        return InputStream.class;
    }

    @Override // c.c.a.m.u.d
    public void b() {
        InputStream inputStream = this.f3567f;
        if (inputStream != null) {
            try {
                inputStream.close();
            } catch (IOException unused) {
            }
        }
        HttpURLConnection httpURLConnection = this.f3566e;
        if (httpURLConnection != null) {
            httpURLConnection.disconnect();
        }
        this.f3566e = null;
    }

    @Override // c.c.a.m.u.d
    public void cancel() {
        this.f3568g = true;
    }

    @Override // c.c.a.m.u.d
    public c.c.a.m.a d() {
        return c.c.a.m.a.REMOTE;
    }

    @Override // c.c.a.m.u.d
    public void e(c.c.a.f fVar, d.a<? super InputStream> aVar) {
        StringBuilder sb;
        int i = c.c.a.s.f.f4187b;
        long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        try {
            try {
                aVar.f(f(this.f3564c.d(), 0, null, this.f3564c.f3839b.a()));
            } catch (IOException e2) {
                if (Log.isLoggable("HttpUrlFetcher", 3)) {
                    Log.d("HttpUrlFetcher", "Failed to load data for url", e2);
                }
                aVar.c(e2);
                if (!Log.isLoggable("HttpUrlFetcher", 2)) {
                    return;
                }
                sb = new StringBuilder();
            }
            if (Log.isLoggable("HttpUrlFetcher", 2)) {
                sb = new StringBuilder();
                sb.append("Finished http url fetcher fetch in ");
                sb.append(c.c.a.s.f.a(elapsedRealtimeNanos));
                Log.v("HttpUrlFetcher", sb.toString());
            }
        } catch (Throwable th) {
            if (Log.isLoggable("HttpUrlFetcher", 2)) {
                StringBuilder x = c.b.a.a.a.x("Finished http url fetcher fetch in ");
                x.append(c.c.a.s.f.a(elapsedRealtimeNanos));
                Log.v("HttpUrlFetcher", x.toString());
            }
            throw th;
        }
    }

    public final InputStream f(URL url, int i, URL url2, Map<String, String> map) {
        if (i < 5) {
            if (url2 != null) {
                try {
                    if (url.toURI().equals(url2.toURI())) {
                        throw new c.c.a.m.e("In re-direct loop", -1, null);
                    }
                } catch (URISyntaxException unused) {
                }
            }
            try {
                HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection();
                for (Map.Entry<String, String> entry : map.entrySet()) {
                    httpURLConnection.addRequestProperty(entry.getKey(), entry.getValue());
                }
                httpURLConnection.setConnectTimeout(this.f3565d);
                httpURLConnection.setReadTimeout(this.f3565d);
                httpURLConnection.setUseCaches(false);
                httpURLConnection.setDoInput(true);
                httpURLConnection.setInstanceFollowRedirects(false);
                this.f3566e = httpURLConnection;
                try {
                    httpURLConnection.connect();
                    this.f3567f = this.f3566e.getInputStream();
                    if (this.f3568g) {
                        return null;
                    }
                    int c2 = c(this.f3566e);
                    int i2 = c2 / 100;
                    if (i2 == 2) {
                        HttpURLConnection httpURLConnection2 = this.f3566e;
                        try {
                            if (TextUtils.isEmpty(httpURLConnection2.getContentEncoding())) {
                                this.f3567f = new c.c.a.s.c(httpURLConnection2.getInputStream(), httpURLConnection2.getContentLength());
                            } else {
                                if (Log.isLoggable("HttpUrlFetcher", 3)) {
                                    Log.d("HttpUrlFetcher", "Got non empty content encoding: " + httpURLConnection2.getContentEncoding());
                                }
                                this.f3567f = httpURLConnection2.getInputStream();
                            }
                            return this.f3567f;
                        } catch (IOException e2) {
                            throw new c.c.a.m.e("Failed to obtain InputStream", c(httpURLConnection2), e2);
                        }
                    }
                    if (!(i2 == 3)) {
                        if (c2 == -1) {
                            throw new c.c.a.m.e("Http request failed", c2, null);
                        }
                        try {
                            throw new c.c.a.m.e(this.f3566e.getResponseMessage(), c2, null);
                        } catch (IOException e3) {
                            throw new c.c.a.m.e("Failed to get a response message", c2, e3);
                        }
                    }
                    String headerField = this.f3566e.getHeaderField(HttpHeaders.LOCATION);
                    if (!TextUtils.isEmpty(headerField)) {
                        try {
                            URL url3 = new URL(url, headerField);
                            b();
                            return f(url3, i + 1, url, map);
                        } catch (MalformedURLException e4) {
                            throw new c.c.a.m.e(c.b.a.a.a.q("Bad redirect url: ", headerField), c2, e4);
                        }
                    }
                    throw new c.c.a.m.e("Received empty or null redirect url", c2, null);
                } catch (IOException e5) {
                    throw new c.c.a.m.e("Failed to connect or obtain data", c(this.f3566e), e5);
                }
            } catch (IOException e6) {
                throw new c.c.a.m.e("URL.openConnection threw", 0, e6);
            }
        }
        throw new c.c.a.m.e("Too many (> 5) redirects!", -1, null);
    }
}