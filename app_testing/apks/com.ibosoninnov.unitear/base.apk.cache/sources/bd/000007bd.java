package c.c.a.m.w;

import android.net.Uri;
import android.text.TextUtils;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Objects;

/* compiled from: GlideUrl.java */
/* loaded from: classes.dex */
public class g implements c.c.a.m.m {

    /* renamed from: b  reason: collision with root package name */
    public final h f3839b;

    /* renamed from: c  reason: collision with root package name */
    public final URL f3840c;

    /* renamed from: d  reason: collision with root package name */
    public final String f3841d;

    /* renamed from: e  reason: collision with root package name */
    public String f3842e;

    /* renamed from: f  reason: collision with root package name */
    public URL f3843f;

    /* renamed from: g  reason: collision with root package name */
    public volatile byte[] f3844g;

    /* renamed from: h  reason: collision with root package name */
    public int f3845h;

    public g(URL url) {
        h hVar = h.f3846a;
        Objects.requireNonNull(url, "Argument must not be null");
        this.f3840c = url;
        this.f3841d = null;
        Objects.requireNonNull(hVar, "Argument must not be null");
        this.f3839b = hVar;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        if (this.f3844g == null) {
            this.f3844g = c().getBytes(c.c.a.m.m.f3537a);
        }
        messageDigest.update(this.f3844g);
    }

    public String c() {
        String str = this.f3841d;
        if (str != null) {
            return str;
        }
        URL url = this.f3840c;
        Objects.requireNonNull(url, "Argument must not be null");
        return url.toString();
    }

    public URL d() {
        if (this.f3843f == null) {
            if (TextUtils.isEmpty(this.f3842e)) {
                String str = this.f3841d;
                if (TextUtils.isEmpty(str)) {
                    URL url = this.f3840c;
                    Objects.requireNonNull(url, "Argument must not be null");
                    str = url.toString();
                }
                this.f3842e = Uri.encode(str, "@#&=*+-_.,:!?()/~'%;$");
            }
            this.f3843f = new URL(this.f3842e);
        }
        return this.f3843f;
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof g) {
            g gVar = (g) obj;
            return c().equals(gVar.c()) && this.f3839b.equals(gVar.f3839b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        if (this.f3845h == 0) {
            int hashCode = c().hashCode();
            this.f3845h = hashCode;
            this.f3845h = this.f3839b.hashCode() + (hashCode * 31);
        }
        return this.f3845h;
    }

    public String toString() {
        return c();
    }

    public g(String str) {
        h hVar = h.f3846a;
        this.f3840c = null;
        if (!TextUtils.isEmpty(str)) {
            this.f3841d = str;
            Objects.requireNonNull(hVar, "Argument must not be null");
            this.f3839b = hVar;
            return;
        }
        throw new IllegalArgumentException("Must not be null or empty");
    }
}