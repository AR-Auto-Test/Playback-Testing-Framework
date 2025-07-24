package b.j.g;

import android.util.Base64;
import java.util.List;
import java.util.Objects;

/* compiled from: FontRequest.java */
/* loaded from: classes.dex */
public final class e {

    /* renamed from: a  reason: collision with root package name */
    public final String f2129a;

    /* renamed from: b  reason: collision with root package name */
    public final String f2130b;

    /* renamed from: c  reason: collision with root package name */
    public final String f2131c;

    /* renamed from: d  reason: collision with root package name */
    public final List<List<byte[]>> f2132d;

    /* renamed from: e  reason: collision with root package name */
    public final String f2133e;

    public e(String str, String str2, String str3, List<List<byte[]>> list) {
        this.f2129a = str;
        this.f2130b = str2;
        this.f2131c = str3;
        Objects.requireNonNull(list);
        this.f2132d = list;
        this.f2133e = str + "-" + str2 + "-" + str3;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        StringBuilder x = c.b.a.a.a.x("FontRequest {mProviderAuthority: ");
        x.append(this.f2129a);
        x.append(", mProviderPackage: ");
        x.append(this.f2130b);
        x.append(", mQuery: ");
        x.append(this.f2131c);
        x.append(", mCertificates:");
        sb.append(x.toString());
        for (int i = 0; i < this.f2132d.size(); i++) {
            sb.append(" [");
            List<byte[]> list = this.f2132d.get(i);
            for (int i2 = 0; i2 < list.size(); i2++) {
                sb.append(" \"");
                sb.append(Base64.encodeToString(list.get(i2), 0));
                sb.append("\"");
            }
            sb.append(" ]");
        }
        return c.b.a.a.a.v(sb, "}", "mCertificatesArray: 0");
    }
}