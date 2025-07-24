package b.v;

import android.content.Intent;
import android.net.Uri;

/* compiled from: NavDeepLinkRequest.java */
/* loaded from: classes.dex */
public class i {

    /* renamed from: a  reason: collision with root package name */
    public final Uri f2640a;

    /* renamed from: b  reason: collision with root package name */
    public final String f2641b;

    /* renamed from: c  reason: collision with root package name */
    public final String f2642c;

    public i(Intent intent) {
        Uri data = intent.getData();
        String action = intent.getAction();
        String type = intent.getType();
        this.f2640a = data;
        this.f2641b = action;
        this.f2642c = type;
    }

    public String toString() {
        StringBuilder A = c.b.a.a.a.A("NavDeepLinkRequest", "{");
        if (this.f2640a != null) {
            A.append(" uri=");
            A.append(this.f2640a.toString());
        }
        if (this.f2641b != null) {
            A.append(" action=");
            A.append(this.f2641b);
        }
        if (this.f2642c != null) {
            A.append(" mimetype=");
            A.append(this.f2642c);
        }
        A.append(" }");
        return A.toString();
    }
}