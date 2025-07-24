package c.c.a.m.w.y;

import android.content.Context;
import android.net.Uri;
import c.c.a.m.p;
import c.c.a.m.u.o.b;
import c.c.a.m.w.n;
import c.c.a.m.w.o;
import c.c.a.m.w.r;
import c.c.a.m.x.c.c0;
import java.io.InputStream;

/* compiled from: MediaStoreVideoThumbLoader.java */
/* loaded from: classes.dex */
public class c implements n<Uri, InputStream> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f3911a;

    /* compiled from: MediaStoreVideoThumbLoader.java */
    /* loaded from: classes.dex */
    public static class a implements o<Uri, InputStream> {

        /* renamed from: a  reason: collision with root package name */
        public final Context f3912a;

        public a(Context context) {
            this.f3912a = context;
        }

        @Override // c.c.a.m.w.o
        public n<Uri, InputStream> b(r rVar) {
            return new c(this.f3912a);
        }
    }

    public c(Context context) {
        this.f3911a = context.getApplicationContext();
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        Uri uri2 = uri;
        return b.v.u.c.n(uri2) && uri2.getPathSegments().contains("video");
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.w.n$a' to match base method */
    @Override // c.c.a.m.w.n
    public n.a<InputStream> b(Uri uri, int i, int i2, p pVar) {
        Uri uri2 = uri;
        if (b.v.u.c.o(i, i2)) {
            Long l = (Long) pVar.c(c0.f3948a);
            if (l != null && l.longValue() == -1) {
                c.c.a.r.d dVar = new c.c.a.r.d(uri2);
                Context context = this.f3911a;
                return new n.a<>(dVar, c.c.a.m.u.o.b.c(context, uri2, new b.C0065b(context.getContentResolver())));
            }
        }
        return null;
    }
}