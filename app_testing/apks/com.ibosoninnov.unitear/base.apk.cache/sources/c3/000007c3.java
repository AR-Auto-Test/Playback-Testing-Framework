package c.c.a.m.w;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.text.TextUtils;
import c.c.a.m.u.d;
import c.c.a.m.w.n;
import java.io.File;
import java.io.FileNotFoundException;

/* compiled from: MediaStoreFileLoader.java */
/* loaded from: classes.dex */
public final class k implements n<Uri, File> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f3853a;

    /* compiled from: MediaStoreFileLoader.java */
    /* loaded from: classes.dex */
    public static final class a implements o<Uri, File> {

        /* renamed from: a  reason: collision with root package name */
        public final Context f3854a;

        public a(Context context) {
            this.f3854a = context;
        }

        @Override // c.c.a.m.w.o
        public n<Uri, File> b(r rVar) {
            return new k(this.f3854a);
        }
    }

    /* compiled from: MediaStoreFileLoader.java */
    /* loaded from: classes.dex */
    public static class b implements c.c.a.m.u.d<File> {

        /* renamed from: b  reason: collision with root package name */
        public static final String[] f3855b = {"_data"};

        /* renamed from: c  reason: collision with root package name */
        public final Context f3856c;

        /* renamed from: d  reason: collision with root package name */
        public final Uri f3857d;

        public b(Context context, Uri uri) {
            this.f3856c = context;
            this.f3857d = uri;
        }

        @Override // c.c.a.m.u.d
        public Class<File> a() {
            return File.class;
        }

        @Override // c.c.a.m.u.d
        public void b() {
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        @Override // c.c.a.m.u.d
        public void e(c.c.a.f fVar, d.a<? super File> aVar) {
            Cursor query = this.f3856c.getContentResolver().query(this.f3857d, f3855b, null, null, null);
            if (query != null) {
                try {
                    r0 = query.moveToFirst() ? query.getString(query.getColumnIndexOrThrow("_data")) : null;
                } finally {
                    query.close();
                }
            }
            if (TextUtils.isEmpty(r0)) {
                StringBuilder x = c.b.a.a.a.x("Failed to find file path for: ");
                x.append(this.f3857d);
                aVar.c(new FileNotFoundException(x.toString()));
                return;
            }
            aVar.f(new File(r0));
        }
    }

    public k(Context context) {
        this.f3853a = context;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        return b.v.u.c.n(uri);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.w.n$a' to match base method */
    @Override // c.c.a.m.w.n
    public n.a<File> b(Uri uri, int i, int i2, c.c.a.m.p pVar) {
        Uri uri2 = uri;
        return new n.a<>(new c.c.a.r.d(uri2), new b(this.f3853a, uri2));
    }
}