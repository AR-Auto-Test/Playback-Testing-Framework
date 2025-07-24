package c.c.a.m.w.y;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.text.TextUtils;
import c.c.a.f;
import c.c.a.m.p;
import c.c.a.m.u.d;
import c.c.a.m.w.n;
import c.c.a.m.w.o;
import c.c.a.m.w.r;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;

/* compiled from: QMediaStoreUriLoader.java */
/* loaded from: classes.dex */
public final class d<DataT> implements n<Uri, DataT> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f3913a;

    /* renamed from: b  reason: collision with root package name */
    public final n<File, DataT> f3914b;

    /* renamed from: c  reason: collision with root package name */
    public final n<Uri, DataT> f3915c;

    /* renamed from: d  reason: collision with root package name */
    public final Class<DataT> f3916d;

    /* compiled from: QMediaStoreUriLoader.java */
    /* loaded from: classes.dex */
    public static abstract class a<DataT> implements o<Uri, DataT> {

        /* renamed from: a  reason: collision with root package name */
        public final Context f3917a;

        /* renamed from: b  reason: collision with root package name */
        public final Class<DataT> f3918b;

        public a(Context context, Class<DataT> cls) {
            this.f3917a = context;
            this.f3918b = cls;
        }

        @Override // c.c.a.m.w.o
        public final n<Uri, DataT> b(r rVar) {
            return new d(this.f3917a, rVar.b(File.class, this.f3918b), rVar.b(Uri.class, this.f3918b), this.f3918b);
        }
    }

    /* compiled from: QMediaStoreUriLoader.java */
    /* loaded from: classes.dex */
    public static final class b extends a<ParcelFileDescriptor> {
        public b(Context context) {
            super(context, ParcelFileDescriptor.class);
        }
    }

    /* compiled from: QMediaStoreUriLoader.java */
    /* loaded from: classes.dex */
    public static final class c extends a<InputStream> {
        public c(Context context) {
            super(context, InputStream.class);
        }
    }

    /* compiled from: QMediaStoreUriLoader.java */
    /* renamed from: c.c.a.m.w.y.d$d  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static final class C0078d<DataT> implements c.c.a.m.u.d<DataT> {

        /* renamed from: b  reason: collision with root package name */
        public static final String[] f3919b = {"_data"};

        /* renamed from: c  reason: collision with root package name */
        public final Context f3920c;

        /* renamed from: d  reason: collision with root package name */
        public final n<File, DataT> f3921d;

        /* renamed from: e  reason: collision with root package name */
        public final n<Uri, DataT> f3922e;

        /* renamed from: f  reason: collision with root package name */
        public final Uri f3923f;

        /* renamed from: g  reason: collision with root package name */
        public final int f3924g;

        /* renamed from: h  reason: collision with root package name */
        public final int f3925h;
        public final p i;
        public final Class<DataT> j;
        public volatile boolean k;
        public volatile c.c.a.m.u.d<DataT> l;

        public C0078d(Context context, n<File, DataT> nVar, n<Uri, DataT> nVar2, Uri uri, int i, int i2, p pVar, Class<DataT> cls) {
            this.f3920c = context.getApplicationContext();
            this.f3921d = nVar;
            this.f3922e = nVar2;
            this.f3923f = uri;
            this.f3924g = i;
            this.f3925h = i2;
            this.i = pVar;
            this.j = cls;
        }

        @Override // c.c.a.m.u.d
        public Class<DataT> a() {
            return this.j;
        }

        @Override // c.c.a.m.u.d
        public void b() {
            c.c.a.m.u.d<DataT> dVar = this.l;
            if (dVar != null) {
                dVar.b();
            }
        }

        public final c.c.a.m.u.d<DataT> c() {
            n.a<DataT> b2;
            Cursor cursor = null;
            if (Environment.isExternalStorageLegacy()) {
                n<File, DataT> nVar = this.f3921d;
                Uri uri = this.f3923f;
                try {
                    Cursor query = this.f3920c.getContentResolver().query(uri, f3919b, null, null, null);
                    if (query != null) {
                        try {
                            if (query.moveToFirst()) {
                                String string = query.getString(query.getColumnIndexOrThrow("_data"));
                                if (!TextUtils.isEmpty(string)) {
                                    File file = new File(string);
                                    query.close();
                                    b2 = nVar.b(file, this.f3924g, this.f3925h, this.i);
                                } else {
                                    throw new FileNotFoundException("File path was empty in media store for: " + uri);
                                }
                            }
                        } catch (Throwable th) {
                            th = th;
                            cursor = query;
                            if (cursor != null) {
                                cursor.close();
                            }
                            throw th;
                        }
                    }
                    throw new FileNotFoundException("Failed to media store entry for: " + uri);
                } catch (Throwable th2) {
                    th = th2;
                }
            } else {
                b2 = this.f3922e.b(this.f3920c.checkSelfPermission("android.permission.ACCESS_MEDIA_LOCATION") == 0 ? MediaStore.setRequireOriginal(this.f3923f) : this.f3923f, this.f3924g, this.f3925h, this.i);
            }
            if (b2 != null) {
                return b2.f3865c;
            }
            return null;
        }

        @Override // c.c.a.m.u.d
        public void cancel() {
            this.k = true;
            c.c.a.m.u.d<DataT> dVar = this.l;
            if (dVar != null) {
                dVar.cancel();
            }
        }

        @Override // c.c.a.m.u.d
        public c.c.a.m.a d() {
            return c.c.a.m.a.LOCAL;
        }

        @Override // c.c.a.m.u.d
        public void e(f fVar, d.a<? super DataT> aVar) {
            try {
                c.c.a.m.u.d<DataT> c2 = c();
                if (c2 == null) {
                    aVar.c(new IllegalArgumentException("Failed to build fetcher for: " + this.f3923f));
                    return;
                }
                this.l = c2;
                if (this.k) {
                    cancel();
                } else {
                    c2.e(fVar, aVar);
                }
            } catch (FileNotFoundException e2) {
                aVar.c(e2);
            }
        }
    }

    public d(Context context, n<File, DataT> nVar, n<Uri, DataT> nVar2, Class<DataT> cls) {
        this.f3913a = context.getApplicationContext();
        this.f3914b = nVar;
        this.f3915c = nVar2;
        this.f3916d = cls;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // c.c.a.m.w.n
    public boolean a(Uri uri) {
        return Build.VERSION.SDK_INT >= 29 && b.v.u.c.n(uri);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    @Override // c.c.a.m.w.n
    public n.a b(Uri uri, int i, int i2, p pVar) {
        Uri uri2 = uri;
        return new n.a(new c.c.a.r.d(uri2), new C0078d(this.f3913a, this.f3914b, this.f3915c, uri2, i, i2, pVar, this.f3916d));
    }
}