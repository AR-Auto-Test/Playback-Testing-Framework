package c.c.a.m.u.o;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import c.c.a.f;
import c.c.a.m.u.d;
import c.c.a.m.u.g;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/* compiled from: ThumbFetcher.java */
/* loaded from: classes.dex */
public class b implements c.c.a.m.u.d<InputStream> {

    /* renamed from: b  reason: collision with root package name */
    public final Uri f3575b;

    /* renamed from: c  reason: collision with root package name */
    public final d f3576c;

    /* renamed from: d  reason: collision with root package name */
    public InputStream f3577d;

    /* compiled from: ThumbFetcher.java */
    /* loaded from: classes.dex */
    public static class a implements c {

        /* renamed from: a  reason: collision with root package name */
        public static final String[] f3578a = {"_data"};

        /* renamed from: b  reason: collision with root package name */
        public final ContentResolver f3579b;

        public a(ContentResolver contentResolver) {
            this.f3579b = contentResolver;
        }

        @Override // c.c.a.m.u.o.c
        public Cursor a(Uri uri) {
            return this.f3579b.query(MediaStore.Images.Thumbnails.EXTERNAL_CONTENT_URI, f3578a, "kind = 1 AND image_id = ?", new String[]{uri.getLastPathSegment()}, null);
        }
    }

    /* compiled from: ThumbFetcher.java */
    /* renamed from: c.c.a.m.u.o.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0065b implements c {

        /* renamed from: a  reason: collision with root package name */
        public static final String[] f3580a = {"_data"};

        /* renamed from: b  reason: collision with root package name */
        public final ContentResolver f3581b;

        public C0065b(ContentResolver contentResolver) {
            this.f3581b = contentResolver;
        }

        @Override // c.c.a.m.u.o.c
        public Cursor a(Uri uri) {
            return this.f3581b.query(MediaStore.Video.Thumbnails.EXTERNAL_CONTENT_URI, f3580a, "kind = 1 AND video_id = ?", new String[]{uri.getLastPathSegment()}, null);
        }
    }

    public b(Uri uri, d dVar) {
        this.f3575b = uri;
        this.f3576c = dVar;
    }

    public static b c(Context context, Uri uri, c cVar) {
        return new b(uri, new d(c.c.a.b.b(context).f3415g.e(), cVar, c.c.a.b.b(context).f3416h, context.getContentResolver()));
    }

    @Override // c.c.a.m.u.d
    public Class<InputStream> a() {
        return InputStream.class;
    }

    @Override // c.c.a.m.u.d
    public void b() {
        InputStream inputStream = this.f3577d;
        if (inputStream != null) {
            try {
                inputStream.close();
            } catch (IOException unused) {
            }
        }
    }

    @Override // c.c.a.m.u.d
    public void cancel() {
    }

    @Override // c.c.a.m.u.d
    public c.c.a.m.a d() {
        return c.c.a.m.a.LOCAL;
    }

    @Override // c.c.a.m.u.d
    public void e(f fVar, d.a<? super InputStream> aVar) {
        try {
            InputStream f2 = f();
            this.f3577d = f2;
            aVar.f(f2);
        } catch (FileNotFoundException e2) {
            if (Log.isLoggable("MediaStoreThumbFetcher", 3)) {
                Log.d("MediaStoreThumbFetcher", "Failed to find thumbnail file", e2);
            }
            aVar.c(e2);
        }
    }

    /* JADX DEBUG: Failed to insert an additional move for type inference into block B:15:0x002b */
    /* JADX WARN: Code restructure failed: missing block: B:13:0x0028, code lost:
        if (r6 != null) goto L56;
     */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x004b, code lost:
        if (r6 != null) goto L56;
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x004d, code lost:
        r6.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:24:0x0050, code lost:
        r7 = null;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Not initialized variable reg: 6, insn: 0x0023: MOVE  (r5 I:??[OBJECT, ARRAY]) = (r6 I:??[OBJECT, ARRAY]), block:B:10:0x0023 */
    /* JADX WARN: Removed duplicated region for block: B:39:0x007f  */
    /* JADX WARN: Removed duplicated region for block: B:60:0x00c8  */
    /* JADX WARN: Removed duplicated region for block: B:66:0x00f7  */
    /* JADX WARN: Removed duplicated region for block: B:85:? A[RETURN, SYNTHETIC] */
    /* JADX WARN: Type inference failed for: r3v5, types: [java.lang.Throwable, java.lang.NullPointerException] */
    /* JADX WARN: Type inference failed for: r5v0, types: [java.io.InputStream] */
    /* JADX WARN: Type inference failed for: r5v1, types: [android.database.Cursor] */
    /* JADX WARN: Type inference failed for: r5v2 */
    /* JADX WARN: Type inference failed for: r6v2 */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public final InputStream f() {
        SecurityException e2;
        Cursor cursor;
        ?? r6;
        InputStream openInputStream;
        int i;
        d dVar = this.f3576c;
        Uri uri = this.f3575b;
        Objects.requireNonNull(dVar);
        boolean z = false;
        ?? r5 = 0;
        InputStream inputStream = null;
        try {
            try {
                cursor = dVar.f3583b.a(uri);
            } catch (SecurityException e3) {
                e2 = e3;
                cursor = null;
            } catch (Throwable th) {
                th = th;
                if (r5 != 0) {
                }
                throw th;
            }
            if (cursor != null) {
                try {
                } catch (SecurityException e4) {
                    e2 = e4;
                    if (Log.isLoggable("ThumbStreamOpener", 3)) {
                        Log.d("ThumbStreamOpener", "Failed to query for thumbnail for Uri: " + uri, e2);
                    }
                }
                if (cursor.moveToFirst()) {
                    String str = cursor.getString(0);
                    cursor.close();
                    if (!TextUtils.isEmpty(str)) {
                        File file = new File(str);
                        if (file.exists() && 0 < file.length()) {
                            z = true;
                        }
                        if (z) {
                            Uri fromFile = Uri.fromFile(file);
                            try {
                                openInputStream = dVar.f3585d.openInputStream(fromFile);
                                if (openInputStream != null) {
                                    d dVar2 = this.f3576c;
                                    Uri uri2 = this.f3575b;
                                    Objects.requireNonNull(dVar2);
                                    try {
                                        try {
                                            inputStream = dVar2.f3585d.openInputStream(uri2);
                                            i = b.v.u.c.i(dVar2.f3586e, inputStream, dVar2.f3584c);
                                            if (inputStream != null) {
                                                try {
                                                    inputStream.close();
                                                } catch (IOException unused) {
                                                }
                                            }
                                        } catch (IOException | NullPointerException e5) {
                                            if (Log.isLoggable("ThumbStreamOpener", 3)) {
                                                Log.d("ThumbStreamOpener", "Failed to open uri: " + uri2, e5);
                                            }
                                            if (inputStream != null) {
                                                try {
                                                    inputStream.close();
                                                } catch (IOException unused2) {
                                                }
                                            }
                                        }
                                        return i != -1 ? new g(openInputStream, i) : openInputStream;
                                    } catch (Throwable th2) {
                                        if (0 != 0) {
                                            try {
                                                r5.close();
                                            } catch (IOException unused3) {
                                            }
                                        }
                                        throw th2;
                                    }
                                }
                                i = -1;
                                if (i != -1) {
                                }
                            } catch (NullPointerException e6) {
                                throw ((FileNotFoundException) new FileNotFoundException("NPE opening uri: " + uri + " -> " + fromFile).initCause(e6));
                            }
                        }
                    }
                    openInputStream = null;
                    if (openInputStream != null) {
                    }
                    i = -1;
                    if (i != -1) {
                    }
                }
            }
        } catch (Throwable th3) {
            th = th3;
            r5 = r6;
            if (r5 != 0) {
                r5.close();
            }
            throw th;
        }
    }
}