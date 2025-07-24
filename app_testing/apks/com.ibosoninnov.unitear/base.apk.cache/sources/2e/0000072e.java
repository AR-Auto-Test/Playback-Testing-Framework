package c.c.a.m.u;

import android.content.ContentResolver;
import android.net.Uri;
import android.util.Log;
import c.c.a.m.u.d;
import java.io.FileNotFoundException;
import java.io.IOException;

/* compiled from: LocalUriFetcher.java */
/* loaded from: classes.dex */
public abstract class l<T> implements d<T> {

    /* renamed from: b  reason: collision with root package name */
    public final Uri f3571b;

    /* renamed from: c  reason: collision with root package name */
    public final ContentResolver f3572c;

    /* renamed from: d  reason: collision with root package name */
    public T f3573d;

    public l(ContentResolver contentResolver, Uri uri) {
        this.f3572c = contentResolver;
        this.f3571b = uri;
    }

    @Override // c.c.a.m.u.d
    public void b() {
        T t = this.f3573d;
        if (t != null) {
            try {
                c(t);
            } catch (IOException unused) {
            }
        }
    }

    public abstract void c(T t);

    @Override // c.c.a.m.u.d
    public void cancel() {
    }

    @Override // c.c.a.m.u.d
    public c.c.a.m.a d() {
        return c.c.a.m.a.LOCAL;
    }

    @Override // c.c.a.m.u.d
    public final void e(c.c.a.f fVar, d.a<? super T> aVar) {
        try {
            T f2 = f(this.f3571b, this.f3572c);
            this.f3573d = f2;
            aVar.f(f2);
        } catch (FileNotFoundException e2) {
            if (Log.isLoggable("LocalUriFetcher", 3)) {
                Log.d("LocalUriFetcher", "Failed to open Uri", e2);
            }
            aVar.c(e2);
        }
    }

    public abstract T f(Uri uri, ContentResolver contentResolver);
}