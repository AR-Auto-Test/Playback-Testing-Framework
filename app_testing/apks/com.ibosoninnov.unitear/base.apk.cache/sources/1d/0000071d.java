package c.c.a.m.u;

import android.content.res.AssetManager;
import android.util.Log;
import c.c.a.m.u.d;
import java.io.IOException;

/* compiled from: AssetPathFetcher.java */
/* loaded from: classes.dex */
public abstract class b<T> implements d<T> {

    /* renamed from: b  reason: collision with root package name */
    public final String f3548b;

    /* renamed from: c  reason: collision with root package name */
    public final AssetManager f3549c;

    /* renamed from: d  reason: collision with root package name */
    public T f3550d;

    public b(AssetManager assetManager, String str) {
        this.f3549c = assetManager;
        this.f3548b = str;
    }

    @Override // c.c.a.m.u.d
    public void b() {
        T t = this.f3550d;
        if (t == null) {
            return;
        }
        try {
            c(t);
        } catch (IOException unused) {
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
    public void e(c.c.a.f fVar, d.a<? super T> aVar) {
        try {
            T f2 = f(this.f3549c, this.f3548b);
            this.f3550d = f2;
            aVar.f(f2);
        } catch (IOException e2) {
            if (Log.isLoggable("AssetPathFetcher", 3)) {
                Log.d("AssetPathFetcher", "Failed to load data from asset manager", e2);
            }
            aVar.c(e2);
        }
    }

    public abstract T f(AssetManager assetManager, String str);
}