package c.c.a.m.x.g;

import android.content.Context;
import android.graphics.Bitmap;
import c.c.a.m.t;
import c.c.a.m.v.w;
import java.security.MessageDigest;
import java.util.Objects;

/* compiled from: GifDrawableTransformation.java */
/* loaded from: classes.dex */
public class f implements t<c> {

    /* renamed from: b  reason: collision with root package name */
    public final t<Bitmap> f4044b;

    public f(t<Bitmap> tVar) {
        Objects.requireNonNull(tVar, "Argument must not be null");
        this.f4044b = tVar;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        this.f4044b.a(messageDigest);
    }

    @Override // c.c.a.m.t
    public w<c> b(Context context, w<c> wVar, int i, int i2) {
        c cVar = wVar.get();
        w<Bitmap> eVar = new c.c.a.m.x.c.e(cVar.b(), c.c.a.b.b(context).f3412d);
        w<Bitmap> b2 = this.f4044b.b(context, eVar, i, i2);
        if (!eVar.equals(b2)) {
            eVar.a();
        }
        t<Bitmap> tVar = this.f4044b;
        cVar.f4036b.f4043a.c(tVar, b2.get());
        return wVar;
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof f) {
            return this.f4044b.equals(((f) obj).f4044b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f4044b.hashCode();
    }
}