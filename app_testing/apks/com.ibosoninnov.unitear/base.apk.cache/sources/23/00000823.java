package c.c.a.m.x.c;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import java.security.MessageDigest;

/* compiled from: DrawableTransformation.java */
/* loaded from: classes.dex */
public class o implements c.c.a.m.t<Drawable> {

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.t<Bitmap> f3985b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f3986c;

    public o(c.c.a.m.t<Bitmap> tVar, boolean z) {
        this.f3985b = tVar;
        this.f3986c = z;
    }

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        this.f3985b.a(messageDigest);
    }

    @Override // c.c.a.m.t
    public c.c.a.m.v.w<Drawable> b(Context context, c.c.a.m.v.w<Drawable> wVar, int i, int i2) {
        c.c.a.m.v.c0.d dVar = c.c.a.b.b(context).f3412d;
        Drawable drawable = wVar.get();
        c.c.a.m.v.w<Bitmap> a2 = n.a(dVar, drawable, i, i2);
        if (a2 == null) {
            if (this.f3986c) {
                throw new IllegalArgumentException("Unable to convert " + drawable + " to a Bitmap");
            }
            return wVar;
        }
        c.c.a.m.v.w<Bitmap> b2 = this.f3985b.b(context, a2, i, i2);
        if (b2.equals(a2)) {
            b2.a();
            return wVar;
        }
        return u.b(context.getResources(), b2);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        if (obj instanceof o) {
            return this.f3985b.equals(((o) obj).f3985b);
        }
        return false;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return this.f3985b.hashCode();
    }
}