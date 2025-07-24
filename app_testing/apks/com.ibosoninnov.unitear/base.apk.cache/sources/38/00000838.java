package c.c.a.m.x.e;

import android.graphics.drawable.Drawable;

/* compiled from: NonOwnedDrawableResource.java */
/* loaded from: classes.dex */
public final class c extends b<Drawable> {
    public c(Drawable drawable) {
        super(drawable);
    }

    @Override // c.c.a.m.v.w
    public void a() {
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return Math.max(1, this.f4023b.getIntrinsicHeight() * this.f4023b.getIntrinsicWidth() * 4);
    }

    /* JADX DEBUG: Type inference failed for r0v1. Raw type applied. Possible types: java.lang.Class<?>, java.lang.Class<android.graphics.drawable.Drawable> */
    @Override // c.c.a.m.v.w
    public Class<Drawable> d() {
        return this.f4023b.getClass();
    }
}