package c.c.a.m.x.h;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import c.c.a.m.p;
import c.c.a.m.v.w;

/* compiled from: DrawableBytesTranscoder.java */
/* loaded from: classes.dex */
public final class c implements e<Drawable, byte[]> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f4067a;

    /* renamed from: b  reason: collision with root package name */
    public final e<Bitmap, byte[]> f4068b;

    /* renamed from: c  reason: collision with root package name */
    public final e<c.c.a.m.x.g.c, byte[]> f4069c;

    public c(c.c.a.m.v.c0.d dVar, e<Bitmap, byte[]> eVar, e<c.c.a.m.x.g.c, byte[]> eVar2) {
        this.f4067a = dVar;
        this.f4068b = eVar;
        this.f4069c = eVar2;
    }

    @Override // c.c.a.m.x.h.e
    public w<byte[]> a(w<Drawable> wVar, p pVar) {
        Drawable drawable = wVar.get();
        if (drawable instanceof BitmapDrawable) {
            return this.f4068b.a(c.c.a.m.x.c.e.b(((BitmapDrawable) drawable).getBitmap(), this.f4067a), pVar);
        }
        if (drawable instanceof c.c.a.m.x.g.c) {
            return this.f4069c.a(wVar, pVar);
        }
        return null;
    }
}