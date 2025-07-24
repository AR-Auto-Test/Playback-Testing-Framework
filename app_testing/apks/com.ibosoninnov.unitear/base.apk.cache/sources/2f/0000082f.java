package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.net.Uri;

/* compiled from: ResourceBitmapDecoder.java */
/* loaded from: classes.dex */
public class x implements c.c.a.m.r<Uri, Bitmap> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.x.e.d f4012a;

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f4013b;

    public x(c.c.a.m.x.e.d dVar, c.c.a.m.v.c0.d dVar2) {
        this.f4012a = dVar;
        this.f4013b = dVar2;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public boolean a(Uri uri, c.c.a.m.p pVar) {
        return "android.resource".equals(uri.getScheme());
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(Uri uri, int i, int i2, c.c.a.m.p pVar) {
        c.c.a.m.v.w c2 = this.f4012a.c(uri);
        if (c2 == null) {
            return null;
        }
        return n.a(this.f4013b, (Drawable) ((c.c.a.m.x.e.b) c2).get(), i, i2);
    }
}