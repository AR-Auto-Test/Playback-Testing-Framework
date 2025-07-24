package c.c.a.m.x.h;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import c.c.a.m.p;
import c.c.a.m.v.w;
import c.c.a.m.x.c.u;

/* compiled from: BitmapDrawableTranscoder.java */
/* loaded from: classes.dex */
public class b implements e<Bitmap, BitmapDrawable> {

    /* renamed from: a  reason: collision with root package name */
    public final Resources f4066a;

    public b(Resources resources) {
        this.f4066a = resources;
    }

    @Override // c.c.a.m.x.h.e
    public w<BitmapDrawable> a(w<Bitmap> wVar, p pVar) {
        return u.b(this.f4066a, wVar);
    }
}