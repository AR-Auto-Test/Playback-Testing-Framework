package c.c.a.m.x.e;

import android.graphics.drawable.Drawable;
import c.c.a.m.p;
import c.c.a.m.r;
import c.c.a.m.v.w;

/* compiled from: UnitDrawableDecoder.java */
/* loaded from: classes.dex */
public class e implements r<Drawable, Drawable> {
    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ boolean a(Drawable drawable, p pVar) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public w<Drawable> b(Drawable drawable, int i, int i2, p pVar) {
        Drawable drawable2 = drawable;
        if (drawable2 != null) {
            return new c(drawable2);
        }
        return null;
    }
}