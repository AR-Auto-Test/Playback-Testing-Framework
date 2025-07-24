package c.c.a.m.x.c;

import android.content.Context;
import android.graphics.Bitmap;

/* compiled from: BitmapTransformation.java */
/* loaded from: classes.dex */
public abstract class f implements c.c.a.m.t<Bitmap> {
    @Override // c.c.a.m.t
    public final c.c.a.m.v.w<Bitmap> b(Context context, c.c.a.m.v.w<Bitmap> wVar, int i, int i2) {
        if (c.c.a.s.j.j(i, i2)) {
            c.c.a.m.v.c0.d dVar = c.c.a.b.b(context).f3412d;
            Bitmap bitmap = wVar.get();
            if (i == Integer.MIN_VALUE) {
                i = bitmap.getWidth();
            }
            if (i2 == Integer.MIN_VALUE) {
                i2 = bitmap.getHeight();
            }
            Bitmap c2 = c(dVar, bitmap, i, i2);
            return bitmap.equals(c2) ? wVar : e.b(c2, dVar);
        }
        throw new IllegalArgumentException("Cannot apply transformation on width: " + i + " or height: " + i2 + " less than or equal to zero and not Target.SIZE_ORIGINAL");
    }

    public abstract Bitmap c(c.c.a.m.v.c0.d dVar, Bitmap bitmap, int i, int i2);
}