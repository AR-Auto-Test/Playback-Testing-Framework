package c.c.a.m.x.c;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;

/* compiled from: BitmapDrawableDecoder.java */
/* loaded from: classes.dex */
public class a<DataType> implements c.c.a.m.r<DataType, BitmapDrawable> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.r<DataType, Bitmap> f3936a;

    /* renamed from: b  reason: collision with root package name */
    public final Resources f3937b;

    public a(Resources resources, c.c.a.m.r<DataType, Bitmap> rVar) {
        this.f3937b = resources;
        this.f3936a = rVar;
    }

    @Override // c.c.a.m.r
    public boolean a(DataType datatype, c.c.a.m.p pVar) {
        return this.f3936a.a(datatype, pVar);
    }

    @Override // c.c.a.m.r
    public c.c.a.m.v.w<BitmapDrawable> b(DataType datatype, int i, int i2, c.c.a.m.p pVar) {
        return u.b(this.f3937b, this.f3936a.b(datatype, i, i2, pVar));
    }
}