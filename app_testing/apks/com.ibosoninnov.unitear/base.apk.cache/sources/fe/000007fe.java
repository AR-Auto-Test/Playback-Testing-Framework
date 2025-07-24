package c.c.a.m.x.c;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import java.io.File;

/* compiled from: BitmapDrawableEncoder.java */
/* loaded from: classes.dex */
public class b implements c.c.a.m.s<BitmapDrawable> {

    /* renamed from: a  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f3942a;

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.m.s<Bitmap> f3943b;

    public b(c.c.a.m.v.c0.d dVar, c.c.a.m.s<Bitmap> sVar) {
        this.f3942a = dVar;
        this.f3943b = sVar;
    }

    @Override // c.c.a.m.d
    public boolean a(Object obj, File file, c.c.a.m.p pVar) {
        return this.f3943b.a(new e(((BitmapDrawable) ((c.c.a.m.v.w) obj).get()).getBitmap(), this.f3942a), file, pVar);
    }

    @Override // c.c.a.m.s
    public c.c.a.m.c b(c.c.a.m.p pVar) {
        return this.f3943b.b(pVar);
    }
}