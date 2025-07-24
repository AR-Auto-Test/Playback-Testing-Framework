package c.c.a.m.v.c0;

import android.graphics.Bitmap;

/* compiled from: BitmapPoolAdapter.java */
/* loaded from: classes.dex */
public class e implements d {
    @Override // c.c.a.m.v.c0.d
    public void a(int i) {
    }

    @Override // c.c.a.m.v.c0.d
    public void b() {
    }

    @Override // c.c.a.m.v.c0.d
    public Bitmap c(int i, int i2, Bitmap.Config config) {
        return Bitmap.createBitmap(i, i2, config);
    }

    @Override // c.c.a.m.v.c0.d
    public void d(Bitmap bitmap) {
        bitmap.recycle();
    }

    @Override // c.c.a.m.v.c0.d
    public Bitmap e(int i, int i2, Bitmap.Config config) {
        return Bitmap.createBitmap(i, i2, config);
    }
}