package c.c.a.m.x.c;

import android.graphics.Bitmap;
import java.util.Objects;

/* compiled from: BitmapResource.java */
/* loaded from: classes.dex */
public class e implements c.c.a.m.v.w<Bitmap>, c.c.a.m.v.s {

    /* renamed from: b  reason: collision with root package name */
    public final Bitmap f3958b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.v.c0.d f3959c;

    public e(Bitmap bitmap, c.c.a.m.v.c0.d dVar) {
        Objects.requireNonNull(bitmap, "Bitmap must not be null");
        this.f3958b = bitmap;
        Objects.requireNonNull(dVar, "BitmapPool must not be null");
        this.f3959c = dVar;
    }

    public static e b(Bitmap bitmap, c.c.a.m.v.c0.d dVar) {
        if (bitmap == null) {
            return null;
        }
        return new e(bitmap, dVar);
    }

    @Override // c.c.a.m.v.w
    public void a() {
        this.f3959c.d(this.f3958b);
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return c.c.a.s.j.d(this.f3958b);
    }

    @Override // c.c.a.m.v.w
    public Class<Bitmap> d() {
        return Bitmap.class;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.v.w
    public Bitmap get() {
        return this.f3958b;
    }

    @Override // c.c.a.m.v.s
    public void initialize() {
        this.f3958b.prepareToDraw();
    }
}