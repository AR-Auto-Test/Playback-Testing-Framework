package c.c.a.m.x.c;

import android.graphics.Bitmap;

/* compiled from: UnitBitmapDecoder.java */
/* loaded from: classes.dex */
public final class b0 implements c.c.a.m.r<Bitmap, Bitmap> {

    /* compiled from: UnitBitmapDecoder.java */
    /* loaded from: classes.dex */
    public static final class a implements c.c.a.m.v.w<Bitmap> {

        /* renamed from: b  reason: collision with root package name */
        public final Bitmap f3944b;

        public a(Bitmap bitmap) {
            this.f3944b = bitmap;
        }

        @Override // c.c.a.m.v.w
        public void a() {
        }

        @Override // c.c.a.m.v.w
        public int c() {
            return c.c.a.s.j.d(this.f3944b);
        }

        @Override // c.c.a.m.v.w
        public Class<Bitmap> d() {
            return Bitmap.class;
        }

        /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
        @Override // c.c.a.m.v.w
        public Bitmap get() {
            return this.f3944b;
        }
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, c.c.a.m.p] */
    @Override // c.c.a.m.r
    public /* bridge */ /* synthetic */ boolean a(Bitmap bitmap, c.c.a.m.p pVar) {
        return true;
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, int, int, c.c.a.m.p] */
    /* JADX DEBUG: Return type fixed from 'c.c.a.m.v.w' to match base method */
    @Override // c.c.a.m.r
    public c.c.a.m.v.w<Bitmap> b(Bitmap bitmap, int i, int i2, c.c.a.m.p pVar) {
        return new a(bitmap);
    }
}