package c.c.a.m.x.c;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import java.util.Objects;

/* compiled from: LazyBitmapDrawableResource.java */
/* loaded from: classes.dex */
public final class u implements c.c.a.m.v.w<BitmapDrawable>, c.c.a.m.v.s {

    /* renamed from: b  reason: collision with root package name */
    public final Resources f4003b;

    /* renamed from: c  reason: collision with root package name */
    public final c.c.a.m.v.w<Bitmap> f4004c;

    public u(Resources resources, c.c.a.m.v.w<Bitmap> wVar) {
        Objects.requireNonNull(resources, "Argument must not be null");
        this.f4003b = resources;
        this.f4004c = wVar;
    }

    public static c.c.a.m.v.w<BitmapDrawable> b(Resources resources, c.c.a.m.v.w<Bitmap> wVar) {
        if (wVar == null) {
            return null;
        }
        return new u(resources, wVar);
    }

    @Override // c.c.a.m.v.w
    public void a() {
        this.f4004c.a();
    }

    @Override // c.c.a.m.v.w
    public int c() {
        return this.f4004c.c();
    }

    @Override // c.c.a.m.v.w
    public Class<BitmapDrawable> d() {
        return BitmapDrawable.class;
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // c.c.a.m.v.w
    public BitmapDrawable get() {
        return new BitmapDrawable(this.f4003b, this.f4004c.get());
    }

    @Override // c.c.a.m.v.s
    public void initialize() {
        c.c.a.m.v.w<Bitmap> wVar = this.f4004c;
        if (wVar instanceof c.c.a.m.v.s) {
            ((c.c.a.m.v.s) wVar).initialize();
        }
    }
}