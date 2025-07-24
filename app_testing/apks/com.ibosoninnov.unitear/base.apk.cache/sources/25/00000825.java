package c.c.a.m.x.c;

import android.graphics.Bitmap;
import java.security.MessageDigest;

/* compiled from: FitCenter.java */
/* loaded from: classes.dex */
public class q extends f {

    /* renamed from: b  reason: collision with root package name */
    public static final byte[] f3987b = "com.bumptech.glide.load.resource.bitmap.FitCenter".getBytes(c.c.a.m.m.f3537a);

    @Override // c.c.a.m.m
    public void a(MessageDigest messageDigest) {
        messageDigest.update(f3987b);
    }

    @Override // c.c.a.m.x.c.f
    public Bitmap c(c.c.a.m.v.c0.d dVar, Bitmap bitmap, int i, int i2) {
        return a0.b(dVar, bitmap, i, i2);
    }

    @Override // c.c.a.m.m
    public boolean equals(Object obj) {
        return obj instanceof q;
    }

    @Override // c.c.a.m.m
    public int hashCode() {
        return 1572326941;
    }
}