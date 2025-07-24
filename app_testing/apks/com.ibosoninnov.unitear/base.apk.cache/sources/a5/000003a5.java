package b.d.b;

import android.media.Image;
import b.d.b.d1.g1;

/* compiled from: AndroidImageProxy.java */
/* loaded from: classes.dex */
public final class z implements r0 {

    /* renamed from: b  reason: collision with root package name */
    public final Image f1697b;

    /* renamed from: c  reason: collision with root package name */
    public final a[] f1698c;

    /* renamed from: d  reason: collision with root package name */
    public final q0 f1699d;

    /* compiled from: AndroidImageProxy.java */
    /* loaded from: classes.dex */
    public static final class a {
        public a(Image.Plane plane) {
        }
    }

    public z(Image image) {
        this.f1697b = image;
        Image.Plane[] planes = image.getPlanes();
        if (planes != null) {
            this.f1698c = new a[planes.length];
            for (int i = 0; i < planes.length; i++) {
                this.f1698c[i] = new a(planes[i]);
            }
        } else {
            this.f1698c = new a[0];
        }
        this.f1699d = new c0(g1.f1479a, image.getTimestamp(), 0);
    }

    @Override // b.d.b.r0, java.lang.AutoCloseable
    public synchronized void close() {
        this.f1697b.close();
    }

    @Override // b.d.b.r0
    public synchronized int getHeight() {
        return this.f1697b.getHeight();
    }

    @Override // b.d.b.r0
    public synchronized int getWidth() {
        return this.f1697b.getWidth();
    }

    @Override // b.d.b.r0
    public q0 n() {
        return this.f1699d;
    }
}