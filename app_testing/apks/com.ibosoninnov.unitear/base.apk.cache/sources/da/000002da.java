package b.d.b;

import android.media.Image;
import android.media.ImageReader;
import android.view.Surface;
import b.d.b.d1.o0;
import java.util.Objects;
import java.util.concurrent.Executor;

/* compiled from: AndroidImageReaderProxy.java */
/* loaded from: classes.dex */
public final class a0 implements b.d.b.d1.o0 {

    /* renamed from: a  reason: collision with root package name */
    public final ImageReader f1378a;

    public a0(ImageReader imageReader) {
        this.f1378a = imageReader;
    }

    @Override // b.d.b.d1.o0
    public synchronized Surface a() {
        return this.f1378a.getSurface();
    }

    @Override // b.d.b.d1.o0
    public synchronized int c() {
        return this.f1378a.getMaxImages();
    }

    @Override // b.d.b.d1.o0
    public synchronized void close() {
        this.f1378a.close();
    }

    @Override // b.d.b.d1.o0
    public synchronized r0 d() {
        Image image;
        try {
            image = this.f1378a.acquireNextImage();
        } catch (RuntimeException e2) {
            if (!"ImageReaderContext is not initialized".equals(e2.getMessage())) {
                throw e2;
            }
            image = null;
        }
        if (image == null) {
            return null;
        }
        return new z(image);
    }

    @Override // b.d.b.d1.o0
    public synchronized void e(final o0.a aVar, final Executor executor) {
        this.f1378a.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() { // from class: b.d.b.b
            @Override // android.media.ImageReader.OnImageAvailableListener
            public final void onImageAvailable(ImageReader imageReader) {
                final a0 a0Var = a0.this;
                Executor executor2 = executor;
                final o0.a aVar2 = aVar;
                Objects.requireNonNull(a0Var);
                executor2.execute(new Runnable() { // from class: b.d.b.c
                    @Override // java.lang.Runnable
                    public final void run() {
                        a0 a0Var2 = a0.this;
                        o0.a aVar3 = aVar2;
                        Objects.requireNonNull(a0Var2);
                        aVar3.a(a0Var2);
                    }
                });
            }
        }, b.d.b.d1.k1.a.a());
    }
}