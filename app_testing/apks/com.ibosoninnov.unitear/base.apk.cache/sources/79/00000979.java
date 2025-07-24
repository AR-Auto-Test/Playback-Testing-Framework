package c.d.d.a;

import android.graphics.SurfaceTexture;
import com.google.mediapipe.components.ExternalTextureConverter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class f implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ExternalTextureConverter f4479b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4480c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ int f4481d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ int f4482e;

    public /* synthetic */ f(ExternalTextureConverter externalTextureConverter, SurfaceTexture surfaceTexture, int i, int i2) {
        this.f4479b = externalTextureConverter;
        this.f4480c = surfaceTexture;
        this.f4481d = i;
        this.f4482e = i2;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.f4479b.c(this.f4480c, this.f4481d, this.f4482e);
    }
}