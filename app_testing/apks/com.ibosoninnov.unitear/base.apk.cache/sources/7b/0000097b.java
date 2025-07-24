package c.d.d.a;

import android.graphics.SurfaceTexture;
import com.google.mediapipe.components.ExternalTextureConverter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class h implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ExternalTextureConverter f4485b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4486c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ int f4487d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ int f4488e;

    public /* synthetic */ h(ExternalTextureConverter externalTextureConverter, SurfaceTexture surfaceTexture, int i, int i2) {
        this.f4485b = externalTextureConverter;
        this.f4486c = surfaceTexture;
        this.f4487d = i;
        this.f4488e = i2;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.f4485b.b(this.f4486c, this.f4487d, this.f4488e);
    }
}