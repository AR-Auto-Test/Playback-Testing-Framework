package c.d.d.a;

import android.graphics.SurfaceTexture;
import com.google.mediapipe.components.ExternalTextureConverter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class j implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ ExternalTextureConverter.RenderThread f4490b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4491c;

    public /* synthetic */ j(ExternalTextureConverter.RenderThread renderThread, SurfaceTexture surfaceTexture) {
        this.f4490b = renderThread;
        this.f4491c = surfaceTexture;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.f4490b.renderNext(this.f4491c);
    }
}