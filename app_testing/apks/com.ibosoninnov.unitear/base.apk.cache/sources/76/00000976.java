package c.d.d.a;

import android.content.Context;
import android.graphics.SurfaceTexture;
import com.google.mediapipe.components.CameraXPreviewHelper;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class c implements SurfaceTexture.OnFrameAvailableListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ CameraXPreviewHelper f4468b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4469c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Context f4470d;

    public /* synthetic */ c(CameraXPreviewHelper cameraXPreviewHelper, SurfaceTexture surfaceTexture, Context context) {
        this.f4468b = cameraXPreviewHelper;
        this.f4469c = surfaceTexture;
        this.f4470d = context;
    }

    @Override // android.graphics.SurfaceTexture.OnFrameAvailableListener
    public final void onFrameAvailable(SurfaceTexture surfaceTexture) {
        this.f4468b.a(this.f4469c, this.f4470d, surfaceTexture);
    }
}