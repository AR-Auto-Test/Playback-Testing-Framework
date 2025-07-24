package c.d.d.a;

import android.graphics.SurfaceTexture;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class e implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ CameraHelper.OnCameraStartedListener f4477b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4478c;

    public /* synthetic */ e(CameraHelper.OnCameraStartedListener onCameraStartedListener, SurfaceTexture surfaceTexture) {
        this.f4477b = onCameraStartedListener;
        this.f4478c = surfaceTexture;
    }

    @Override // java.lang.Runnable
    public final void run() {
        CameraHelper.OnCameraStartedListener onCameraStartedListener = this.f4477b;
        SurfaceTexture surfaceTexture = this.f4478c;
        int i = CameraXPreviewHelper.a;
        onCameraStartedListener.onCameraStarted(surfaceTexture);
    }
}