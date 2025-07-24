package c.d.d.a;

import android.graphics.SurfaceTexture;
import android.util.Log;
import android.view.Surface;
import b.d.b.z0;
import com.google.mediapipe.components.CameraXPreviewHelper;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class b implements b.j.i.a {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ SurfaceTexture f4466a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ Surface f4467b;

    public /* synthetic */ b(SurfaceTexture surfaceTexture, Surface surface) {
        this.f4466a = surfaceTexture;
        this.f4467b = surface;
    }

    @Override // b.j.i.a
    public final void accept(Object obj) {
        SurfaceTexture surfaceTexture = this.f4466a;
        Surface surface = this.f4467b;
        int i = CameraXPreviewHelper.a;
        Log.d("CameraXPreviewHelper", "Surface request result: " + ((z0.f) obj));
        surfaceTexture.release();
        surface.release();
    }
}