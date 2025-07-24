package c.d.d.a;

import android.content.Context;
import android.util.Size;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class d implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ CameraXPreviewHelper f4471b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ListenableFuture f4472c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ Size f4473d;

    /* renamed from: e  reason: collision with root package name */
    public final /* synthetic */ CameraHelper.CameraFacing f4474e;

    /* renamed from: f  reason: collision with root package name */
    public final /* synthetic */ Context f4475f;

    /* renamed from: g  reason: collision with root package name */
    public final /* synthetic */ b.t.h f4476g;

    public /* synthetic */ d(CameraXPreviewHelper cameraXPreviewHelper, ListenableFuture listenableFuture, Size size, CameraHelper.CameraFacing cameraFacing, Context context, b.t.h hVar) {
        this.f4471b = cameraXPreviewHelper;
        this.f4472c = listenableFuture;
        this.f4473d = size;
        this.f4474e = cameraFacing;
        this.f4475f = context;
        this.f4476g = hVar;
    }

    @Override // java.lang.Runnable
    public final void run() {
        this.f4471b.c(this.f4472c, this.f4473d, this.f4474e, this.f4475f, this.f4476g);
    }
}