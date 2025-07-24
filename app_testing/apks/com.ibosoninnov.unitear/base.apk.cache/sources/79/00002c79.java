package com.ibosoninnov.instanttrackinglib;

import android.app.Activity;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;
import android.view.Surface;
import androidx.recyclerview.widget.RecyclerView;
import b.d.b.d1.i0;
import b.d.b.d1.k1.c.e;
import b.d.b.d1.n0;
import b.d.b.e0;
import b.d.b.j0;
import b.d.b.w0;
import b.d.b.z0;
import b.d.c.c;
import b.j.c.a;
import b.t.h;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.glutil.EglManager;
import com.ibosoninnov.instanttrackinglib.CameraXPreviewHelper;
import java.util.Arrays;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import javax.microedition.khronos.egl.EGLSurface;

/* loaded from: classes2.dex */
public class CameraXPreviewHelper extends CameraHelper {
    private static final int CLOCK_OFFSET_CALIBRATION_ATTEMPTS = 3;
    private static final String TAG = "CameraXPreviewHelper";
    private static final Size TARGET_SIZE = new Size(1280, 720);

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f5657a = 0;
    private e0 camera;
    private c cameraProvider;
    private int frameRotation;
    private Size frameSize;
    private w0 preview;
    private final SingleThreadHandlerExecutor renderExecutor = new SingleThreadHandlerExecutor("RenderThread", 0);
    private CameraCharacteristics cameraCharacteristics = null;
    private float focalLengthPixels = Float.MIN_VALUE;
    private int cameraTimestampSource = 0;

    /* loaded from: classes2.dex */
    public static final class SingleThreadHandlerExecutor implements Executor {
        private final Handler handler;
        private final HandlerThread handlerThread;

        public SingleThreadHandlerExecutor(String str, int i) {
            HandlerThread handlerThread = new HandlerThread(str, i);
            this.handlerThread = handlerThread;
            handlerThread.start();
            this.handler = new Handler(handlerThread.getLooper());
        }

        @Override // java.util.concurrent.Executor
        public void execute(Runnable runnable) {
            if (this.handler.post(runnable)) {
                return;
            }
            throw new RejectedExecutionException(this.handlerThread.getName() + " is shutting down.");
        }

        public Handler getHandler() {
            return this.handler;
        }

        public boolean shutdown() {
            return this.handlerThread.quitSafely();
        }
    }

    private float calculateFocalLengthInPixels() {
        float f2 = ((float[]) this.cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS))[0];
        return (this.frameSize.getWidth() * f2) / ((SizeF) this.cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)).getWidth();
    }

    private static SurfaceTexture createSurfaceTexture() {
        EglManager eglManager = new EglManager(null);
        EGLSurface createOffscreenSurface = eglManager.createOffscreenSurface(1, 1);
        eglManager.makeCurrent(createOffscreenSurface, createOffscreenSurface);
        int[] iArr = new int[1];
        GLES20.glGenTextures(1, iArr, 0);
        return new SurfaceTexture(iArr[0]);
    }

    private static CameraCharacteristics getCameraCharacteristics(Context context, Integer num) {
        CameraManager cameraManager = (CameraManager) context.getSystemService("camera");
        try {
            for (String str : Arrays.asList(cameraManager.getCameraIdList())) {
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(str);
                Integer num2 = (Integer) cameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
                if (num2 != null && num2.equals(num)) {
                    return cameraCharacteristics;
                }
            }
            return null;
        } catch (CameraAccessException e2) {
            Log.e(TAG, "Accessing camera ID info got error: " + e2);
            return null;
        }
    }

    private static long getOffsetFromRealtimeTimestampSource() {
        long j = RecyclerView.FOREVER_NS;
        long j2 = Long.MAX_VALUE;
        for (int i = 0; i < 3; i++) {
            long nanoTime = System.nanoTime();
            long elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
            long nanoTime2 = System.nanoTime();
            long j3 = nanoTime2 - nanoTime;
            if (j3 < j2) {
                j = ((nanoTime + nanoTime2) / 2) - elapsedRealtimeNanos;
                j2 = j3;
            }
        }
        return j;
    }

    private static long getOffsetFromUnknownTimestampSource() {
        return 0L;
    }

    private Size getOptimalViewSize(Size size) {
        Size[] outputSizes;
        CameraCharacteristics cameraCharacteristics = this.cameraCharacteristics;
        if (cameraCharacteristics != null) {
            float f2 = 1000.0f;
            float width = size.getWidth() / size.getHeight();
            int i = -1;
            int i2 = -1;
            for (Size size2 : ((StreamConfigurationMap) cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)).getOutputSizes(SurfaceTexture.class)) {
                float abs = Math.abs((size2.getWidth() / size2.getHeight()) - width);
                if (abs <= f2 && ((i == -1 && i2 == -1) || (size2.getWidth() <= i && size2.getWidth() >= this.frameSize.getWidth() && size2.getHeight() <= i2 && size2.getHeight() >= this.frameSize.getHeight()))) {
                    i = size2.getWidth();
                    i2 = size2.getHeight();
                    f2 = abs;
                }
            }
            if (i == -1 || i2 == -1) {
                return null;
            }
            return new Size(i, i2);
        }
        return null;
    }

    private void onInitialFrameReceived(Context context, final SurfaceTexture surfaceTexture) {
        surfaceTexture.setOnFrameAvailableListener(null);
        surfaceTexture.updateTexImage();
        surfaceTexture.detachFromGLContext();
        if (!this.preview.f1385g.equals(this.frameSize)) {
            this.frameSize = this.preview.f1385g;
            this.frameRotation = this.camera.b().a();
            if (this.frameSize.getWidth() == 0 || this.frameSize.getHeight() == 0) {
                Log.d(TAG, "Invalid frameSize.");
                return;
            }
        }
        CameraCharacteristics cameraCharacteristics = getCameraCharacteristics(context, Integer.valueOf(this.cameraFacing == CameraHelper.CameraFacing.FRONT ? 0 : 1));
        this.cameraCharacteristics = cameraCharacteristics;
        if (cameraCharacteristics != null) {
            this.cameraTimestampSource = ((Integer) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)).intValue();
            this.focalLengthPixels = calculateFocalLengthInPixels();
        }
        final CameraHelper.OnCameraStartedListener onCameraStartedListener = this.onCameraStartedListener;
        if (onCameraStartedListener != null) {
            a.b(context).execute(new Runnable() { // from class: c.e.a.g
                @Override // java.lang.Runnable
                public final void run() {
                    CameraHelper.OnCameraStartedListener onCameraStartedListener2 = CameraHelper.OnCameraStartedListener.this;
                    SurfaceTexture surfaceTexture2 = surfaceTexture;
                    int i = CameraXPreviewHelper.f5657a;
                    onCameraStartedListener2.onCameraStarted(surfaceTexture2);
                }
            });
        }
    }

    public /* synthetic */ void a(SurfaceTexture surfaceTexture, Context context, SurfaceTexture surfaceTexture2) {
        if (surfaceTexture2 == surfaceTexture) {
            onInitialFrameReceived(context, surfaceTexture2);
        }
    }

    public void b(final Context context, z0 z0Var) {
        Size size = z0Var.f1700a;
        Log.d(TAG, String.format("Received surface request for resolution %dx%d", Integer.valueOf(size.getWidth()), Integer.valueOf(size.getHeight())));
        final SurfaceTexture createSurfaceTexture = createSurfaceTexture();
        createSurfaceTexture.setDefaultBufferSize(size.getWidth(), size.getHeight());
        createSurfaceTexture.setOnFrameAvailableListener(new SurfaceTexture.OnFrameAvailableListener() { // from class: c.e.a.f
            @Override // android.graphics.SurfaceTexture.OnFrameAvailableListener
            public final void onFrameAvailable(SurfaceTexture surfaceTexture) {
                CameraXPreviewHelper.this.a(createSurfaceTexture, context, surfaceTexture);
            }
        }, this.renderExecutor.getHandler());
        final Surface surface = new Surface(createSurfaceTexture);
        Log.d(TAG, "Providing surface");
        z0Var.a(surface, this.renderExecutor, new b.j.i.a() { // from class: c.e.a.c
            @Override // b.j.i.a
            public final void accept(Object obj) {
                SurfaceTexture surfaceTexture = createSurfaceTexture;
                Surface surface2 = surface;
                int i = CameraXPreviewHelper.f5657a;
                Log.d("CameraXPreviewHelper", "Surface request result: " + ((z0.f) obj));
                surfaceTexture.release();
                surface2.release();
            }
        });
    }

    public void c(ListenableFuture listenableFuture, Size size, CameraHelper.CameraFacing cameraFacing, final Context context, h hVar) {
        try {
            this.cameraProvider = (c) listenableFuture.get();
            w0.b bVar = new w0.b();
            bVar.f1687a.A(n0.f1576d, i0.c.OPTIONAL, size);
            w0 a2 = bVar.a();
            this.preview = a2;
            j0 j0Var = cameraFacing == CameraHelper.CameraFacing.FRONT ? j0.f1629a : j0.f1630b;
            a2.q(this.renderExecutor, new w0.d() { // from class: c.e.a.e
                @Override // b.d.b.w0.d
                public final void a(z0 z0Var) {
                    CameraXPreviewHelper.this.b(context, z0Var);
                }
            });
            this.cameraProvider.c();
            this.camera = this.cameraProvider.a(hVar, j0Var, this.preview);
        } catch (Exception e2) {
            if (e2 instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            Log.e(TAG, "Unable to get ProcessCameraProvider: ", e2);
        }
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public Size computeDisplaySizeFromViewSize(Size size) {
        return this.frameSize;
    }

    public e0 getCamera() {
        return this.camera;
    }

    public float getFocalLengthPixels() {
        return this.focalLengthPixels;
    }

    public Size getFrameSize() {
        return this.frameSize;
    }

    public long getTimeOffsetToMonoClockNanos() {
        return this.cameraTimestampSource == 1 ? getOffsetFromRealtimeTimestampSource() : getOffsetFromUnknownTimestampSource();
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public boolean isCameraRotated() {
        return this.frameRotation % BaseTransientBottomBar.ANIMATION_FADE_DURATION == 90;
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public void startCamera(Activity activity, CameraHelper.CameraFacing cameraFacing, SurfaceTexture surfaceTexture) {
        startCamera(activity, (h) activity, cameraFacing, TARGET_SIZE);
    }

    public void startCamera(Activity activity, CameraHelper.CameraFacing cameraFacing, SurfaceTexture surfaceTexture, Size size) {
        startCamera(activity, (h) activity, cameraFacing, size);
    }

    public void startCamera(final Context context, final h hVar, final CameraHelper.CameraFacing cameraFacing, Size size) {
        Executor b2 = a.b(context);
        final ListenableFuture<c> b3 = c.b(context);
        if (size == null) {
            size = TARGET_SIZE;
        }
        final Size size2 = new Size(size.getHeight(), size.getWidth());
        ((e) b3).f1543b.addListener(new Runnable() { // from class: c.e.a.d
            @Override // java.lang.Runnable
            public final void run() {
                CameraXPreviewHelper.this.c(b3, size2, cameraFacing, context, hVar);
            }
        }, b2);
    }
}