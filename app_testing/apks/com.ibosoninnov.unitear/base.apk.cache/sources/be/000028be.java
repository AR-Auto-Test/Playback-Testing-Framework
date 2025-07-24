package com.google.mediapipe.components;

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
import androidx.camera.core.Camera;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.glutil.EglManager;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGLSurface;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/CameraXPreviewHelper.class */
public class CameraXPreviewHelper extends CameraHelper {
    private static final String TAG = "CameraXPreviewHelper";
    private static final Size TARGET_SIZE = new Size(1280, 720);
    private static final int CLOCK_OFFSET_CALIBRATION_ATTEMPTS = 3;
    private ProcessCameraProvider cameraProvider;
    private Preview preview;
    private Camera camera;
    private Size frameSize;
    private int frameRotation;
    private final SingleThreadHandlerExecutor renderExecutor = new SingleThreadHandlerExecutor("RenderThread", 0);
    @Nullable
    private CameraCharacteristics cameraCharacteristics = null;
    private float focalLengthPixels = Float.MIN_VALUE;
    private int cameraTimestampSource = 0;

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/CameraXPreviewHelper$SingleThreadHandlerExecutor.class */
    private static final class SingleThreadHandlerExecutor implements Executor {
        private final HandlerThread handlerThread;
        private final Handler handler;

        SingleThreadHandlerExecutor(String threadName, int priority) {
            this.handlerThread = new HandlerThread(threadName, priority);
            this.handlerThread.start();
            this.handler = new Handler(this.handlerThread.getLooper());
        }

        Handler getHandler() {
            return this.handler;
        }

        @Override // java.util.concurrent.Executor
        public void execute(Runnable command) {
            if (!this.handler.post(command)) {
                throw new RejectedExecutionException(this.handlerThread.getName() + " is shutting down.");
            }
        }

        boolean shutdown() {
            return this.handlerThread.quitSafely();
        }
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public void startCamera(Activity activity, CameraHelper.CameraFacing cameraFacing, SurfaceTexture unusedSurfaceTexture) {
        startCamera(activity, (LifecycleOwner) activity, cameraFacing, TARGET_SIZE);
    }

    public void startCamera(Activity activity, CameraHelper.CameraFacing cameraFacing, SurfaceTexture unusedSurfaceTexture, @Nullable Size targetSize) {
        startCamera(activity, (LifecycleOwner) activity, cameraFacing, targetSize);
    }

    public void startCamera(Context context, LifecycleOwner lifecycleOwner, CameraHelper.CameraFacing cameraFacing, @Nullable Size targetSize) {
        Executor mainThreadExecutor = ContextCompat.getMainExecutor(context);
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);
        Size targetSize2 = targetSize == null ? TARGET_SIZE : targetSize;
        Size rotatedSize = new Size(targetSize2.getHeight(), targetSize2.getWidth());
        cameraProviderFuture.addListener(()
        /*  JADX ERROR: Method code generation error
            jadx.core.utils.exceptions.CodegenException: Error generate insn: 0x003f: INVOKE  
              (r0v3 'cameraProviderFuture' com.google.common.util.concurrent.ListenableFuture<androidx.camera.lifecycle.ProcessCameraProvider> A[D('cameraProviderFuture' com.google.common.util.concurrent.ListenableFuture<androidx.camera.lifecycle.ProcessCameraProvider>)])
              (wrap: java.lang.Runnable : 0x0038: INVOKE_CUSTOM (r1v2 java.lang.Runnable A[REMOVE]) = 
              (r8v0 'this' com.google.mediapipe.components.CameraXPreviewHelper A[D('this' com.google.mediapipe.components.CameraXPreviewHelper), DONT_INLINE, IMMUTABLE_TYPE, THIS])
              (r0v3 'cameraProviderFuture' com.google.common.util.concurrent.ListenableFuture<androidx.camera.lifecycle.ProcessCameraProvider> A[D('cameraProviderFuture' com.google.common.util.concurrent.ListenableFuture<androidx.camera.lifecycle.ProcessCameraProvider>), DONT_INLINE])
              (r0v7 'rotatedSize' android.util.Size A[D('rotatedSize' android.util.Size), DONT_INLINE])
              (r11v0 'cameraFacing' com.google.mediapipe.components.CameraHelper$CameraFacing A[D('cameraFacing' com.google.mediapipe.components.CameraHelper$CameraFacing), DONT_INLINE])
              (r9v0 'context' android.content.Context A[D('context' android.content.Context), DONT_INLINE])
              (r10v0 'lifecycleOwner' androidx.lifecycle.LifecycleOwner A[D('lifecycleOwner' androidx.lifecycle.LifecycleOwner), DONT_INLINE])
            
             handle type: INVOKE_DIRECT
             lambda: java.lang.Runnable.run():void
             call insn: ?: INVOKE  
              (r1 I:com.google.mediapipe.components.CameraXPreviewHelper)
              (r2 I:com.google.common.util.concurrent.ListenableFuture)
              (r3 I:android.util.Size)
              (r4 I:com.google.mediapipe.components.CameraHelper$CameraFacing)
              (r5 I:android.content.Context)
              (r6 I:androidx.lifecycle.LifecycleOwner)
             type: DIRECT call: com.google.mediapipe.components.CameraXPreviewHelper.lambda$startCamera$3(com.google.common.util.concurrent.ListenableFuture, android.util.Size, com.google.mediapipe.components.CameraHelper$CameraFacing, android.content.Context, androidx.lifecycle.LifecycleOwner):void)
              (r0v1 'mainThreadExecutor' java.util.concurrent.Executor A[D('mainThreadExecutor' java.util.concurrent.Executor)])
             type: INTERFACE call: com.google.common.util.concurrent.ListenableFuture.addListener(java.lang.Runnable, java.util.concurrent.Executor):void in method: com.google.mediapipe.components.CameraXPreviewHelper.startCamera(android.content.Context, androidx.lifecycle.LifecycleOwner, com.google.mediapipe.components.CameraHelper$CameraFacing, android.util.Size):void, file: base.apk:classes.jar:com/google/mediapipe/components/CameraXPreviewHelper.class
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:289)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:252)
            	at jadx.core.codegen.RegionGen.makeSimpleBlock(RegionGen.java:91)
            	at jadx.core.dex.nodes.IBlock.generate(IBlock.java:15)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.dex.regions.Region.generate(Region.java:35)
            	at jadx.core.codegen.RegionGen.makeRegion(RegionGen.java:63)
            	at jadx.core.codegen.MethodGen.addRegionInsns(MethodGen.java:296)
            	at jadx.core.codegen.MethodGen.addInstructions(MethodGen.java:275)
            	at jadx.core.codegen.ClassGen.addMethodCode(ClassGen.java:377)
            	at jadx.core.codegen.ClassGen.addMethod(ClassGen.java:306)
            	at jadx.core.codegen.ClassGen.lambda$addInnerClsAndMethods$2(ClassGen.java:272)
            	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
            	at java.util.ArrayList.forEach(ArrayList.java:1259)
            	at java.util.stream.SortedOps$RefSortingSink.end(SortedOps.java:390)
            	at java.util.stream.Sink$ChainedReference.end(Sink.java:258)
            Caused by: java.lang.IndexOutOfBoundsException: Index: 5, Size: 5
            	at java.util.ArrayList.rangeCheck(ArrayList.java:659)
            	at java.util.ArrayList.get(ArrayList.java:435)
            	at jadx.core.codegen.InsnGen.makeInlinedLambdaMethod(InsnGen.java:998)
            	at jadx.core.codegen.InsnGen.makeInvokeLambda(InsnGen.java:903)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:794)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.addWrappedArg(InsnGen.java:143)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:119)
            	at jadx.core.codegen.InsnGen.addArg(InsnGen.java:106)
            	at jadx.core.codegen.InsnGen.generateMethodArguments(InsnGen.java:1075)
            	at jadx.core.codegen.InsnGen.makeInvoke(InsnGen.java:851)
            	at jadx.core.codegen.InsnGen.makeInsnBody(InsnGen.java:401)
            	at jadx.core.codegen.InsnGen.makeInsn(InsnGen.java:282)
            	... 15 more
            */
        /*
            this = this;
            r0 = r9
            java.util.concurrent.Executor r0 = androidx.core.content.ContextCompat.getMainExecutor(r0)
            r13 = r0
            r0 = r9
            com.google.common.util.concurrent.ListenableFuture r0 = androidx.camera.lifecycle.ProcessCameraProvider.getInstance(r0)
            r14 = r0
            r0 = r12
            if (r0 != 0) goto L17
            android.util.Size r0 = com.google.mediapipe.components.CameraXPreviewHelper.TARGET_SIZE
            goto L19
        L17:
            r0 = r12
        L19:
            r12 = r0
            android.util.Size r0 = new android.util.Size
            r1 = r0
            r2 = r12
            int r2 = r2.getHeight()
            r3 = r12
            int r3 = r3.getWidth()
            r1.<init>(r2, r3)
            r15 = r0
            r0 = r14
            r1 = r8
            r2 = r14
            r3 = r15
            r4 = r11
            r5 = r9
            r6 = r10
            void r1 = () -> { // java.lang.Runnable.run():void
                r1.lambda$startCamera$3(r2, r3, r4, r5, r6);
            }
            r2 = r13
            r0.addListener(r1, r2)
            return
        */
        throw new UnsupportedOperationException("Method not decompiled: com.google.mediapipe.components.CameraXPreviewHelper.startCamera(android.content.Context, androidx.lifecycle.LifecycleOwner, com.google.mediapipe.components.CameraHelper$CameraFacing, android.util.Size):void");
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public boolean isCameraRotated() {
        return this.frameRotation % BaseTransientBottomBar.ANIMATION_FADE_DURATION == 90;
    }

    @Override // com.google.mediapipe.components.CameraHelper
    public Size computeDisplaySizeFromViewSize(Size viewSize) {
        return this.frameSize;
    }

    @Nullable
    private Size getOptimalViewSize(Size targetSize) {
        if (this.cameraCharacteristics != null) {
            StreamConfigurationMap map = (StreamConfigurationMap) this.cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] outputSizes = map.getOutputSizes(SurfaceTexture.class);
            int selectedWidth = -1;
            int selectedHeight = -1;
            float selectedAspectRatioDifference = 1000.0f;
            float targetAspectRatio = targetSize.getWidth() / targetSize.getHeight();
            for (Size size : outputSizes) {
                float aspectRatio = size.getWidth() / size.getHeight();
                float aspectRatioDifference = Math.abs(aspectRatio - targetAspectRatio);
                if (aspectRatioDifference <= selectedAspectRatioDifference && ((selectedWidth == -1 && selectedHeight == -1) || (size.getWidth() <= selectedWidth && size.getWidth() >= this.frameSize.getWidth() && size.getHeight() <= selectedHeight && size.getHeight() >= this.frameSize.getHeight()))) {
                    selectedWidth = size.getWidth();
                    selectedHeight = size.getHeight();
                    selectedAspectRatioDifference = aspectRatioDifference;
                }
            }
            if (selectedWidth != -1 && selectedHeight != -1) {
                return new Size(selectedWidth, selectedHeight);
            }
            return null;
        }
        return null;
    }

    public long getTimeOffsetToMonoClockNanos() {
        if (this.cameraTimestampSource == 1) {
            return getOffsetFromRealtimeTimestampSource();
        }
        return getOffsetFromUnknownTimestampSource();
    }

    private static long getOffsetFromUnknownTimestampSource() {
        return 0L;
    }

    private static long getOffsetFromRealtimeTimestampSource() {
        long offset = Long.MAX_VALUE;
        long lowestGap = Long.MAX_VALUE;
        for (int i = 0; i < 3; i++) {
            long startMonoTs = System.nanoTime();
            long realTs = SystemClock.elapsedRealtimeNanos();
            long endMonoTs = System.nanoTime();
            long gapMonoTs = endMonoTs - startMonoTs;
            if (gapMonoTs < lowestGap) {
                lowestGap = gapMonoTs;
                offset = ((startMonoTs + endMonoTs) / 2) - realTs;
            }
        }
        return offset;
    }

    public float getFocalLengthPixels() {
        return this.focalLengthPixels;
    }

    public Size getFrameSize() {
        return this.frameSize;
    }

    private void onInitialFrameReceived(Context context, SurfaceTexture previewFrameTexture) {
        int i;
        previewFrameTexture.setOnFrameAvailableListener(null);
        previewFrameTexture.updateTexImage();
        previewFrameTexture.detachFromGLContext();
        if (!this.preview.getAttachedSurfaceResolution().equals(this.frameSize)) {
            this.frameSize = this.preview.getAttachedSurfaceResolution();
            this.frameRotation = this.camera.getCameraInfo().getSensorRotationDegrees();
            if (this.frameSize.getWidth() == 0 || this.frameSize.getHeight() == 0) {
                Log.d("CameraXPreviewHelper", "Invalid frameSize.");
                return;
            }
        }
        if (this.cameraFacing == CameraHelper.CameraFacing.FRONT) {
            i = 0;
        } else {
            i = 1;
        }
        Integer selectedLensFacing = Integer.valueOf(i);
        this.cameraCharacteristics = getCameraCharacteristics(context, selectedLensFacing);
        if (this.cameraCharacteristics != null) {
            this.cameraTimestampSource = ((Integer) this.cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)).intValue();
            this.focalLengthPixels = calculateFocalLengthInPixels();
        }
        CameraHelper.OnCameraStartedListener listener = this.onCameraStartedListener;
        if (listener != null) {
            ContextCompat.getMainExecutor(context).execute(() -> {
                listener.onCameraStarted(previewFrameTexture);
            });
        }
    }

    private float calculateFocalLengthInPixels() {
        float focalLengthMm = ((float[]) this.cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS))[0];
        float sensorWidthMm = ((SizeF) this.cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)).getWidth();
        return (this.frameSize.getWidth() * focalLengthMm) / sensorWidthMm;
    }

    private static SurfaceTexture createSurfaceTexture() {
        EglManager eglManager = new EglManager(null);
        EGLSurface tempEglSurface = eglManager.createOffscreenSurface(1, 1);
        eglManager.makeCurrent(tempEglSurface, tempEglSurface);
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        SurfaceTexture previewFrameTexture = new SurfaceTexture(textures[0]);
        return previewFrameTexture;
    }

    @Nullable
    private static CameraCharacteristics getCameraCharacteristics(Context context, Integer lensFacing) {
        CameraManager cameraManager = (CameraManager) context.getSystemService("camera");
        try {
            List<String> cameraList = Arrays.asList(cameraManager.getCameraIdList());
            for (String availableCameraId : cameraList) {
                CameraCharacteristics availableCameraCharacteristics = cameraManager.getCameraCharacteristics(availableCameraId);
                Integer availableLensFacing = (Integer) availableCameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
                if (availableLensFacing != null && availableLensFacing.equals(lensFacing)) {
                    return availableCameraCharacteristics;
                }
            }
            return null;
        } catch (CameraAccessException e2) {
            Log.e("CameraXPreviewHelper", "Accessing camera ID info got error: " + e2);
            return null;
        }
    }
}