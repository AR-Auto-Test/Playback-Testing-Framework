package org.opencv.android;

import android.annotation.TargetApi;
import android.hardware.Camera;
import android.util.Log;
import java.io.IOException;
import java.util.List;

@TargetApi(15)
/* loaded from: classes2.dex */
public class CameraRenderer extends CameraGLRendererBase {
    public static final String LOGTAG = "CameraRenderer";
    private Camera mCamera;
    private boolean mPreviewStarted;

    public CameraRenderer(CameraGLSurfaceView cameraGLSurfaceView) {
        super(cameraGLSurfaceView);
        this.mPreviewStarted = false;
    }

    @Override // org.opencv.android.CameraGLRendererBase
    public synchronized void closeCamera() {
        Log.i(LOGTAG, "closeCamera");
        Camera camera = this.mCamera;
        if (camera != null) {
            camera.stopPreview();
            this.mPreviewStarted = false;
            this.mCamera.release();
            this.mCamera = null;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:41:0x00d8 A[Catch: all -> 0x017d, TryCatch #3 {, blocks: (B:3:0x0001, B:5:0x0010, B:6:0x0017, B:10:0x0039, B:13:0x003e, B:15:0x0044, B:16:0x005f, B:22:0x008e, B:19:0x0068, B:49:0x012f, B:51:0x0133, B:54:0x013c, B:56:0x0146, B:58:0x014e, B:59:0x0153, B:60:0x0158, B:63:0x0161, B:9:0x001f, B:23:0x0091, B:25:0x0099, B:26:0x00a5, B:28:0x00ab, B:31:0x00b3, B:41:0x00d8, B:43:0x00e2, B:44:0x00ea, B:45:0x0105, B:48:0x010d, B:33:0x00b8, B:34:0x00c4, B:36:0x00ca, B:39:0x00d3), top: B:75:0x0001, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:42:0x00e0  */
    /* JADX WARN: Removed duplicated region for block: B:51:0x0133 A[Catch: all -> 0x017d, TRY_LEAVE, TryCatch #3 {, blocks: (B:3:0x0001, B:5:0x0010, B:6:0x0017, B:10:0x0039, B:13:0x003e, B:15:0x0044, B:16:0x005f, B:22:0x008e, B:19:0x0068, B:49:0x012f, B:51:0x0133, B:54:0x013c, B:56:0x0146, B:58:0x014e, B:59:0x0153, B:60:0x0158, B:63:0x0161, B:9:0x001f, B:23:0x0091, B:25:0x0099, B:26:0x00a5, B:28:0x00ab, B:31:0x00b3, B:41:0x00d8, B:43:0x00e2, B:44:0x00ea, B:45:0x0105, B:48:0x010d, B:33:0x00b8, B:34:0x00c4, B:36:0x00ca, B:39:0x00d3), top: B:75:0x0001, inners: #0, #1, #2, #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:54:0x013c A[Catch: all -> 0x017d, TRY_ENTER, TryCatch #3 {, blocks: (B:3:0x0001, B:5:0x0010, B:6:0x0017, B:10:0x0039, B:13:0x003e, B:15:0x0044, B:16:0x005f, B:22:0x008e, B:19:0x0068, B:49:0x012f, B:51:0x0133, B:54:0x013c, B:56:0x0146, B:58:0x014e, B:59:0x0153, B:60:0x0158, B:63:0x0161, B:9:0x001f, B:23:0x0091, B:25:0x0099, B:26:0x00a5, B:28:0x00ab, B:31:0x00b3, B:41:0x00d8, B:43:0x00e2, B:44:0x00ea, B:45:0x0105, B:48:0x010d, B:33:0x00b8, B:34:0x00c4, B:36:0x00ca, B:39:0x00d3), top: B:75:0x0001, inners: #0, #1, #2, #4 }] */
    @Override // org.opencv.android.CameraGLRendererBase
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public synchronized void openCamera(int i) {
        Camera camera;
        Log.i(LOGTAG, "openCamera");
        closeCamera();
        int i2 = 0;
        if (i == -1) {
            Log.d(LOGTAG, "Trying to open camera with old open()");
            try {
                this.mCamera = Camera.open();
            } catch (Exception e2) {
                Log.e(LOGTAG, "Camera is not available (in use or does not exist): " + e2.getLocalizedMessage());
            }
            if (this.mCamera == null) {
                boolean z = false;
                while (i2 < Camera.getNumberOfCameras()) {
                    Log.d(LOGTAG, "Trying to open camera with new open(" + i2 + ")");
                    try {
                        this.mCamera = Camera.open(i2);
                        z = true;
                    } catch (RuntimeException e3) {
                        Log.e(LOGTAG, "Camera #" + i2 + "failed to open: " + e3.getLocalizedMessage());
                    }
                    if (z) {
                        break;
                    }
                    i2++;
                }
            }
            camera = this.mCamera;
            if (camera == null) {
                Log.e(LOGTAG, "Error: can't open camera");
                return;
            }
            Camera.Parameters parameters = camera.getParameters();
            List<String> supportedFocusModes = parameters.getSupportedFocusModes();
            if (supportedFocusModes != null && supportedFocusModes.contains("continuous-video")) {
                parameters.setFocusMode("continuous-video");
            }
            this.mCamera.setParameters(parameters);
            try {
                this.mCamera.setPreviewTexture(this.mSTexture);
            } catch (IOException e4) {
                Log.e(LOGTAG, "setPreviewTexture() failed: " + e4.getMessage());
            }
            return;
        }
        int i3 = this.mCameraIndex;
        if (i3 == 99) {
            Log.i(LOGTAG, "Trying to open BACK camera");
            Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
            while (i2 < Camera.getNumberOfCameras()) {
                Camera.getCameraInfo(i2, cameraInfo);
                if (cameraInfo.facing == 0) {
                    i3 = i2;
                    break;
                }
                i2++;
            }
            if (i3 != 99) {
                Log.e(LOGTAG, "Back camera not found!");
            } else if (i3 == 98) {
                Log.e(LOGTAG, "Front camera not found!");
            } else {
                Log.d(LOGTAG, "Trying to open camera with new open(" + i3 + ")");
                try {
                    this.mCamera = Camera.open(i3);
                } catch (RuntimeException e5) {
                    Log.e(LOGTAG, "Camera #" + i3 + "failed to open: " + e5.getLocalizedMessage());
                }
            }
            camera = this.mCamera;
            if (camera == null) {
            }
        } else {
            if (i3 == 98) {
                Log.i(LOGTAG, "Trying to open FRONT camera");
                Camera.CameraInfo cameraInfo2 = new Camera.CameraInfo();
                while (i2 < Camera.getNumberOfCameras()) {
                    Camera.getCameraInfo(i2, cameraInfo2);
                    if (cameraInfo2.facing == 1) {
                        i3 = i2;
                        break;
                    }
                    i2++;
                }
            }
            if (i3 != 99) {
            }
            camera = this.mCamera;
            if (camera == null) {
            }
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:36:0x00f4 A[Catch: all -> 0x0118, TryCatch #0 {, blocks: (B:3:0x0001, B:5:0x0023, B:8:0x002c, B:12:0x0033, B:16:0x003a, B:18:0x0048, B:19:0x0052, B:21:0x0058, B:26:0x0088, B:32:0x00a3, B:34:0x00f0, B:36:0x00f4, B:37:0x00fb, B:33:0x00c2, B:38:0x0102), top: B:44:0x0001 }] */
    @Override // org.opencv.android.CameraGLRendererBase
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public synchronized void setCameraPreviewSize(int i, int i2) {
        Log.i(LOGTAG, "setCameraPreviewSize: " + i + "x" + i2);
        Camera camera = this.mCamera;
        if (camera == null) {
            Log.e(LOGTAG, "Camera isn't initialized!");
            return;
        }
        int i3 = this.mMaxCameraWidth;
        if (i3 > 0 && i3 < i) {
            i = i3;
        }
        int i4 = this.mMaxCameraHeight;
        if (i4 > 0 && i4 < i2) {
            i2 = i4;
        }
        Camera.Parameters parameters = camera.getParameters();
        List<Camera.Size> supportedPreviewSizes = parameters.getSupportedPreviewSizes();
        if (supportedPreviewSizes.size() > 0) {
            float f2 = i / i2;
            int i5 = 0;
            int i6 = 0;
            for (Camera.Size size : supportedPreviewSizes) {
                int i7 = size.width;
                int i8 = size.height;
                Log.d(LOGTAG, "checking camera preview size: " + i7 + "x" + i8);
                if (i7 <= i && i8 <= i2 && i7 >= i5 && i8 >= i6 && Math.abs(f2 - (i7 / i8)) < 0.2d) {
                    i6 = i8;
                    i5 = i7;
                }
            }
            if (i5 > 0 && i6 > 0) {
                Log.i(LOGTAG, "Selected best size: " + i5 + " x " + i6);
                if (this.mPreviewStarted) {
                    this.mCamera.stopPreview();
                    this.mPreviewStarted = false;
                }
                this.mCameraWidth = i5;
                this.mCameraHeight = i6;
                parameters.setPreviewSize(i5, i6);
            }
            i5 = supportedPreviewSizes.get(0).width;
            i6 = supportedPreviewSizes.get(0).height;
            Log.e(LOGTAG, "Error: best size was not selected, using " + i5 + " x " + i6);
            if (this.mPreviewStarted) {
            }
            this.mCameraWidth = i5;
            this.mCameraHeight = i6;
            parameters.setPreviewSize(i5, i6);
        }
        parameters.set("orientation", "landscape");
        this.mCamera.setParameters(parameters);
        this.mCamera.startPreview();
        this.mPreviewStarted = true;
    }
}