package org.opencv.android;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.List;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/* loaded from: classes2.dex */
public class JavaCameraView extends CameraBridgeViewBase implements Camera.PreviewCallback {
    private static final int MAGIC_TEXTURE_ID = 10;
    private static final String TAG = "JavaCameraView";
    private byte[] mBuffer;
    public Camera mCamera;
    public JavaCameraFrame[] mCameraFrame;
    private boolean mCameraFrameReady;
    private int mChainIdx;
    private Mat[] mFrameChain;
    private int mPreviewFormat;
    private boolean mStopThread;
    private SurfaceTexture mSurfaceTexture;
    private Thread mThread;

    /* loaded from: classes2.dex */
    public class CameraWorker implements Runnable {
        private CameraWorker() {
        }

        @Override // java.lang.Runnable
        public void run() {
            boolean z;
            do {
                synchronized (JavaCameraView.this) {
                    while (!JavaCameraView.this.mCameraFrameReady && !JavaCameraView.this.mStopThread) {
                        try {
                            JavaCameraView.this.wait();
                        } catch (InterruptedException e2) {
                            e2.printStackTrace();
                        }
                    }
                    z = false;
                    if (JavaCameraView.this.mCameraFrameReady) {
                        JavaCameraView javaCameraView = JavaCameraView.this;
                        javaCameraView.mChainIdx = 1 - javaCameraView.mChainIdx;
                        JavaCameraView.this.mCameraFrameReady = false;
                        z = true;
                    }
                }
                if (!JavaCameraView.this.mStopThread && z && !JavaCameraView.this.mFrameChain[1 - JavaCameraView.this.mChainIdx].empty()) {
                    JavaCameraView javaCameraView2 = JavaCameraView.this;
                    javaCameraView2.deliverAndDrawFrame(javaCameraView2.mCameraFrame[1 - javaCameraView2.mChainIdx]);
                }
            } while (!JavaCameraView.this.mStopThread);
            Log.d(JavaCameraView.TAG, "Finish processing thread");
        }
    }

    /* loaded from: classes2.dex */
    public class JavaCameraFrame implements CameraBridgeViewBase.CvCameraViewFrame {
        private int mHeight;
        private Mat mRgba = new Mat();
        private int mWidth;
        private Mat mYuvFrameData;

        public JavaCameraFrame(Mat mat, int i, int i2) {
            this.mWidth = i;
            this.mHeight = i2;
            this.mYuvFrameData = mat;
        }

        @Override // org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
        public Mat gray() {
            return this.mYuvFrameData.submat(0, this.mHeight, 0, this.mWidth);
        }

        public void release() {
            this.mRgba.release();
        }

        @Override // org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
        public Mat rgba() {
            if (JavaCameraView.this.mPreviewFormat != 17) {
                if (JavaCameraView.this.mPreviewFormat == 842094169) {
                    Imgproc.cvtColor(this.mYuvFrameData, this.mRgba, 100, 4);
                } else {
                    throw new IllegalArgumentException("Preview Format can be NV21 or YV12");
                }
            } else {
                Imgproc.cvtColor(this.mYuvFrameData, this.mRgba, 96, 4);
            }
            return this.mRgba;
        }
    }

    /* loaded from: classes2.dex */
    public static class JavaCameraSizeAccessor implements CameraBridgeViewBase.ListItemAccessor {
        @Override // org.opencv.android.CameraBridgeViewBase.ListItemAccessor
        public int getHeight(Object obj) {
            return ((Camera.Size) obj).height;
        }

        @Override // org.opencv.android.CameraBridgeViewBase.ListItemAccessor
        public int getWidth(Object obj) {
            return ((Camera.Size) obj).width;
        }
    }

    public JavaCameraView(Context context, int i) {
        super(context, i);
        this.mChainIdx = 0;
        this.mPreviewFormat = 17;
        this.mCameraFrameReady = false;
    }

    @Override // org.opencv.android.CameraBridgeViewBase
    public boolean connectCamera(int i, int i2) {
        Log.d(TAG, "Connecting to camera");
        if (initializeCamera(i, i2)) {
            this.mCameraFrameReady = false;
            Log.d(TAG, "Starting processing thread");
            this.mStopThread = false;
            Thread thread = new Thread(new CameraWorker());
            this.mThread = thread;
            thread.start();
            return true;
        }
        return false;
    }

    @Override // org.opencv.android.CameraBridgeViewBase
    public void disconnectCamera() {
        Log.d(TAG, "Disconnecting from camera");
        try {
            try {
                this.mStopThread = true;
                Log.d(TAG, "Notify thread");
                synchronized (this) {
                    notify();
                }
                Log.d(TAG, "Waiting for thread");
                Thread thread = this.mThread;
                if (thread != null) {
                    thread.join();
                }
            } catch (InterruptedException e2) {
                e2.printStackTrace();
            }
            this.mThread = null;
            releaseCamera();
            this.mCameraFrameReady = false;
        } catch (Throwable th) {
            this.mThread = null;
            throw th;
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:110:0x0140 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:43:0x00df A[Catch: all -> 0x02ec, TryCatch #4 {, blocks: (B:5:0x0009, B:7:0x0012, B:8:0x0019, B:12:0x003b, B:15:0x0041, B:17:0x0047, B:18:0x0066, B:24:0x0095, B:21:0x006f, B:51:0x013a, B:53:0x013e, B:55:0x0140, B:57:0x0151, B:59:0x0164, B:61:0x016c, B:63:0x0176, B:65:0x017e, B:67:0x0186, B:69:0x0190, B:71:0x019a, B:73:0x01a4, B:76:0x01af, B:78:0x01bb, B:80:0x0200, B:81:0x0203, B:83:0x0209, B:85:0x0211, B:86:0x0216, B:88:0x0239, B:90:0x0241, B:92:0x0255, B:94:0x0259, B:95:0x0260, B:91:0x0252, B:77:0x01b5, B:99:0x02ea, B:98:0x02e7, B:11:0x0021, B:27:0x009e, B:28:0x00ab, B:30:0x00b1, B:33:0x00b9, B:43:0x00df, B:45:0x00e9, B:46:0x00f1, B:47:0x0110, B:50:0x0118, B:35:0x00be, B:36:0x00cb, B:38:0x00d1, B:41:0x00da), top: B:112:0x0009, inners: #0, #1, #2, #3 }] */
    /* JADX WARN: Removed duplicated region for block: B:44:0x00e7  */
    /* JADX WARN: Removed duplicated region for block: B:53:0x013e A[Catch: all -> 0x02ec, DONT_GENERATE, TRY_LEAVE, TryCatch #4 {, blocks: (B:5:0x0009, B:7:0x0012, B:8:0x0019, B:12:0x003b, B:15:0x0041, B:17:0x0047, B:18:0x0066, B:24:0x0095, B:21:0x006f, B:51:0x013a, B:53:0x013e, B:55:0x0140, B:57:0x0151, B:59:0x0164, B:61:0x016c, B:63:0x0176, B:65:0x017e, B:67:0x0186, B:69:0x0190, B:71:0x019a, B:73:0x01a4, B:76:0x01af, B:78:0x01bb, B:80:0x0200, B:81:0x0203, B:83:0x0209, B:85:0x0211, B:86:0x0216, B:88:0x0239, B:90:0x0241, B:92:0x0255, B:94:0x0259, B:95:0x0260, B:91:0x0252, B:77:0x01b5, B:99:0x02ea, B:98:0x02e7, B:11:0x0021, B:27:0x009e, B:28:0x00ab, B:30:0x00b1, B:33:0x00b9, B:43:0x00df, B:45:0x00e9, B:46:0x00f1, B:47:0x0110, B:50:0x0118, B:35:0x00be, B:36:0x00cb, B:38:0x00d1, B:41:0x00da), top: B:112:0x0009, inners: #0, #1, #2, #3 }] */
    /* JADX WARN: Removed duplicated region for block: B:80:0x0200 A[Catch: Exception -> 0x02e6, all -> 0x02ec, TryCatch #3 {Exception -> 0x02e6, blocks: (B:55:0x0140, B:57:0x0151, B:59:0x0164, B:61:0x016c, B:63:0x0176, B:65:0x017e, B:67:0x0186, B:69:0x0190, B:71:0x019a, B:73:0x01a4, B:76:0x01af, B:78:0x01bb, B:80:0x0200, B:81:0x0203, B:83:0x0209, B:85:0x0211, B:86:0x0216, B:88:0x0239, B:90:0x0241, B:92:0x0255, B:94:0x0259, B:95:0x0260, B:91:0x0252, B:77:0x01b5), top: B:110:0x0140, outer: #4 }] */
    /* JADX WARN: Removed duplicated region for block: B:94:0x0259 A[Catch: Exception -> 0x02e6, all -> 0x02ec, TryCatch #3 {Exception -> 0x02e6, blocks: (B:55:0x0140, B:57:0x0151, B:59:0x0164, B:61:0x016c, B:63:0x0176, B:65:0x017e, B:67:0x0186, B:69:0x0190, B:71:0x019a, B:73:0x01a4, B:76:0x01af, B:78:0x01bb, B:80:0x0200, B:81:0x0203, B:83:0x0209, B:85:0x0211, B:86:0x0216, B:88:0x0239, B:90:0x0241, B:92:0x0255, B:94:0x0259, B:95:0x0260, B:91:0x0252, B:77:0x01b5), top: B:110:0x0140, outer: #4 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean initializeCamera(int i, int i2) {
        int i3;
        Camera camera;
        List<String> supportedFocusModes;
        FpsMeter fpsMeter;
        Log.d(TAG, "Initialize java camera");
        synchronized (this) {
            this.mCamera = null;
            int i4 = this.mCameraIndex;
            boolean z = false;
            if (i4 == -1) {
                Log.d(TAG, "Trying to open camera with old open()");
                try {
                    this.mCamera = Camera.open();
                } catch (Exception e2) {
                    Log.e(TAG, "Camera is not available (in use or does not exist): " + e2.getLocalizedMessage());
                }
                if (this.mCamera == null) {
                    boolean z2 = false;
                    for (int i5 = 0; i5 < Camera.getNumberOfCameras(); i5++) {
                        Log.d(TAG, "Trying to open camera with new open(" + Integer.valueOf(i5) + ")");
                        try {
                            this.mCamera = Camera.open(i5);
                            z2 = true;
                        } catch (RuntimeException e3) {
                            Log.e(TAG, "Camera #" + i5 + "failed to open: " + e3.getLocalizedMessage());
                        }
                        if (z2) {
                            break;
                        }
                    }
                }
                camera = this.mCamera;
                if (camera == null) {
                    return false;
                }
                try {
                    Camera.Parameters parameters = camera.getParameters();
                    Log.d(TAG, "getSupportedPreviewSizes()");
                    List<Camera.Size> supportedPreviewSizes = parameters.getSupportedPreviewSizes();
                    if (supportedPreviewSizes != null) {
                        Size calculateCameraFrameSize = calculateCameraFrameSize(supportedPreviewSizes, new JavaCameraSizeAccessor(), i, i2);
                        String str = Build.FINGERPRINT;
                        if (!str.startsWith("generic") && !str.startsWith("unknown")) {
                            String str2 = Build.MODEL;
                            if (!str2.contains("google_sdk") && !str2.contains("Emulator") && !str2.contains("Android SDK built for x86") && !Build.MANUFACTURER.contains("Genymotion") && ((!Build.BRAND.startsWith("generic") || !Build.DEVICE.startsWith("generic")) && !"google_sdk".equals(Build.PRODUCT))) {
                                parameters.setPreviewFormat(17);
                                this.mPreviewFormat = parameters.getPreviewFormat();
                                Log.d(TAG, "Set preview size to " + Integer.valueOf((int) calculateCameraFrameSize.width) + "x" + Integer.valueOf((int) calculateCameraFrameSize.height));
                                parameters.setPreviewSize((int) calculateCameraFrameSize.width, (int) calculateCameraFrameSize.height);
                                if (!Build.MODEL.equals("GT-I9100")) {
                                    parameters.setRecordingHint(true);
                                }
                                supportedFocusModes = parameters.getSupportedFocusModes();
                                if (supportedFocusModes != null && supportedFocusModes.contains("continuous-video")) {
                                    parameters.setFocusMode("continuous-video");
                                }
                                this.mCamera.setParameters(parameters);
                                Camera.Parameters parameters2 = this.mCamera.getParameters();
                                this.mFrameWidth = parameters2.getPreviewSize().width;
                                this.mFrameHeight = parameters2.getPreviewSize().height;
                                if (getLayoutParams().width != -1 && getLayoutParams().height == -1) {
                                    this.mScale = Math.min(i2 / this.mFrameHeight, i / this.mFrameWidth);
                                } else {
                                    this.mScale = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                                }
                                fpsMeter = this.mFpsMeter;
                                if (fpsMeter != null) {
                                    fpsMeter.setResolution(this.mFrameWidth, this.mFrameHeight);
                                }
                                byte[] bArr = new byte[((this.mFrameWidth * this.mFrameHeight) * ImageFormat.getBitsPerPixel(parameters2.getPreviewFormat())) / 8];
                                this.mBuffer = bArr;
                                this.mCamera.addCallbackBuffer(bArr);
                                this.mCamera.setPreviewCallbackWithBuffer(this);
                                Mat[] matArr = new Mat[2];
                                this.mFrameChain = matArr;
                                int i6 = this.mFrameHeight;
                                int i7 = this.mFrameWidth;
                                int i8 = CvType.CV_8UC1;
                                matArr[0] = new Mat(i6 + (i6 / 2), i7, i8);
                                Mat[] matArr2 = this.mFrameChain;
                                int i9 = this.mFrameHeight;
                                matArr2[1] = new Mat(i9 + (i9 / 2), this.mFrameWidth, i8);
                                AllocateCache();
                                JavaCameraFrame[] javaCameraFrameArr = new JavaCameraFrame[2];
                                this.mCameraFrame = javaCameraFrameArr;
                                javaCameraFrameArr[0] = new JavaCameraFrame(this.mFrameChain[0], this.mFrameWidth, this.mFrameHeight);
                                this.mCameraFrame[1] = new JavaCameraFrame(this.mFrameChain[1], this.mFrameWidth, this.mFrameHeight);
                                SurfaceTexture surfaceTexture = new SurfaceTexture(10);
                                this.mSurfaceTexture = surfaceTexture;
                                this.mCamera.setPreviewTexture(surfaceTexture);
                                Log.d(TAG, "startPreview");
                                this.mCamera.startPreview();
                                z = true;
                            }
                        }
                        parameters.setPreviewFormat(842094169);
                        this.mPreviewFormat = parameters.getPreviewFormat();
                        Log.d(TAG, "Set preview size to " + Integer.valueOf((int) calculateCameraFrameSize.width) + "x" + Integer.valueOf((int) calculateCameraFrameSize.height));
                        parameters.setPreviewSize((int) calculateCameraFrameSize.width, (int) calculateCameraFrameSize.height);
                        if (!Build.MODEL.equals("GT-I9100")) {
                        }
                        supportedFocusModes = parameters.getSupportedFocusModes();
                        if (supportedFocusModes != null) {
                            parameters.setFocusMode("continuous-video");
                        }
                        this.mCamera.setParameters(parameters);
                        Camera.Parameters parameters22 = this.mCamera.getParameters();
                        this.mFrameWidth = parameters22.getPreviewSize().width;
                        this.mFrameHeight = parameters22.getPreviewSize().height;
                        if (getLayoutParams().width != -1) {
                        }
                        this.mScale = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
                        fpsMeter = this.mFpsMeter;
                        if (fpsMeter != null) {
                        }
                        byte[] bArr2 = new byte[((this.mFrameWidth * this.mFrameHeight) * ImageFormat.getBitsPerPixel(parameters22.getPreviewFormat())) / 8];
                        this.mBuffer = bArr2;
                        this.mCamera.addCallbackBuffer(bArr2);
                        this.mCamera.setPreviewCallbackWithBuffer(this);
                        Mat[] matArr3 = new Mat[2];
                        this.mFrameChain = matArr3;
                        int i62 = this.mFrameHeight;
                        int i72 = this.mFrameWidth;
                        int i82 = CvType.CV_8UC1;
                        matArr3[0] = new Mat(i62 + (i62 / 2), i72, i82);
                        Mat[] matArr22 = this.mFrameChain;
                        int i92 = this.mFrameHeight;
                        matArr22[1] = new Mat(i92 + (i92 / 2), this.mFrameWidth, i82);
                        AllocateCache();
                        JavaCameraFrame[] javaCameraFrameArr2 = new JavaCameraFrame[2];
                        this.mCameraFrame = javaCameraFrameArr2;
                        javaCameraFrameArr2[0] = new JavaCameraFrame(this.mFrameChain[0], this.mFrameWidth, this.mFrameHeight);
                        this.mCameraFrame[1] = new JavaCameraFrame(this.mFrameChain[1], this.mFrameWidth, this.mFrameHeight);
                        SurfaceTexture surfaceTexture2 = new SurfaceTexture(10);
                        this.mSurfaceTexture = surfaceTexture2;
                        this.mCamera.setPreviewTexture(surfaceTexture2);
                        Log.d(TAG, "startPreview");
                        this.mCamera.startPreview();
                        z = true;
                    }
                } catch (Exception e4) {
                    e4.printStackTrace();
                }
                return z;
            } else if (i4 == 99) {
                Log.i(TAG, "Trying to open back camera");
                Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                i3 = 0;
                while (i3 < Camera.getNumberOfCameras()) {
                    Camera.getCameraInfo(i3, cameraInfo);
                    if (cameraInfo.facing == 0) {
                        i4 = i3;
                        break;
                    }
                    i3++;
                }
                if (i4 != 99) {
                    Log.e(TAG, "Back camera not found!");
                } else if (i4 == 98) {
                    Log.e(TAG, "Front camera not found!");
                } else {
                    Log.d(TAG, "Trying to open camera with new open(" + Integer.valueOf(i4) + ")");
                    try {
                        this.mCamera = Camera.open(i4);
                    } catch (RuntimeException e5) {
                        Log.e(TAG, "Camera #" + i4 + "failed to open: " + e5.getLocalizedMessage());
                    }
                }
                camera = this.mCamera;
                if (camera == null) {
                }
            } else {
                if (i4 == 98) {
                    Log.i(TAG, "Trying to open front camera");
                    Camera.CameraInfo cameraInfo2 = new Camera.CameraInfo();
                    i3 = 0;
                    while (i3 < Camera.getNumberOfCameras()) {
                        Camera.getCameraInfo(i3, cameraInfo2);
                        if (cameraInfo2.facing == 1) {
                            i4 = i3;
                            break;
                        }
                        i3++;
                    }
                }
                if (i4 != 99) {
                }
                camera = this.mCamera;
                if (camera == null) {
                }
            }
        }
    }

    @Override // android.hardware.Camera.PreviewCallback
    public void onPreviewFrame(byte[] bArr, Camera camera) {
        synchronized (this) {
            this.mFrameChain[this.mChainIdx].put(0, 0, bArr);
            this.mCameraFrameReady = true;
            notify();
        }
        Camera camera2 = this.mCamera;
        if (camera2 != null) {
            camera2.addCallbackBuffer(this.mBuffer);
        }
    }

    public void releaseCamera() {
        synchronized (this) {
            Camera camera = this.mCamera;
            if (camera != null) {
                camera.stopPreview();
                this.mCamera.setPreviewCallback(null);
                this.mCamera.release();
            }
            this.mCamera = null;
            Mat[] matArr = this.mFrameChain;
            if (matArr != null) {
                matArr[0].release();
                this.mFrameChain[1].release();
            }
            JavaCameraFrame[] javaCameraFrameArr = this.mCameraFrame;
            if (javaCameraFrameArr != null) {
                javaCameraFrameArr[0].release();
                this.mCameraFrame[1].release();
            }
        }
    }

    public JavaCameraView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mChainIdx = 0;
        this.mPreviewFormat = 17;
        this.mCameraFrameReady = false;
    }
}