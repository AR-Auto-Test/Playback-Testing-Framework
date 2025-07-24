package org.opencv.android;

import android.annotation.TargetApi;
import android.content.Context;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

@TargetApi(21)
/* loaded from: classes2.dex */
public class JavaCamera2View extends CameraBridgeViewBase {
    public static final /* synthetic */ boolean $assertionsDisabled = false;
    private static final String LOGTAG = "JavaCamera2View";
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private CameraDevice mCameraDevice;
    private String mCameraID;
    private CameraCaptureSession mCaptureSession;
    private ImageReader mImageReader;
    private int mPreviewFormat;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private Size mPreviewSize;
    private final CameraDevice.StateCallback mStateCallback;

    /* loaded from: classes2.dex */
    public class JavaCamera2Frame implements CameraBridgeViewBase.CvCameraViewFrame {
        public static final /* synthetic */ boolean $assertionsDisabled = false;
        private Image mImage;
        private Mat mRgba = new Mat();
        private Mat mGray = new Mat();

        public JavaCamera2Frame(Image image) {
            this.mImage = image;
        }

        @Override // org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
        public Mat gray() {
            Image.Plane[] planes = this.mImage.getPlanes();
            int width = this.mImage.getWidth();
            Mat mat = new Mat(this.mImage.getHeight(), width, CvType.CV_8UC1, planes[0].getBuffer(), planes[0].getRowStride());
            this.mGray = mat;
            return mat;
        }

        public void release() {
            this.mRgba.release();
            this.mGray.release();
        }

        @Override // org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
        public Mat rgba() {
            int i;
            Image.Plane[] planes = this.mImage.getPlanes();
            int width = this.mImage.getWidth();
            int height = this.mImage.getHeight();
            if (planes[1].getPixelStride() == 2) {
                ByteBuffer buffer = planes[0].getBuffer();
                int rowStride = planes[0].getRowStride();
                ByteBuffer buffer2 = planes[1].getBuffer();
                int rowStride2 = planes[1].getRowStride();
                ByteBuffer buffer3 = planes[2].getBuffer();
                int rowStride3 = planes[2].getRowStride();
                Mat mat = new Mat(height, width, CvType.CV_8UC1, buffer, rowStride);
                int i2 = height / 2;
                int i3 = width / 2;
                int i4 = CvType.CV_8UC2;
                Mat mat2 = new Mat(i2, i3, i4, buffer2, rowStride2);
                Mat mat3 = new Mat(i2, i3, i4, buffer3, rowStride3);
                if (mat3.dataAddr() - mat2.dataAddr() > 0) {
                    Imgproc.cvtColorTwoPlane(mat, mat2, this.mRgba, 94);
                } else {
                    Imgproc.cvtColorTwoPlane(mat, mat3, this.mRgba, 96);
                }
                return this.mRgba;
            }
            int i5 = height / 2;
            int i6 = height + i5;
            byte[] bArr = new byte[width * i6];
            ByteBuffer buffer4 = planes[0].getBuffer();
            ByteBuffer buffer5 = planes[1].getBuffer();
            ByteBuffer buffer6 = planes[2].getBuffer();
            int rowStride4 = planes[0].getRowStride();
            if (rowStride4 == width) {
                i = width * height;
                buffer4.get(bArr, 0, i);
            } else {
                int i7 = rowStride4 - width;
                int i8 = 0;
                for (int i9 = 0; i9 < height; i9++) {
                    buffer4.get(bArr, i8, width);
                    i8 += width;
                    if (i9 < height - 1) {
                        buffer4.position(buffer4.position() + i7);
                    }
                }
                i = i8;
            }
            int i10 = width / 2;
            int rowStride5 = planes[1].getRowStride() - i10;
            if (rowStride5 == 0) {
                int i11 = (height * width) / 4;
                buffer5.get(bArr, i, i11);
                buffer6.get(bArr, i + i11, i11);
            } else {
                for (int i12 = 0; i12 < i5; i12++) {
                    buffer5.get(bArr, i, i10);
                    i += i10;
                    if (i12 < i5 - 1) {
                        buffer5.position(buffer5.position() + rowStride5);
                    }
                }
                for (int i13 = 0; i13 < i5; i13++) {
                    buffer6.get(bArr, i, i10);
                    i += i10;
                    if (i13 < i5 - 1) {
                        buffer6.position(buffer6.position() + rowStride5);
                    }
                }
            }
            Mat mat4 = new Mat(i6, width, CvType.CV_8UC1);
            mat4.put(0, 0, bArr);
            Imgproc.cvtColor(mat4, this.mRgba, 104, 4);
            return this.mRgba;
        }
    }

    /* loaded from: classes2.dex */
    public static class JavaCameraSizeAccessor implements CameraBridgeViewBase.ListItemAccessor {
        @Override // org.opencv.android.CameraBridgeViewBase.ListItemAccessor
        public int getHeight(Object obj) {
            return ((Size) obj).getHeight();
        }

        @Override // org.opencv.android.CameraBridgeViewBase.ListItemAccessor
        public int getWidth(Object obj) {
            return ((Size) obj).getWidth();
        }
    }

    public JavaCamera2View(Context context, int i) {
        super(context, i);
        this.mPreviewFormat = 35;
        this.mPreviewSize = new Size(-1, -1);
        this.mStateCallback = new CameraDevice.StateCallback() { // from class: org.opencv.android.JavaCamera2View.1
            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onDisconnected(CameraDevice cameraDevice) {
                cameraDevice.close();
                JavaCamera2View.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onError(CameraDevice cameraDevice, int i2) {
                cameraDevice.close();
                JavaCamera2View.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onOpened(CameraDevice cameraDevice) {
                JavaCamera2View.this.mCameraDevice = cameraDevice;
                JavaCamera2View.this.createCameraPreviewSession();
            }
        };
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void createCameraPreviewSession() {
        int width = this.mPreviewSize.getWidth();
        int height = this.mPreviewSize.getHeight();
        Log.i(LOGTAG, "createCameraPreviewSession(" + width + "x" + height + ")");
        if (width < 0 || height < 0) {
            return;
        }
        try {
            if (this.mCameraDevice == null) {
                Log.e(LOGTAG, "createCameraPreviewSession: camera isn't opened");
            } else if (this.mCaptureSession != null) {
                Log.e(LOGTAG, "createCameraPreviewSession: mCaptureSession is already started");
            } else {
                ImageReader newInstance = ImageReader.newInstance(width, height, this.mPreviewFormat, 2);
                this.mImageReader = newInstance;
                newInstance.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() { // from class: org.opencv.android.JavaCamera2View.2
                    public static final /* synthetic */ boolean $assertionsDisabled = false;

                    @Override // android.media.ImageReader.OnImageAvailableListener
                    public void onImageAvailable(ImageReader imageReader) {
                        Image acquireLatestImage = imageReader.acquireLatestImage();
                        if (acquireLatestImage == null) {
                            return;
                        }
                        acquireLatestImage.getPlanes();
                        JavaCamera2Frame javaCamera2Frame = new JavaCamera2Frame(acquireLatestImage);
                        JavaCamera2View.this.deliverAndDrawFrame(javaCamera2Frame);
                        javaCamera2Frame.release();
                        acquireLatestImage.close();
                    }
                }, this.mBackgroundHandler);
                Surface surface = this.mImageReader.getSurface();
                CaptureRequest.Builder createCaptureRequest = this.mCameraDevice.createCaptureRequest(1);
                this.mPreviewRequestBuilder = createCaptureRequest;
                createCaptureRequest.addTarget(surface);
                this.mCameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback() { // from class: org.opencv.android.JavaCamera2View.3
                    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
                    public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                        Log.e(JavaCamera2View.LOGTAG, "createCameraPreviewSession failed");
                    }

                    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
                    public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                        Log.i(JavaCamera2View.LOGTAG, "createCaptureSession::onConfigured");
                        if (JavaCamera2View.this.mCameraDevice == null) {
                            return;
                        }
                        JavaCamera2View.this.mCaptureSession = cameraCaptureSession;
                        try {
                            JavaCamera2View.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, 4);
                            JavaCamera2View.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, 2);
                            JavaCamera2View.this.mCaptureSession.setRepeatingRequest(JavaCamera2View.this.mPreviewRequestBuilder.build(), null, JavaCamera2View.this.mBackgroundHandler);
                            Log.i(JavaCamera2View.LOGTAG, "CameraPreviewSession has been started");
                        } catch (Exception e2) {
                            Log.e(JavaCamera2View.LOGTAG, "createCaptureSession failed", e2);
                        }
                    }
                }, null);
            }
        } catch (CameraAccessException e2) {
            Log.e(LOGTAG, "createCameraPreviewSession", e2);
        }
    }

    private void startBackgroundThread() {
        Log.i(LOGTAG, "startBackgroundThread");
        stopBackgroundThread();
        HandlerThread handlerThread = new HandlerThread("OpenCVCameraBackground");
        this.mBackgroundThread = handlerThread;
        handlerThread.start();
        this.mBackgroundHandler = new Handler(this.mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        Log.i(LOGTAG, "stopBackgroundThread");
        HandlerThread handlerThread = this.mBackgroundThread;
        if (handlerThread == null) {
            return;
        }
        handlerThread.quitSafely();
        try {
            this.mBackgroundThread.join();
            this.mBackgroundThread = null;
            this.mBackgroundHandler = null;
        } catch (InterruptedException e2) {
            Log.e(LOGTAG, "stopBackgroundThread", e2);
        }
    }

    public boolean calcPreviewSize(int i, int i2) {
        Log.i(LOGTAG, "calcPreviewSize: " + i + "x" + i2);
        if (this.mCameraID == null) {
            Log.e(LOGTAG, "Camera isn't initialized!");
            return false;
        }
        try {
            org.opencv.core.Size calculateCameraFrameSize = calculateCameraFrameSize(Arrays.asList(((StreamConfigurationMap) ((CameraManager) getContext().getSystemService("camera")).getCameraCharacteristics(this.mCameraID).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)).getOutputSizes(ImageReader.class)), new JavaCameraSizeAccessor(), i, i2);
            Log.i(LOGTAG, "Selected preview size to " + Integer.valueOf((int) calculateCameraFrameSize.width) + "x" + Integer.valueOf((int) calculateCameraFrameSize.height));
            if (this.mPreviewSize.getWidth() == calculateCameraFrameSize.width && this.mPreviewSize.getHeight() == calculateCameraFrameSize.height) {
                return false;
            }
            this.mPreviewSize = new Size((int) calculateCameraFrameSize.width, (int) calculateCameraFrameSize.height);
            return true;
        } catch (CameraAccessException e2) {
            Log.e(LOGTAG, "calcPreviewSize - Camera Access Exception", e2);
            return false;
        } catch (IllegalArgumentException e3) {
            Log.e(LOGTAG, "calcPreviewSize - Illegal Argument Exception", e3);
            return false;
        } catch (SecurityException e4) {
            Log.e(LOGTAG, "calcPreviewSize - Security Exception", e4);
            return false;
        }
    }

    @Override // org.opencv.android.CameraBridgeViewBase
    public boolean connectCamera(int i, int i2) {
        Log.i(LOGTAG, "setCameraPreviewSize(" + i + "x" + i2 + ")");
        startBackgroundThread();
        initializeCamera();
        try {
            boolean calcPreviewSize = calcPreviewSize(i, i2);
            this.mFrameWidth = this.mPreviewSize.getWidth();
            this.mFrameHeight = this.mPreviewSize.getHeight();
            if (getLayoutParams().width == -1 && getLayoutParams().height == -1) {
                this.mScale = Math.min(i2 / this.mFrameHeight, i / this.mFrameWidth);
            } else {
                this.mScale = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }
            AllocateCache();
            if (calcPreviewSize) {
                if (this.mCaptureSession != null) {
                    Log.d(LOGTAG, "closing existing previewSession");
                    this.mCaptureSession.close();
                    this.mCaptureSession = null;
                }
                createCameraPreviewSession();
                return true;
            }
            return true;
        } catch (RuntimeException e2) {
            throw new RuntimeException("Interrupted while setCameraPreviewSize.", e2);
        }
    }

    @Override // org.opencv.android.CameraBridgeViewBase
    public void disconnectCamera() {
        Log.i(LOGTAG, "close camera");
        try {
            CameraDevice cameraDevice = this.mCameraDevice;
            this.mCameraDevice = null;
            CameraCaptureSession cameraCaptureSession = this.mCaptureSession;
            if (cameraCaptureSession != null) {
                cameraCaptureSession.close();
                this.mCaptureSession = null;
            }
            if (cameraDevice != null) {
                cameraDevice.close();
            }
            Log.i(LOGTAG, "camera closed!");
        } finally {
            stopBackgroundThread();
            ImageReader imageReader = this.mImageReader;
            if (imageReader != null) {
                imageReader.close();
                this.mImageReader = null;
            }
        }
    }

    public boolean initializeCamera() {
        Log.i(LOGTAG, "initializeCamera");
        CameraManager cameraManager = (CameraManager) getContext().getSystemService("camera");
        try {
            String[] cameraIdList = cameraManager.getCameraIdList();
            if (cameraIdList.length == 0) {
                Log.e(LOGTAG, "Error: camera isn't detected.");
                return false;
            }
            if (this.mCameraIndex == -1) {
                this.mCameraID = cameraIdList[0];
            } else {
                for (String str : cameraIdList) {
                    CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(str);
                    if ((this.mCameraIndex == 99 && ((Integer) cameraCharacteristics.get(CameraCharacteristics.LENS_FACING)).intValue() == 1) || (this.mCameraIndex == 98 && ((Integer) cameraCharacteristics.get(CameraCharacteristics.LENS_FACING)).intValue() == 0)) {
                        this.mCameraID = str;
                        break;
                    }
                }
            }
            if (this.mCameraID != null) {
                Log.i(LOGTAG, "Opening camera: " + this.mCameraID);
                cameraManager.openCamera(this.mCameraID, this.mStateCallback, this.mBackgroundHandler);
            } else {
                Log.i(LOGTAG, "Trying to open camera with the value (" + this.mCameraIndex + ")");
                int i = this.mCameraIndex;
                if (i < cameraIdList.length) {
                    String str2 = cameraIdList[i];
                    this.mCameraID = str2;
                    cameraManager.openCamera(str2, this.mStateCallback, this.mBackgroundHandler);
                } else {
                    throw new CameraAccessException(2);
                }
            }
            return true;
        } catch (CameraAccessException e2) {
            Log.e(LOGTAG, "OpenCamera - Camera Access Exception", e2);
            return false;
        } catch (IllegalArgumentException e3) {
            Log.e(LOGTAG, "OpenCamera - Illegal Argument Exception", e3);
            return false;
        } catch (SecurityException e4) {
            Log.e(LOGTAG, "OpenCamera - Security Exception", e4);
            return false;
        }
    }

    public JavaCamera2View(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        this.mPreviewFormat = 35;
        this.mPreviewSize = new Size(-1, -1);
        this.mStateCallback = new CameraDevice.StateCallback() { // from class: org.opencv.android.JavaCamera2View.1
            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onDisconnected(CameraDevice cameraDevice) {
                cameraDevice.close();
                JavaCamera2View.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onError(CameraDevice cameraDevice, int i2) {
                cameraDevice.close();
                JavaCamera2View.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onOpened(CameraDevice cameraDevice) {
                JavaCamera2View.this.mCameraDevice = cameraDevice;
                JavaCamera2View.this.createCameraPreviewSession();
            }
        };
    }
}