package com.google.ar.sceneform;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.MeteringRectangle;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Handler;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import c.b.a.a.a;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/* loaded from: classes.dex */
public class CamPreviewHelper {
    public static int MAX_PREVIEW_HEIGHT = 0;
    public static int MAX_PREVIEW_WIDTH = 0;
    private static final SparseIntArray ORIENTATIONS;
    private static final int STATE_PICTURE_TAKEN = 4;
    private static final int STATE_PREVIEW = 0;
    private static final int STATE_WAITING_LOCK = 1;
    private static final int STATE_WAITING_NON_PRECAPTURE = 3;
    private static final int STATE_WAITING_PRECAPTURE = 2;
    private static final String TAG = "CamPreviewHelper";
    private static float fovX;
    private static float fovY;
    private static int mSensorOrientation;
    public Activity activity;
    private int camera2SupportLevel;
    private int cameraSceneMode;
    private File file;
    private Range<Integer> fpsRange;
    private boolean isOpticalStablisationSupported;
    private boolean isVideoStabilisationSupported;
    private boolean istouchFocuSupported;
    private CameraDevice mCameraDevice;
    private String mCameraId;
    private Semaphore mCameraOpenCloseLock;
    private CameraCaptureSession mCaptureSession;
    private boolean mFlashSupported;
    private CaptureRequest mPreviewRequest;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private Size mPreviewSize;
    private int mState;
    private final CameraDevice.StateCallback mStateCallback;
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener;
    private Rect sensorArraySize;
    private SimpleSceneView simpleSceneView;
    private int stabilisationMode;
    public Surface surface;
    public SurfaceTexture surfaceTexture;
    private TextureView textureView;

    /* loaded from: classes.dex */
    public static class CompareSizesByArea implements Comparator<Size> {
        /* JADX DEBUG: Method merged with bridge method */
        @Override // java.util.Comparator
        public int compare(Size size, Size size2) {
            return Long.signum((size.getWidth() * size.getHeight()) - (size2.getWidth() * size2.getHeight()));
        }
    }

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        ORIENTATIONS = sparseIntArray;
        sparseIntArray.append(0, 90);
        sparseIntArray.append(1, 0);
        sparseIntArray.append(2, 270);
        sparseIntArray.append(3, BaseTransientBottomBar.ANIMATION_FADE_DURATION);
        MAX_PREVIEW_WIDTH = 640;
        MAX_PREVIEW_HEIGHT = 480;
    }

    public CamPreviewHelper(Activity activity, int i) {
        this.mState = 0;
        this.mCameraOpenCloseLock = new Semaphore(1);
        this.istouchFocuSupported = false;
        this.isOpticalStablisationSupported = false;
        this.isVideoStabilisationSupported = false;
        this.mStateCallback = new CameraDevice.StateCallback() { // from class: com.google.ar.sceneform.CamPreviewHelper.1
            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onDisconnected(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onError(CameraDevice cameraDevice, int i2) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
                Activity activity2 = CamPreviewHelper.this.activity;
                if (activity2 != null) {
                    activity2.finish();
                }
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onOpened(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                CamPreviewHelper.this.mCameraDevice = cameraDevice;
                CamPreviewHelper.this.createCameraPreviewSession();
                Log.d(CamPreviewHelper.TAG, "createCameraPreviewSession");
            }
        };
        this.mSurfaceTextureListener = new TextureView.SurfaceTextureListener() { // from class: com.google.ar.sceneform.CamPreviewHelper.2
            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i2, int i3) {
                CamPreviewHelper.this.openCamera(i2, i3);
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return true;
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i2, int i3) {
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
            }
        };
        this.activity = activity;
        this.surfaceTexture = new SurfaceTexture(i);
        this.surface = new Surface(this.surfaceTexture);
    }

    private static Size chooseOptimalSize(Size[] sizeArr, int i, int i2, int i3, int i4, Size size) {
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList();
        int width = size.getWidth();
        int height = size.getHeight();
        for (Size size2 : sizeArr) {
            if (size2.getWidth() <= i3 && size2.getHeight() <= i4 && size2.getHeight() != size2.getWidth()) {
                StringBuilder x = a.x("Available Preview Size ");
                x.append(size2.getWidth());
                x.append(" x ");
                x.append(size2.getHeight());
                x.append("  Max =");
                x.append(i3);
                x.append("x");
                x.append(i4);
                x.append("  Aspect =");
                x.append(width);
                x.append("x");
                x.append(height);
                Log.d("CamPreiviewHelper", x.toString());
                if (size2.getWidth() >= i && size2.getHeight() >= i2) {
                    arrayList.add(size2);
                } else {
                    arrayList2.add(size2);
                }
            }
        }
        if (arrayList.size() > 0) {
            return (Size) Collections.min(arrayList, new CompareSizesByArea());
        }
        if (arrayList2.size() > 0) {
            return (Size) Collections.max(arrayList2, new CompareSizesByArea());
        }
        Log.e(TAG, "Couldn't find any suitable preview size");
        return sizeArr[0];
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void createCameraPreviewSession() {
        try {
            CaptureRequest.Builder createCaptureRequest = this.mCameraDevice.createCaptureRequest(1);
            this.mPreviewRequestBuilder = createCaptureRequest;
            createCaptureRequest.addTarget(this.surface);
            this.mCameraDevice.createCaptureSession(Arrays.asList(this.surface), new CameraCaptureSession.StateCallback() { // from class: com.google.ar.sceneform.CamPreviewHelper.3
                @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
                public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                    Log.d(CamPreviewHelper.TAG, "Configuration Failed");
                }

                @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
                public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                    if (CamPreviewHelper.this.mCameraDevice == null) {
                        return;
                    }
                    CamPreviewHelper.this.mCaptureSession = cameraCaptureSession;
                    try {
                        CamPreviewHelper.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, 4);
                        CamPreviewHelper.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 1);
                        if (CamPreviewHelper.this.fpsRange != null) {
                            CamPreviewHelper.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, CamPreviewHelper.this.fpsRange);
                        }
                        CamPreviewHelper.this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_CAPTURE_INTENT, 3);
                        CamPreviewHelper camPreviewHelper = CamPreviewHelper.this;
                        camPreviewHelper.mPreviewRequest = camPreviewHelper.mPreviewRequestBuilder.build();
                        CamPreviewHelper.this.mCaptureSession.setRepeatingRequest(CamPreviewHelper.this.mPreviewRequest, null, null);
                    } catch (CameraAccessException e2) {
                        e2.printStackTrace();
                    }
                }
            }, null);
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        }
    }

    public static Range<Integer> getOptimalFpsRange(CameraCharacteristics cameraCharacteristics) {
        Range[] rangeArr = (Range[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        Range<Integer> range = null;
        if (rangeArr == null) {
            Log.e(TAG, "Failed to get FPS ranges.");
            return null;
        } else if (rangeArr.length == 0) {
            Log.e(TAG, "Failed to get FPS ranges.");
            return null;
        } else {
            for (Range range2 : rangeArr) {
                int intValue = ((Integer) range2.getLower()).intValue();
                int intValue2 = ((Integer) range2.getUpper()).intValue();
                if (intValue2 > 1000) {
                    Log.w(TAG, "Device uses FPS range in a 1000 scale. Normalizing. MaxFPS=" + intValue2 + " MinFPS=" + intValue);
                    intValue /= 1000;
                    intValue2 /= 1000;
                }
                if (!(intValue == intValue2 || intValue < 0 || intValue2 > 30)) {
                    if (range == null || (intValue2 >= range.getUpper().intValue() && intValue2 - intValue >= range.getUpper().intValue() - range.getLower().intValue())) {
                        range = Range.create(Integer.valueOf(intValue), Integer.valueOf(intValue2));
                    }
                }
            }
            return range;
        }
    }

    /* JADX WARN: Can't wrap try/catch for region: R(29:14|15|16|(25:(2:19|(2:21|(24:23|24|25|(1:27)(1:91)|28|(1:30)(1:90)|31|(1:33)(1:89)|34|(1:36)(1:88)|37|(2:83|84)|39|(2:41|(3:43|(2:45|46)(1:48)|47))|49|50|51|(1:53)|54|(3:56|(3:58|(2:60|61)(1:63)|62)|64)|65|(3:67|(3:69|(2:71|72)(1:74)|73)|75)|76|77)))|(25:93|(1:95)|24|25|(0)(0)|28|(0)(0)|31|(0)(0)|34|(0)(0)|37|(0)|39|(0)|49|50|51|(0)|54|(0)|65|(0)|76|77)|96|25|(0)(0)|28|(0)(0)|31|(0)(0)|34|(0)(0)|37|(0)|39|(0)|49|50|51|(0)|54|(0)|65|(0)|76|77)|97|(25:99|(1:101)|24|25|(0)(0)|28|(0)(0)|31|(0)(0)|34|(0)(0)|37|(0)|39|(0)|49|50|51|(0)|54|(0)|65|(0)|76|77)|96|25|(0)(0)|28|(0)(0)|31|(0)(0)|34|(0)(0)|37|(0)|39|(0)|49|50|51|(0)|54|(0)|65|(0)|76|77) */
    /* JADX WARN: Code restructure failed: missing block: B:65:0x01af, code lost:
        r0 = move-exception;
     */
    /* JADX WARN: Code restructure failed: missing block: B:66:0x01b0, code lost:
        android.util.Log.e("JavaCamera2View", r0.toString());
     */
    /* JADX WARN: Removed duplicated region for block: B:35:0x00ba  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x00c4  */
    /* JADX WARN: Removed duplicated region for block: B:39:0x00cc  */
    /* JADX WARN: Removed duplicated region for block: B:40:0x00ce  */
    /* JADX WARN: Removed duplicated region for block: B:43:0x00d3  */
    /* JADX WARN: Removed duplicated region for block: B:44:0x00d5  */
    /* JADX WARN: Removed duplicated region for block: B:47:0x0122  */
    /* JADX WARN: Removed duplicated region for block: B:48:0x0124 A[Catch: NullPointerException -> 0x021d, CameraAccessException -> 0x0222, TryCatch #0 {NullPointerException -> 0x021d, blocks: (B:3:0x000e, B:5:0x0017, B:7:0x0027, B:12:0x0038, B:10:0x002e, B:13:0x003b, B:21:0x007a, B:33:0x00a2, B:37:0x00c8, B:41:0x00cf, B:45:0x00d6, B:49:0x0128, B:55:0x0172, B:57:0x017c, B:59:0x0180, B:61:0x0185, B:66:0x01b0, B:67:0x01b7, B:69:0x01eb, B:70:0x01ed, B:72:0x01f7, B:74:0x01fb, B:76:0x01ff, B:77:0x0201, B:78:0x0204, B:80:0x020e, B:82:0x0211, B:84:0x0215, B:85:0x0217, B:86:0x021a, B:54:0x016b, B:48:0x0124), top: B:93:0x000e }] */
    /* JADX WARN: Removed duplicated region for block: B:57:0x017c A[Catch: NullPointerException -> 0x021d, CameraAccessException -> 0x0222, TryCatch #0 {NullPointerException -> 0x021d, blocks: (B:3:0x000e, B:5:0x0017, B:7:0x0027, B:12:0x0038, B:10:0x002e, B:13:0x003b, B:21:0x007a, B:33:0x00a2, B:37:0x00c8, B:41:0x00cf, B:45:0x00d6, B:49:0x0128, B:55:0x0172, B:57:0x017c, B:59:0x0180, B:61:0x0185, B:66:0x01b0, B:67:0x01b7, B:69:0x01eb, B:70:0x01ed, B:72:0x01f7, B:74:0x01fb, B:76:0x01ff, B:77:0x0201, B:78:0x0204, B:80:0x020e, B:82:0x0211, B:84:0x0215, B:85:0x0217, B:86:0x021a, B:54:0x016b, B:48:0x0124), top: B:93:0x000e }] */
    /* JADX WARN: Removed duplicated region for block: B:69:0x01eb A[Catch: NullPointerException -> 0x021d, CameraAccessException -> 0x0222, TryCatch #0 {NullPointerException -> 0x021d, blocks: (B:3:0x000e, B:5:0x0017, B:7:0x0027, B:12:0x0038, B:10:0x002e, B:13:0x003b, B:21:0x007a, B:33:0x00a2, B:37:0x00c8, B:41:0x00cf, B:45:0x00d6, B:49:0x0128, B:55:0x0172, B:57:0x017c, B:59:0x0180, B:61:0x0185, B:66:0x01b0, B:67:0x01b7, B:69:0x01eb, B:70:0x01ed, B:72:0x01f7, B:74:0x01fb, B:76:0x01ff, B:77:0x0201, B:78:0x0204, B:80:0x020e, B:82:0x0211, B:84:0x0215, B:85:0x0217, B:86:0x021a, B:54:0x016b, B:48:0x0124), top: B:93:0x000e }] */
    /* JADX WARN: Removed duplicated region for block: B:72:0x01f7 A[Catch: NullPointerException -> 0x021d, CameraAccessException -> 0x0222, TryCatch #0 {NullPointerException -> 0x021d, blocks: (B:3:0x000e, B:5:0x0017, B:7:0x0027, B:12:0x0038, B:10:0x002e, B:13:0x003b, B:21:0x007a, B:33:0x00a2, B:37:0x00c8, B:41:0x00cf, B:45:0x00d6, B:49:0x0128, B:55:0x0172, B:57:0x017c, B:59:0x0180, B:61:0x0185, B:66:0x01b0, B:67:0x01b7, B:69:0x01eb, B:70:0x01ed, B:72:0x01f7, B:74:0x01fb, B:76:0x01ff, B:77:0x0201, B:78:0x0204, B:80:0x020e, B:82:0x0211, B:84:0x0215, B:85:0x0217, B:86:0x021a, B:54:0x016b, B:48:0x0124), top: B:93:0x000e }] */
    /* JADX WARN: Removed duplicated region for block: B:80:0x020e A[Catch: NullPointerException -> 0x021d, CameraAccessException -> 0x0222, TryCatch #0 {NullPointerException -> 0x021d, blocks: (B:3:0x000e, B:5:0x0017, B:7:0x0027, B:12:0x0038, B:10:0x002e, B:13:0x003b, B:21:0x007a, B:33:0x00a2, B:37:0x00c8, B:41:0x00cf, B:45:0x00d6, B:49:0x0128, B:55:0x0172, B:57:0x017c, B:59:0x0180, B:61:0x0185, B:66:0x01b0, B:67:0x01b7, B:69:0x01eb, B:70:0x01ed, B:72:0x01f7, B:74:0x01fb, B:76:0x01ff, B:77:0x0201, B:78:0x0204, B:80:0x020e, B:82:0x0211, B:84:0x0215, B:85:0x0217, B:86:0x021a, B:54:0x016b, B:48:0x0124), top: B:93:0x000e }] */
    /* JADX WARN: Removed duplicated region for block: B:97:0x013c A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void setUpCameraOutputs(int i, int i2) {
        String[] cameraIdList;
        StreamConfigurationMap streamConfigurationMap;
        boolean z;
        int i3;
        int i4;
        float[] fArr;
        int[] iArr;
        int[] iArr2;
        int[] iArr3;
        CameraManager cameraManager = (CameraManager) this.activity.getSystemService("camera");
        try {
            try {
                for (String str : cameraManager.getCameraIdList()) {
                    CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(str);
                    Integer num = (Integer) cameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
                    if ((num == null || num.intValue() != 0) && (streamConfigurationMap = (StreamConfigurationMap) cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)) != null) {
                        Size size = (Size) Collections.max(Arrays.asList(streamConfigurationMap.getOutputSizes(256)), new CompareSizesByArea());
                        int rotation = this.activity.getWindowManager().getDefaultDisplay().getRotation();
                        int intValue = ((Integer) cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)).intValue();
                        mSensorOrientation = intValue;
                        if (rotation != 0) {
                            if (rotation != 1) {
                                if (rotation != 2) {
                                    if (rotation != 3) {
                                        Log.e(TAG, "Display rotation is invalid: " + rotation);
                                        z = false;
                                        Point point = new Point();
                                        this.activity.getWindowManager().getDefaultDisplay().getSize(point);
                                        int i5 = point.x;
                                        int i6 = point.y;
                                        if (z) {
                                            i4 = i;
                                            i3 = i2;
                                            i5 = i6;
                                            i6 = i5;
                                        } else {
                                            i3 = i;
                                            i4 = i2;
                                        }
                                        int i7 = MAX_PREVIEW_WIDTH;
                                        int i8 = i5 > i7 ? i7 : i5;
                                        int i9 = MAX_PREVIEW_HEIGHT;
                                        this.mPreviewSize = chooseOptimalSize(streamConfigurationMap.getOutputSizes(SurfaceTexture.class), i3, i4, i8, i6 > i9 ? i9 : i6, size);
                                        Log.d("CamPreiviewHelper", "Preview Size " + this.mPreviewSize.getWidth() + " x " + this.mPreviewSize.getHeight());
                                        int i10 = this.activity.getResources().getConfiguration().orientation;
                                        Boolean bool = (Boolean) cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                                        this.mFlashSupported = bool == null ? false : bool.booleanValue();
                                        SizeF sizeF = (SizeF) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                                        if (((float[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) != null) {
                                            try {
                                                fovX = ((((float) Math.atan(sizeF.getWidth() / (fArr[0] * 2.0f))) * 2.0f) * 180.0f) / 3.14f;
                                                fovY = ((((float) Math.atan(sizeF.getHeight() / (fArr[0] * 2.0f))) * 2.0f) * 180.0f) / 3.14f;
                                            } catch (NullPointerException e2) {
                                                Log.e(TAG, e2.toString());
                                            }
                                        }
                                        iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
                                        if (iArr != null) {
                                            for (int i11 : iArr) {
                                                if (i11 == 5) {
                                                    this.cameraSceneMode = i11;
                                                    Log.d(TAG, "SceneModeNight " + this.cameraSceneMode);
                                                }
                                            }
                                        }
                                        this.camera2SupportLevel = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
                                        Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.camera2SupportLevel);
                                        this.fpsRange = getOptimalFpsRange(cameraCharacteristics);
                                        this.sensorArraySize = (Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                                        if (((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue() >= 1) {
                                            this.istouchFocuSupported = true;
                                        }
                                        iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
                                        if (iArr2 != null) {
                                            for (int i12 : iArr2) {
                                                if (i12 == 1) {
                                                    this.isVideoStabilisationSupported = true;
                                                }
                                            }
                                        }
                                        iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                                        if (iArr3 != null) {
                                            for (int i13 : iArr3) {
                                                if (i13 == 1) {
                                                    this.isOpticalStablisationSupported = true;
                                                }
                                            }
                                        }
                                        this.mCameraId = str;
                                        return;
                                    }
                                }
                            }
                            if (intValue != 0) {
                                if (intValue == 180) {
                                }
                                z = false;
                                Point point2 = new Point();
                                this.activity.getWindowManager().getDefaultDisplay().getSize(point2);
                                int i52 = point2.x;
                                int i62 = point2.y;
                                if (z) {
                                }
                                int i72 = MAX_PREVIEW_WIDTH;
                                if (i52 > i72) {
                                }
                                int i92 = MAX_PREVIEW_HEIGHT;
                                this.mPreviewSize = chooseOptimalSize(streamConfigurationMap.getOutputSizes(SurfaceTexture.class), i3, i4, i8, i62 > i92 ? i92 : i62, size);
                                Log.d("CamPreiviewHelper", "Preview Size " + this.mPreviewSize.getWidth() + " x " + this.mPreviewSize.getHeight());
                                int i102 = this.activity.getResources().getConfiguration().orientation;
                                Boolean bool2 = (Boolean) cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                                this.mFlashSupported = bool2 == null ? false : bool2.booleanValue();
                                SizeF sizeF2 = (SizeF) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                                if (((float[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) != null) {
                                }
                                iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
                                if (iArr != null) {
                                }
                                this.camera2SupportLevel = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
                                Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.camera2SupportLevel);
                                this.fpsRange = getOptimalFpsRange(cameraCharacteristics);
                                this.sensorArraySize = (Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                                if (((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue() >= 1) {
                                }
                                iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
                                if (iArr2 != null) {
                                }
                                iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                                if (iArr3 != null) {
                                }
                                this.mCameraId = str;
                                return;
                            }
                            z = true;
                            Point point22 = new Point();
                            this.activity.getWindowManager().getDefaultDisplay().getSize(point22);
                            int i522 = point22.x;
                            int i622 = point22.y;
                            if (z) {
                            }
                            int i722 = MAX_PREVIEW_WIDTH;
                            if (i522 > i722) {
                            }
                            int i922 = MAX_PREVIEW_HEIGHT;
                            this.mPreviewSize = chooseOptimalSize(streamConfigurationMap.getOutputSizes(SurfaceTexture.class), i3, i4, i8, i622 > i922 ? i922 : i622, size);
                            Log.d("CamPreiviewHelper", "Preview Size " + this.mPreviewSize.getWidth() + " x " + this.mPreviewSize.getHeight());
                            int i1022 = this.activity.getResources().getConfiguration().orientation;
                            Boolean bool22 = (Boolean) cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                            this.mFlashSupported = bool22 == null ? false : bool22.booleanValue();
                            SizeF sizeF22 = (SizeF) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                            if (((float[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) != null) {
                            }
                            iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
                            if (iArr != null) {
                            }
                            this.camera2SupportLevel = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
                            Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.camera2SupportLevel);
                            this.fpsRange = getOptimalFpsRange(cameraCharacteristics);
                            this.sensorArraySize = (Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                            if (((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue() >= 1) {
                            }
                            iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
                            if (iArr2 != null) {
                            }
                            iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                            if (iArr3 != null) {
                            }
                            this.mCameraId = str;
                            return;
                        }
                        if (intValue != 90) {
                            if (intValue == 270) {
                            }
                            z = false;
                            Point point222 = new Point();
                            this.activity.getWindowManager().getDefaultDisplay().getSize(point222);
                            int i5222 = point222.x;
                            int i6222 = point222.y;
                            if (z) {
                            }
                            int i7222 = MAX_PREVIEW_WIDTH;
                            if (i5222 > i7222) {
                            }
                            int i9222 = MAX_PREVIEW_HEIGHT;
                            this.mPreviewSize = chooseOptimalSize(streamConfigurationMap.getOutputSizes(SurfaceTexture.class), i3, i4, i8, i6222 > i9222 ? i9222 : i6222, size);
                            Log.d("CamPreiviewHelper", "Preview Size " + this.mPreviewSize.getWidth() + " x " + this.mPreviewSize.getHeight());
                            int i10222 = this.activity.getResources().getConfiguration().orientation;
                            Boolean bool222 = (Boolean) cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                            this.mFlashSupported = bool222 == null ? false : bool222.booleanValue();
                            SizeF sizeF222 = (SizeF) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                            if (((float[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) != null) {
                            }
                            iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
                            if (iArr != null) {
                            }
                            this.camera2SupportLevel = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
                            Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.camera2SupportLevel);
                            this.fpsRange = getOptimalFpsRange(cameraCharacteristics);
                            this.sensorArraySize = (Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                            if (((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue() >= 1) {
                            }
                            iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
                            if (iArr2 != null) {
                            }
                            iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                            if (iArr3 != null) {
                            }
                            this.mCameraId = str;
                            return;
                        }
                        z = true;
                        Point point2222 = new Point();
                        this.activity.getWindowManager().getDefaultDisplay().getSize(point2222);
                        int i52222 = point2222.x;
                        int i62222 = point2222.y;
                        if (z) {
                        }
                        int i72222 = MAX_PREVIEW_WIDTH;
                        if (i52222 > i72222) {
                        }
                        int i92222 = MAX_PREVIEW_HEIGHT;
                        this.mPreviewSize = chooseOptimalSize(streamConfigurationMap.getOutputSizes(SurfaceTexture.class), i3, i4, i8, i62222 > i92222 ? i92222 : i62222, size);
                        Log.d("CamPreiviewHelper", "Preview Size " + this.mPreviewSize.getWidth() + " x " + this.mPreviewSize.getHeight());
                        int i102222 = this.activity.getResources().getConfiguration().orientation;
                        Boolean bool2222 = (Boolean) cameraCharacteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
                        this.mFlashSupported = bool2222 == null ? false : bool2222.booleanValue();
                        SizeF sizeF2222 = (SizeF) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                        if (((float[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) != null) {
                        }
                        iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
                        if (iArr != null) {
                        }
                        this.camera2SupportLevel = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
                        Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.camera2SupportLevel);
                        this.fpsRange = getOptimalFpsRange(cameraCharacteristics);
                        this.sensorArraySize = (Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                        if (((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue() >= 1) {
                        }
                        iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
                        if (iArr2 != null) {
                        }
                        iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                        if (iArr3 != null) {
                        }
                        this.mCameraId = str;
                        return;
                    }
                }
            } catch (CameraAccessException e3) {
                e3.printStackTrace();
            }
        } catch (NullPointerException e4) {
            e4.printStackTrace();
        }
    }

    public void closeCamera() {
        try {
            try {
                this.mCameraOpenCloseLock.acquire();
                CameraCaptureSession cameraCaptureSession = this.mCaptureSession;
                if (cameraCaptureSession != null) {
                    cameraCaptureSession.close();
                    this.mCaptureSession = null;
                }
                CameraDevice cameraDevice = this.mCameraDevice;
                if (cameraDevice != null) {
                    cameraDevice.close();
                    this.mCameraDevice = null;
                }
            } catch (InterruptedException e2) {
                throw new RuntimeException("Interrupted while trying to lock camera closing.", e2);
            }
        } finally {
            this.mCameraOpenCloseLock.release();
        }
    }

    public void enableAutoFocus() {
        try {
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, 4);
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 1);
            this.mCaptureSession.setRepeatingRequest(this.mPreviewRequestBuilder.build(), null, null);
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        }
    }

    public float getFovX() {
        return fovX;
    }

    public float getFovY() {
        return fovY;
    }

    public int getSensorOrientation() {
        return mSensorOrientation;
    }

    @SuppressLint({"MissingPermission"})
    public void openCamera(int i, int i2) {
        if (b.j.c.a.a(this.activity, "android.permission.CAMERA") != 0) {
            return;
        }
        setUpCameraOutputs(i, i2);
        CameraManager cameraManager = (CameraManager) this.activity.getSystemService("camera");
        try {
            if (this.mCameraOpenCloseLock.tryAcquire(2500L, TimeUnit.MILLISECONDS)) {
                cameraManager.openCamera(this.mCameraId, this.mStateCallback, (Handler) null);
                return;
            }
            throw new RuntimeException("Time out waiting to lock camera opening.");
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        } catch (InterruptedException e3) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e3);
        }
    }

    public void toggleFlash(boolean z) {
        StringBuilder x = a.x("FlashSupported ");
        x.append(this.mFlashSupported);
        Log.d(TAG, x.toString());
        try {
            if (this.mCameraId.equals(CrashlyticsReportDataCapture.SIGNAL_DEFAULT) && this.mFlashSupported) {
                if (!z) {
                    Log.d(TAG, "Flash Turned OFF");
                    this.mPreviewRequestBuilder.set(CaptureRequest.FLASH_MODE, 0);
                    this.mCaptureSession.setRepeatingRequest(this.mPreviewRequestBuilder.build(), null, null);
                } else {
                    Log.d(TAG, "Flash Turned ON");
                    this.mPreviewRequestBuilder.set(CaptureRequest.FLASH_MODE, 2);
                    this.mCaptureSession.setRepeatingRequest(this.mPreviewRequestBuilder.build(), null, null);
                }
            }
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        }
    }

    public void touchToFocus(float f2, float f3, float f4, float f5) {
        int height;
        int width;
        try {
            if (this.camera2SupportLevel == 2) {
                Log.d(TAG, "Touch focus may not be supported");
                return;
            }
            int i = mSensorOrientation;
            if (i == 0) {
                width = (int) ((f2 / f4) * this.sensorArraySize.height());
                height = (int) ((f3 / f5) * this.sensorArraySize.width());
            } else if (i == 270) {
                int width2 = (int) (1.0f - ((f3 / f5) * this.sensorArraySize.width()));
                height = (int) (1.0f - ((f2 / f4) * this.sensorArraySize.height()));
                width = width2;
            } else {
                height = (int) ((f2 / f4) * this.sensorArraySize.height());
                width = (int) ((f3 / f5) * this.sensorArraySize.width());
            }
            MeteringRectangle meteringRectangle = new MeteringRectangle(Math.max(width - 50, 0), Math.max(height - 50, 0), 100, 100, 999);
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER, 2);
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, 0);
            this.mCaptureSession.capture(this.mPreviewRequestBuilder.build(), null, null);
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_MODE, 1);
            if (this.istouchFocuSupported) {
                Log.d(TAG, "Touch focus supported");
                this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AE_REGIONS, new MeteringRectangle[]{meteringRectangle});
                this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_REGIONS, new MeteringRectangle[]{meteringRectangle});
            }
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, 1);
            this.mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER, 1);
            this.mCaptureSession.setRepeatingRequest(this.mPreviewRequestBuilder.build(), null, null);
            Log.d(TAG, "AutoFocus pressed " + this.sensorArraySize + " view = " + (f2 / f4) + ", " + (f3 / f5) + " " + meteringRectangle.toString());
        } catch (CameraAccessException e2) {
            e2.printStackTrace();
        }
    }

    public CamPreviewHelper(Activity activity, TextureView textureView, SimpleSceneView simpleSceneView) {
        this.mState = 0;
        this.mCameraOpenCloseLock = new Semaphore(1);
        this.istouchFocuSupported = false;
        this.isOpticalStablisationSupported = false;
        this.isVideoStabilisationSupported = false;
        this.mStateCallback = new CameraDevice.StateCallback() { // from class: com.google.ar.sceneform.CamPreviewHelper.1
            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onDisconnected(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onError(CameraDevice cameraDevice, int i2) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
                Activity activity2 = CamPreviewHelper.this.activity;
                if (activity2 != null) {
                    activity2.finish();
                }
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onOpened(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                CamPreviewHelper.this.mCameraDevice = cameraDevice;
                CamPreviewHelper.this.createCameraPreviewSession();
                Log.d(CamPreviewHelper.TAG, "createCameraPreviewSession");
            }
        };
        TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() { // from class: com.google.ar.sceneform.CamPreviewHelper.2
            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i2, int i3) {
                CamPreviewHelper.this.openCamera(i2, i3);
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return true;
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i2, int i3) {
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
            }
        };
        this.mSurfaceTextureListener = surfaceTextureListener;
        this.activity = activity;
        this.textureView = textureView;
        this.simpleSceneView = simpleSceneView;
        textureView.setSurfaceTextureListener(surfaceTextureListener);
    }

    public CamPreviewHelper(Activity activity, Surface surface, SurfaceTexture surfaceTexture) {
        this.mState = 0;
        this.mCameraOpenCloseLock = new Semaphore(1);
        this.istouchFocuSupported = false;
        this.isOpticalStablisationSupported = false;
        this.isVideoStabilisationSupported = false;
        this.mStateCallback = new CameraDevice.StateCallback() { // from class: com.google.ar.sceneform.CamPreviewHelper.1
            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onDisconnected(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onError(CameraDevice cameraDevice, int i2) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                cameraDevice.close();
                CamPreviewHelper.this.mCameraDevice = null;
                Activity activity2 = CamPreviewHelper.this.activity;
                if (activity2 != null) {
                    activity2.finish();
                }
            }

            @Override // android.hardware.camera2.CameraDevice.StateCallback
            public void onOpened(CameraDevice cameraDevice) {
                CamPreviewHelper.this.mCameraOpenCloseLock.release();
                CamPreviewHelper.this.mCameraDevice = cameraDevice;
                CamPreviewHelper.this.createCameraPreviewSession();
                Log.d(CamPreviewHelper.TAG, "createCameraPreviewSession");
            }
        };
        this.mSurfaceTextureListener = new TextureView.SurfaceTextureListener() { // from class: com.google.ar.sceneform.CamPreviewHelper.2
            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture2, int i2, int i3) {
                CamPreviewHelper.this.openCamera(i2, i3);
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture2) {
                return true;
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture2, int i2, int i3) {
            }

            @Override // android.view.TextureView.SurfaceTextureListener
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture2) {
            }
        };
        this.activity = activity;
        this.surfaceTexture = surfaceTexture;
        this.surface = surface;
    }
}