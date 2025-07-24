package com.ibosoninnov.instanttrackinglib;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketCallback;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;
import com.google.mediapipe.tracking.ModelMatrixProto;
import com.google.protobuf.InvalidProtocolBufferException;
import com.ibosoninnov.instanttrackinglib.InstantTrackingHelper;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/* loaded from: classes2.dex */
public class InstantTrackingHelper {
    private static final String ASPECT_RATIO_SIDE_PACKET_TAG = "aspect_ratio";
    private static final String ASSET_3D_FILE = "robot/robot.obj.uuu";
    private static final String ASSET_3D_TAG = "asset_3d";
    private static final String ASSET_3D_TEXTURE = "robot/robot_texture.jpg";
    private static final String ASSET_3D_TEXTURE_TAG = "texture_3d";
    private static final String BINARY_GRAPH_NAME = "instant_motion_tracking.binarypb";
    private static CameraHelper.CameraFacing CAMERA_FACING = null;
    private static final long CLICK_DURATION = 300;
    private static final String DEFAULT_GIF_TEXTURE = "gif/default_gif_texture.jpg";
    private static final boolean FLIP_FRAMES_VERTICALLY = true;
    private static final String FOV_SIDE_PACKET_TAG = "vertical_fov_radians";
    private static final String GIF_ASPECT_RATIO_TAG = "gif_aspect_ratio";
    private static final String GIF_ASSET_TAG = "gif_asset_name";
    private static final String GIF_FILE = "gif/gif.obj.uuu";
    private static final int GIF_FRAME_RATE = 20;
    private static final String GIF_TEXTURE_TAG = "gif_texture";
    private static final String IMU_MATRIX_TAG = "imu_rotation_matrix";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final int NUM_BUFFERS = 2;
    private static final String OUTPUT_MATRIX_STREAM_NAME = "asset_3d_matrices";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final float ROTATION_SPEED = 5.0f;
    private static final float SCALING_FACTOR = 0.025f;
    private static final int SENSOR_SAMPLE_DELAY = 0;
    private static final String STICKER_PROTO_TAG = "sticker_proto_string";
    private static final String STICKER_SENTINEL_TAG = "sticker_sentinel";
    private static final String TAG = "InstantTrackingHelper";
    private static final float TARGET_CAMERA_ASPECT_RATIO = 0.6f;
    private static final int TARGET_CAMERA_HEIGHT = 800;
    private static final int TARGET_CAMERA_WIDTH = 480;
    private static final float VERTICAL_FOV_RADIANS;
    private Activity activity;
    private Bitmap asset3dTexture;
    private int cameraHeight;
    public CameraXPreviewHelper cameraHelper;
    private int cameraWidth;
    public Handler checkBlackScreenHandler;
    private long clickStartMillis;
    private Context context;
    private ExternalTextureConverter converter;
    private StickerManager currentSticker;
    private Bitmap defaultGIFTexture;
    private GIFEditText editText;
    private EglManager eglManager;
    private ArrayList<Bitmap> gifBitmaps;
    private int gifCurrentIndex;
    private long gifLastFrameUpdateMS;
    private MediaPipePacketManager mediaPipePacketManager;
    public boolean outputCallbackAdded;
    public SurfaceView previewDisplayView;
    private SurfaceTexture previewFrameTexture;
    public FrameProcessor processor;
    public float retryInterval;
    public float[] rotMatFromVec;
    private final float[] rotationMatrix;
    private Surface sceneformSurface;
    private SurfaceTexture sceneformSurfaceTexture;
    private ArrayList<StickerManager> stickerArrayList;
    private int stickerSentinel;
    private TrackingListener trackingListener;
    public boolean trackingOutputReceived;
    private ViewGroup viewGroup;

    /* loaded from: classes2.dex */
    public class MediaPipePacketManager implements FrameProcessor.OnWillAddFrameListener {
        private MediaPipePacketManager() {
        }

        @Override // com.google.mediapipe.components.FrameProcessor.OnWillAddFrameListener
        public void onWillAddFrame(long j) {
            Bitmap bitmap = InstantTrackingHelper.this.defaultGIFTexture;
            if (InstantTrackingHelper.this.gifCurrentIndex <= InstantTrackingHelper.this.gifBitmaps.size() - 1) {
                bitmap = (Bitmap) InstantTrackingHelper.this.gifBitmaps.get(InstantTrackingHelper.this.gifCurrentIndex);
            }
            InstantTrackingHelper.this.updateGIFFrame();
            float width = bitmap.getWidth() / bitmap.getHeight();
            Packet createInt32 = InstantTrackingHelper.this.processor.getPacketCreator().createInt32(InstantTrackingHelper.this.stickerSentinel);
            InstantTrackingHelper.this.stickerSentinel = -1;
            Packet createSerializedProto = InstantTrackingHelper.this.processor.getPacketCreator().createSerializedProto(StickerManager.getMessageLiteData(InstantTrackingHelper.this.stickerArrayList));
            Packet createFloat32Array = InstantTrackingHelper.this.processor.getPacketCreator().createFloat32Array(InstantTrackingHelper.this.rotationMatrix);
            Packet createRgbaImageFrame = InstantTrackingHelper.this.processor.getPacketCreator().createRgbaImageFrame(bitmap);
            Packet createFloat32 = InstantTrackingHelper.this.processor.getPacketCreator().createFloat32(width);
            InstantTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(InstantTrackingHelper.STICKER_SENTINEL_TAG, createInt32, j);
            InstantTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(InstantTrackingHelper.STICKER_PROTO_TAG, createSerializedProto, j);
            InstantTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(InstantTrackingHelper.IMU_MATRIX_TAG, createFloat32Array, j);
            InstantTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(InstantTrackingHelper.GIF_TEXTURE_TAG, createRgbaImageFrame, j);
            InstantTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(InstantTrackingHelper.GIF_ASPECT_RATIO_TAG, createFloat32, j);
            createInt32.release();
            createSerializedProto.release();
            createFloat32Array.release();
            createRgbaImageFrame.release();
            createFloat32.release();
        }
    }

    /* loaded from: classes2.dex */
    public interface TrackingListener {
        void onTracking(List<Float> list);
    }

    static {
        System.loadLibrary("mediapipe_jni");
        try {
            System.loadLibrary("opencv_java3");
        } catch (UnsatisfiedLinkError unused) {
            System.loadLibrary("opencv_java4");
        }
        CAMERA_FACING = CameraHelper.CameraFacing.BACK;
        VERTICAL_FOV_RADIANS = (float) Math.toRadians(68.0d);
    }

    public InstantTrackingHelper(Context context, Activity activity, ViewGroup viewGroup, TrackingListener trackingListener) {
        this.clickStartMillis = 0L;
        this.stickerSentinel = -1;
        this.rotationMatrix = new float[9];
        this.asset3dTexture = null;
        this.gifBitmaps = new ArrayList<>();
        this.gifCurrentIndex = 0;
        this.defaultGIFTexture = null;
        this.gifLastFrameUpdateMS = System.currentTimeMillis();
        this.cameraWidth = 480;
        this.cameraHeight = 800;
        this.outputCallbackAdded = false;
        this.trackingOutputReceived = false;
        this.retryInterval = 1.0f;
        this.rotMatFromVec = new float[]{1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f};
        this.activity = activity;
        this.context = context;
        this.viewGroup = viewGroup;
        this.trackingListener = trackingListener;
        init();
    }

    private void addSticker(float f2, float f3) {
        StickerManager stickerManager = new StickerManager();
        this.stickerArrayList.add(stickerManager);
        this.currentSticker = stickerManager;
        stickerManager.setAnchorCoordinate(f2, f3);
        this.currentSticker.setScaleFactor(0.01f);
    }

    private static float calculateRotationRadians(MotionEvent motionEvent) {
        float degrees = ((float) Math.toDegrees(((float) Math.atan2(motionEvent.getY(1) - motionEvent.getY(0), motionEvent.getX(1) - motionEvent.getX(0))) - ((float) Math.atan2(motionEvent.getHistoricalY(1, 0) - motionEvent.getHistoricalY(0, 0), motionEvent.getHistoricalX(1, 0) - motionEvent.getHistoricalX(0, 0))))) % 360.0f;
        return (float) ((((degrees + (degrees >= -180.0f ? degrees > 180.0f ? -360.0f : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD : 360.0f)) * ROTATION_SPEED) / 180.0f) * 3.141592653589793d);
    }

    private void checkAndHandleBlackScreenBug() {
        this.trackingOutputReceived = false;
        Handler handler = new Handler();
        this.checkBlackScreenHandler = handler;
        handler.postDelayed(new Runnable() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.4
            @Override // java.lang.Runnable
            public void run() {
                InstantTrackingHelper instantTrackingHelper = InstantTrackingHelper.this;
                if (instantTrackingHelper.trackingOutputReceived) {
                    return;
                }
                instantTrackingHelper.restartInstantTracking();
            }
        }, this.retryInterval * 2000.0f);
    }

    private void deleteSticker() {
        StickerManager stickerManager = this.currentSticker;
        if (stickerManager != null) {
            this.stickerArrayList.remove(stickerManager);
            this.stickerArrayList.clear();
            this.currentSticker = null;
        }
    }

    private static Bitmap flipHorizontal(Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.preScale(-1.0f, 1.0f);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    private static double getDistance(double d2, double d3, double d4, double d5) {
        return Math.hypot(d5 - d3, d4 - d2);
    }

    private static float getNewScaleFactor(MotionEvent motionEvent, float f2) {
        return ((getDistance((double) motionEvent.getX(0), (double) motionEvent.getY(0), (double) motionEvent.getX(1), (double) motionEvent.getY(1)) < getDistance((double) motionEvent.getHistoricalX(0, 0), (double) motionEvent.getHistoricalY(0, 0), (double) motionEvent.getHistoricalX(1, 0), (double) motionEvent.getHistoricalY(1, 0)) ? -0.025f : SCALING_FACTOR) + 1.0f) * f2;
    }

    private void init() {
        this.previewDisplayView = new SurfaceView(this.context);
        setupPreviewDisplayView();
        AndroidAssetUtil.initializeNativeAssetManager(this.context);
        EglManager eglManager = new EglManager(null);
        this.eglManager = eglManager;
        FrameProcessor frameProcessor = new FrameProcessor(this.context, eglManager.getNativeContext(), BINARY_GRAPH_NAME, INPUT_VIDEO_STREAM_NAME, OUTPUT_VIDEO_STREAM_NAME);
        this.processor = frameProcessor;
        frameProcessor.getVideoSurfaceOutput().setFlipY(true);
        PermissionHelper.checkAndRequestCameraPermissions(this.activity);
        prepareDemoAssets();
        AndroidPacketCreator packetCreator = this.processor.getPacketCreator();
        HashMap hashMap = new HashMap();
        hashMap.put(ASSET_3D_TEXTURE_TAG, packetCreator.createRgbaImageFrame(this.asset3dTexture));
        hashMap.put(ASSET_3D_TAG, packetCreator.createString(ASSET_3D_FILE));
        hashMap.put(GIF_ASSET_TAG, packetCreator.createString(GIF_FILE));
        this.processor.setInputSidePackets(hashMap);
        MediaPipePacketManager mediaPipePacketManager = new MediaPipePacketManager();
        this.mediaPipePacketManager = mediaPipePacketManager;
        this.processor.setOnWillAddFrameListener(mediaPipePacketManager);
        HashMap hashMap2 = new HashMap();
        hashMap2.put(ASPECT_RATIO_SIDE_PACKET_TAG, packetCreator.createFloat32(TARGET_CAMERA_ASPECT_RATIO));
        hashMap2.put(FOV_SIDE_PACKET_TAG, packetCreator.createFloat32(VERTICAL_FOV_RADIANS));
        this.processor.setInputSidePackets(hashMap2);
        this.stickerArrayList = new ArrayList<>();
        this.currentSticker = null;
        SensorManager sensorManager = (SensorManager) this.context.getSystemService("sensor");
        Sensor defaultSensor = sensorManager.getDefaultSensor(11);
        Sensor defaultSensor2 = sensorManager.getDefaultSensor(9);
        if (defaultSensor != null) {
            sensorManager.registerListener(new SensorEventListener() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.1
                @Override // android.hardware.SensorEventListener
                public void onAccuracyChanged(Sensor sensor, int i) {
                }

                @Override // android.hardware.SensorEventListener
                public void onSensorChanged(SensorEvent sensorEvent) {
                    SensorManager.getRotationMatrixFromVector(InstantTrackingHelper.this.rotMatFromVec, sensorEvent.values);
                    InstantTrackingHelper instantTrackingHelper = InstantTrackingHelper.this;
                    SensorManager.remapCoordinateSystem(instantTrackingHelper.rotMatFromVec, 129, 2, instantTrackingHelper.rotationMatrix);
                }
            }, defaultSensor, 0);
        } else {
            sensorManager.registerListener(new SensorEventListener() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.2
                @Override // android.hardware.SensorEventListener
                public void onAccuracyChanged(Sensor sensor, int i) {
                }

                @Override // android.hardware.SensorEventListener
                public void onSensorChanged(SensorEvent sensorEvent) {
                    SensorManager.getRotationMatrixFromVector(InstantTrackingHelper.this.rotMatFromVec, sensorEvent.values);
                    InstantTrackingHelper instantTrackingHelper = InstantTrackingHelper.this;
                    SensorManager.remapCoordinateSystem(instantTrackingHelper.rotMatFromVec, 129, 2, instantTrackingHelper.rotationMatrix);
                }
            }, defaultSensor2, 0);
        }
        SensorManager.remapCoordinateSystem(this.rotMatFromVec, 129, 2, this.rotationMatrix);
        this.viewGroup.setOnTouchListener(new View.OnTouchListener() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.3
            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                return InstantTrackingHelper.this.manageUiTouch(motionEvent);
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean manageUiTouch(MotionEvent motionEvent) {
        if (this.currentSticker != null) {
            int action = motionEvent.getAction();
            if (action == 0) {
                this.clickStartMillis = System.currentTimeMillis();
            } else if (action != 1) {
                if (action == 2 && motionEvent.getPointerCount() == 2 && motionEvent.getHistorySize() > 1) {
                    this.currentSticker.setScaleFactor(getNewScaleFactor(motionEvent, this.currentSticker.getScaleFactor()));
                    float calculateRotationRadians = calculateRotationRadians(motionEvent);
                    StickerManager stickerManager = this.currentSticker;
                    stickerManager.setRotation(stickerManager.getRotation() + calculateRotationRadians);
                }
            } else if (System.currentTimeMillis() - this.clickStartMillis <= CLICK_DURATION) {
                recordClick(motionEvent);
            }
        }
        return true;
    }

    private void prepareDemoAssets() {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        options.inDither = false;
        options.inPremultiplied = false;
        try {
            InputStream open = this.context.getAssets().open(DEFAULT_GIF_TEXTURE);
            this.defaultGIFTexture = flipHorizontal(BitmapFactory.decodeStream(open, null, options));
            open.close();
            try {
                InputStream open2 = this.context.getAssets().open(ASSET_3D_TEXTURE);
                this.asset3dTexture = BitmapFactory.decodeStream(open2, null, options);
                open2.close();
            } catch (Exception e2) {
                Log.e(TAG, "Error parsing object texture; error: ", e2);
                throw new IllegalStateException(e2);
            }
        } catch (Exception e3) {
            Log.e(TAG, "Error parsing object texture; error: ", e3);
            throw new IllegalStateException(e3);
        }
    }

    private void recordClick(MotionEvent motionEvent) {
        float x = motionEvent.getX() / this.viewGroup.getWidth();
        float y = motionEvent.getY() / this.viewGroup.getHeight();
        float width = this.viewGroup.getWidth() / this.cameraWidth;
        float height = this.viewGroup.getHeight() / this.cameraHeight;
        float max = Math.max(width, height);
        float f2 = width / max;
        float f3 = height / max;
        this.currentSticker.setAnchorCoordinate(a.a(1.0f, f2, 0.5f, x * f2), a.a(1.0f, f3, 0.5f, y * f3));
        this.stickerSentinel = this.currentSticker.getstickerId();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void restartInstantTracking() {
        onPause();
        new Handler().postDelayed(new Runnable() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.5
            @Override // java.lang.Runnable
            public void run() {
                InstantTrackingHelper.this.onResume();
            }
        }, this.retryInterval * 1000.0f);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUpOutputListener() {
        if (this.outputCallbackAdded) {
            return;
        }
        this.outputCallbackAdded = true;
        this.processor.addPacketCallback(OUTPUT_MATRIX_STREAM_NAME, new PacketCallback() { // from class: c.e.a.i
            @Override // com.google.mediapipe.framework.PacketCallback
            public final void process(Packet packet) {
                InstantTrackingHelper.this.a(packet);
            }
        });
    }

    private void setupPreviewDisplayView() {
        this.previewDisplayView.setVisibility(8);
        this.viewGroup.addView(this.previewDisplayView);
        this.previewDisplayView.getHolder().addCallback(new SurfaceHolder.Callback() { // from class: com.ibosoninnov.instanttrackinglib.InstantTrackingHelper.6
            @Override // android.view.SurfaceHolder.Callback
            public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
                InstantTrackingHelper.this.onPreviewDisplaySurfaceChanged(surfaceHolder, i, i2, i3);
            }

            @Override // android.view.SurfaceHolder.Callback
            public void surfaceCreated(SurfaceHolder surfaceHolder) {
                StringBuilder x = a.x("surfaceCreated - setSurface to sceneform ");
                x.append(surfaceHolder.getSurfaceFrame().width());
                x.append(" x ");
                x.append(surfaceHolder.getSurfaceFrame().height());
                Log.d(InstantTrackingHelper.TAG, x.toString());
                if (InstantTrackingHelper.this.sceneformSurfaceTexture != null) {
                    InstantTrackingHelper.this.sceneformSurfaceTexture.setDefaultBufferSize(surfaceHolder.getSurfaceFrame().width(), surfaceHolder.getSurfaceFrame().height());
                    InstantTrackingHelper.this.processor.getVideoSurfaceOutput().setSurface(InstantTrackingHelper.this.sceneformSurface);
                } else {
                    InstantTrackingHelper.this.processor.getVideoSurfaceOutput().setSurface(surfaceHolder.getSurface());
                }
                InstantTrackingHelper.this.setUpOutputListener();
            }

            @Override // android.view.SurfaceHolder.Callback
            public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
                InstantTrackingHelper.this.processor.getVideoSurfaceOutput().setSurface(null);
                Log.d(InstantTrackingHelper.TAG, "surfaceDestroyed");
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void updateGIFFrame() {
        if (System.currentTimeMillis() - this.gifLastFrameUpdateMS >= 50) {
            this.gifLastFrameUpdateMS = System.currentTimeMillis();
            this.gifCurrentIndex = this.gifBitmaps.isEmpty() ? 1 : (this.gifCurrentIndex + 1) % this.gifBitmaps.size();
        }
    }

    public /* synthetic */ void a(Packet packet) {
        byte[] protoBytes = PacketGetter.getProtoBytes(packet);
        if (protoBytes.length > 0) {
            try {
                this.trackingOutputReceived = true;
                ModelMatrixProto.TimedVectorProtoList parseFrom = ModelMatrixProto.TimedVectorProtoList.parseFrom(protoBytes);
                if (parseFrom != null) {
                    this.trackingListener.onTracking(parseFrom.getVectorList(0).getVectorEntriesList());
                } else {
                    this.trackingListener.onTracking(null);
                }
            } catch (InvalidProtocolBufferException e2) {
                e2.printStackTrace();
            }
        }
    }

    public Size cameraTargetResolution() {
        return new Size(800, 480);
    }

    public Size computeViewSize(int i, int i2) {
        return new Size(i2, (int) (i2 * TARGET_CAMERA_ASPECT_RATIO));
    }

    public SurfaceTexture getPreviewFrameTexture() {
        return this.previewFrameTexture;
    }

    public boolean isFrontCamera() {
        return CAMERA_FACING == CameraHelper.CameraFacing.FRONT;
    }

    public void onCameraStarted(SurfaceTexture surfaceTexture) {
        this.previewFrameTexture = surfaceTexture;
        this.previewDisplayView.setVisibility(0);
        checkAndHandleBlackScreenBug();
    }

    public void onPause() {
        try {
            ExternalTextureConverter externalTextureConverter = this.converter;
            if (externalTextureConverter != null) {
                externalTextureConverter.close();
            }
            SurfaceTexture surfaceTexture = this.previewFrameTexture;
            if (surfaceTexture != null) {
                surfaceTexture.release();
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
        this.previewDisplayView.setVisibility(8);
    }

    public void onPreviewDisplaySurfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
        Size computeDisplaySizeFromViewSize = this.cameraHelper.computeDisplaySizeFromViewSize(computeViewSize(i2, i3));
        boolean isCameraRotated = this.cameraHelper.isCameraRotated();
        this.converter.setSurfaceTextureAndAttachToGLContext(this.previewFrameTexture, isCameraRotated ? computeDisplaySizeFromViewSize.getHeight() : computeDisplaySizeFromViewSize.getWidth(), isCameraRotated ? computeDisplaySizeFromViewSize.getWidth() : computeDisplaySizeFromViewSize.getHeight());
    }

    public void onResume() {
        ExternalTextureConverter externalTextureConverter = new ExternalTextureConverter(this.eglManager.getContext(), 2);
        this.converter = externalTextureConverter;
        externalTextureConverter.setFlipY(true);
        this.converter.setConsumer(this.processor);
        if (PermissionHelper.cameraPermissionsGranted(this.activity)) {
            startCamera();
        }
    }

    public void resetAnchor(float f2, float f3) {
        stopTracking();
        addSticker(f2, f3);
    }

    public void setRetryInterval(float f2) {
        this.retryInterval = f2;
    }

    public void startCamera() {
        CameraXPreviewHelper cameraXPreviewHelper = new CameraXPreviewHelper();
        this.cameraHelper = cameraXPreviewHelper;
        cameraXPreviewHelper.setOnCameraStartedListener(new CameraHelper.OnCameraStartedListener() { // from class: c.e.a.h
            @Override // com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
            public final void onCameraStarted(SurfaceTexture surfaceTexture) {
                InstantTrackingHelper.this.onCameraStarted(surfaceTexture);
            }
        });
        this.cameraHelper.startCamera(this.activity, CAMERA_FACING, (SurfaceTexture) null, cameraTargetResolution());
    }

    public void startTracking() {
        addSticker(0.5f, 0.5f);
    }

    public void stopTracking() {
        deleteSticker();
    }

    public void swapCamera() {
        onPause();
        CameraHelper.CameraFacing cameraFacing = CAMERA_FACING;
        CameraHelper.CameraFacing cameraFacing2 = CameraHelper.CameraFacing.BACK;
        if (cameraFacing == cameraFacing2) {
            CAMERA_FACING = CameraHelper.CameraFacing.FRONT;
        } else {
            CAMERA_FACING = cameraFacing2;
        }
        onResume();
    }

    public boolean toggleFlash(boolean z) {
        CameraXPreviewHelper cameraXPreviewHelper = this.cameraHelper;
        if (cameraXPreviewHelper == null || cameraXPreviewHelper.getCamera() == null || !this.cameraHelper.getCamera().b().e()) {
            return false;
        }
        this.cameraHelper.getCamera().a().b(z);
        return false;
    }

    public void startTracking(float f2, float f3) {
        addSticker(f2, f3);
    }

    public InstantTrackingHelper(Context context, Activity activity, ViewGroup viewGroup, Surface surface, SurfaceTexture surfaceTexture, TrackingListener trackingListener) {
        this.clickStartMillis = 0L;
        this.stickerSentinel = -1;
        this.rotationMatrix = new float[9];
        this.asset3dTexture = null;
        this.gifBitmaps = new ArrayList<>();
        this.gifCurrentIndex = 0;
        this.defaultGIFTexture = null;
        this.gifLastFrameUpdateMS = System.currentTimeMillis();
        this.cameraWidth = 480;
        this.cameraHeight = 800;
        this.outputCallbackAdded = false;
        this.trackingOutputReceived = false;
        this.retryInterval = 1.0f;
        this.rotMatFromVec = new float[]{1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, 1.0f};
        this.activity = activity;
        this.context = context;
        this.viewGroup = viewGroup;
        this.trackingListener = trackingListener;
        this.sceneformSurface = surface;
        this.sceneformSurfaceTexture = surfaceTexture;
        init();
    }
}