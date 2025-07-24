package com.ibosoninnov.instanttrackinglib;

import android.app.Activity;
import android.content.Context;
import android.graphics.PointF;
import android.graphics.SurfaceTexture;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketCallback;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.glutil.EglManager;
import com.google.mediapipe.tracking.BoxTrackerProto;
import com.google.protobuf.InvalidProtocolBufferException;
import com.ibosoninnov.instanttrackinglib.BoxTrackingHelper;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes2.dex */
public class BoxTrackingHelper {
    private static final String BINARY_GRAPH_NAME = "mobile_gpu.binarypb";
    private static final String CANCEL_OBJECT_ID = "cancel_object_id";
    private static final boolean FLIP_FRAMES_VERTICALLY = true;
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final int NUM_BUFFERS = 2;
    private static final String OUTPUT_BOXES_STREAM_NAME = "boxes";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String START_POS = "start_pos";
    private static final String TAG = "BoxTrackingHelper";
    private static final CameraHelper.CameraFacing cameraFacing = CameraHelper.CameraFacing.BACK;
    private Activity activity;
    public List<PointF> boxAnchors;
    public com.google.mediapipe.components.CameraXPreviewHelper cameraHelper;
    private Context context;
    private ExternalTextureConverter converter;
    private EglManager eglManager;
    private SurfaceView previewDisplayView;
    private SurfaceTexture previewFrameTexture;
    public FrameProcessor processor;
    private Surface sceneformSurface;
    private SurfaceTexture sceneformSurfaceTexture;
    public float screenHeight;
    public float screenWidth;
    public List<PointF> trackedBoxAnchors;
    private BoxTrackingListener trackingListener;
    private ViewGroup viewGroup;
    public boolean addBox = false;
    public boolean clearBoxTracking = false;
    public boolean outputCallbackAdded = false;

    /* loaded from: classes2.dex */
    public interface BoxTrackingListener {
        void onTracking(List<PointF> list);
    }

    static {
        System.loadLibrary("mediapipe_jni");
        try {
            System.loadLibrary("opencv_java3");
        } catch (UnsatisfiedLinkError unused) {
            System.loadLibrary("opencv_java4");
        }
    }

    public BoxTrackingHelper(Activity activity, Context context, ViewGroup viewGroup, BoxTrackingListener boxTrackingListener) {
        this.activity = activity;
        this.context = context;
        this.viewGroup = viewGroup;
        this.trackingListener = boxTrackingListener;
        init();
    }

    private void init() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        this.activity.getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        this.screenHeight = displayMetrics.heightPixels;
        this.screenWidth = displayMetrics.widthPixels;
        this.previewDisplayView = new SurfaceView(this.context);
        setupPreviewDisplayView();
        AndroidAssetUtil.initializeNativeAssetManager(this.context);
        EglManager eglManager = new EglManager(null);
        this.eglManager = eglManager;
        FrameProcessor frameProcessor = new FrameProcessor(this.context, eglManager.getNativeContext(), BINARY_GRAPH_NAME, INPUT_VIDEO_STREAM_NAME, OUTPUT_VIDEO_STREAM_NAME);
        this.processor = frameProcessor;
        frameProcessor.getVideoSurfaceOutput().setFlipY(true);
        setInputRect();
        PermissionHelper.checkAndRequestCameraPermissions(this.activity);
    }

    private void setInputRect() {
        this.boxAnchors = new ArrayList();
        this.trackedBoxAnchors = new ArrayList();
        ProtoUtil.registerTypeName(BoxTrackerProto.TimedBoxProtoList.class, "mediapipe.TimedBoxProtoList");
        this.processor.setOnWillAddFrameListener(new FrameProcessor.OnWillAddFrameListener() { // from class: com.ibosoninnov.instanttrackinglib.BoxTrackingHelper.1
            @Override // com.google.mediapipe.components.FrameProcessor.OnWillAddFrameListener
            public void onWillAddFrame(long j) {
                BoxTrackingHelper boxTrackingHelper = BoxTrackingHelper.this;
                if (boxTrackingHelper.clearBoxTracking) {
                    boxTrackingHelper.clearBoxTracking = false;
                    for (int i = 0; i < BoxTrackingHelper.this.boxAnchors.size(); i++) {
                        Packet createInt32 = BoxTrackingHelper.this.processor.getPacketCreator().createInt32(i);
                        BoxTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(BoxTrackingHelper.CANCEL_OBJECT_ID, createInt32, i + j);
                        createInt32.release();
                    }
                    Log.d(BoxTrackingHelper.TAG, "Cleared box tracking");
                }
                BoxTrackingHelper boxTrackingHelper2 = BoxTrackingHelper.this;
                if (boxTrackingHelper2.addBox) {
                    boxTrackingHelper2.addBox = false;
                    float f2 = boxTrackingHelper2.screenHeight / boxTrackingHelper2.screenWidth;
                    BoxTrackerProto.TimedBoxProtoList.Builder newBuilder = BoxTrackerProto.TimedBoxProtoList.newBuilder();
                    for (int i2 = 0; i2 < BoxTrackingHelper.this.boxAnchors.size(); i2++) {
                        PointF pointF = BoxTrackingHelper.this.boxAnchors.get(i2);
                        float f3 = pointF.x;
                        float f4 = pointF.y;
                        float f5 = 0.05f / f2;
                        newBuilder.addBox(BoxTrackerProto.TimedBoxProto.newBuilder().setId(i2).setLeft(f3 - 0.05f).setTop(f4 - f5).setRight(f3 + 0.05f).setBottom(f4 + f5).setAspectRatio(1.0f).setConfidence(0.9f).setLabel("box" + i2).setRotation(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).build());
                    }
                    Packet createProto = BoxTrackingHelper.this.processor.getPacketCreator().createProto(newBuilder.build());
                    BoxTrackingHelper.this.processor.getGraph().addConsumablePacketToInputStream(BoxTrackingHelper.START_POS, createProto, j);
                    createProto.release();
                    Log.d(BoxTrackingHelper.TAG, "onWillAddFrame");
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setUpOutputListener() {
        if (this.outputCallbackAdded) {
            return;
        }
        this.outputCallbackAdded = true;
        this.processor.addPacketCallback(OUTPUT_BOXES_STREAM_NAME, new PacketCallback() { // from class: c.e.a.a
            @Override // com.google.mediapipe.framework.PacketCallback
            public final void process(Packet packet) {
                BoxTrackingHelper.this.a(packet);
            }
        });
    }

    private void setupPreviewDisplayView() {
        this.previewDisplayView.setVisibility(8);
        this.viewGroup.addView(this.previewDisplayView);
        this.previewDisplayView.getHolder().addCallback(new SurfaceHolder.Callback() { // from class: com.ibosoninnov.instanttrackinglib.BoxTrackingHelper.2
            @Override // android.view.SurfaceHolder.Callback
            public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
                BoxTrackingHelper.this.onPreviewDisplaySurfaceChanged(surfaceHolder, i, i2, i3);
            }

            @Override // android.view.SurfaceHolder.Callback
            public void surfaceCreated(SurfaceHolder surfaceHolder) {
                BoxTrackingHelper.this.processor.getVideoSurfaceOutput().setSurface(surfaceHolder.getSurface());
                Log.d(BoxTrackingHelper.TAG, "surfaceCreated");
                BoxTrackingHelper.this.setUpOutputListener();
            }

            @Override // android.view.SurfaceHolder.Callback
            public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
                BoxTrackingHelper.this.processor.getVideoSurfaceOutput().setSurface(null);
            }
        });
    }

    public /* synthetic */ void a(Packet packet) {
        byte[] protoBytes = PacketGetter.getProtoBytes(packet);
        if (protoBytes.length > 0) {
            try {
                this.trackedBoxAnchors.clear();
                for (BoxTrackerProto.TimedBoxProto timedBoxProto : BoxTrackerProto.TimedBoxProtoList.parseFrom(protoBytes).getBoxList()) {
                    this.trackedBoxAnchors.add(new PointF((timedBoxProto.getLeft() + timedBoxProto.getRight()) / 2.0f, (timedBoxProto.getTop() + timedBoxProto.getBottom()) / 2.0f));
                    Log.v(TAG, "Box " + timedBoxProto.getId() + " conf=" + timedBoxProto.getConfidence() + " " + timedBoxProto.getLeft() + ", " + timedBoxProto.getTop() + "   " + timedBoxProto.getRight() + ", " + timedBoxProto.getBottom());
                }
                this.trackingListener.onTracking(this.trackedBoxAnchors);
            } catch (InvalidProtocolBufferException e2) {
                Log.e(TAG, e2.toString());
            }
        }
    }

    public void addAnchors(List<PointF> list) {
        this.addBox = true;
        this.clearBoxTracking = true;
        this.boxAnchors = list;
    }

    public Size cameraTargetResolution() {
        return null;
    }

    public void clearBoxTracking() {
        this.clearBoxTracking = true;
    }

    public Size computeViewSize(int i, int i2) {
        return new Size(i, i2);
    }

    public void onCameraStarted(SurfaceTexture surfaceTexture) {
        this.previewFrameTexture = surfaceTexture;
        this.previewDisplayView.setVisibility(0);
        Log.d(TAG, "onCameraStarted");
    }

    public void onPause() {
        this.converter.close();
        this.previewDisplayView.setVisibility(8);
    }

    public void onPreviewDisplaySurfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
        Size computeDisplaySizeFromViewSize = this.cameraHelper.computeDisplaySizeFromViewSize(computeViewSize(i2, i3));
        boolean isCameraRotated = this.cameraHelper.isCameraRotated();
        this.converter.setSurfaceTextureAndAttachToGLContext(this.previewFrameTexture, isCameraRotated ? computeDisplaySizeFromViewSize.getHeight() : computeDisplaySizeFromViewSize.getWidth(), isCameraRotated ? computeDisplaySizeFromViewSize.getWidth() : computeDisplaySizeFromViewSize.getHeight());
    }

    public void onRequestPermissionsResult(int i, String[] strArr, int[] iArr) {
        PermissionHelper.onRequestPermissionsResult(i, strArr, iArr);
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

    public void startCamera() {
        com.google.mediapipe.components.CameraXPreviewHelper cameraXPreviewHelper = new com.google.mediapipe.components.CameraXPreviewHelper();
        this.cameraHelper = cameraXPreviewHelper;
        cameraXPreviewHelper.setOnCameraStartedListener(new CameraHelper.OnCameraStartedListener() { // from class: c.e.a.b
            @Override // com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
            public final void onCameraStarted(SurfaceTexture surfaceTexture) {
                BoxTrackingHelper.this.onCameraStarted(surfaceTexture);
            }
        });
        this.cameraHelper.startCamera(this.activity, cameraFacing, (SurfaceTexture) null, cameraTargetResolution());
    }

    public BoxTrackingHelper(Activity activity, Context context, ViewGroup viewGroup, Surface surface, SurfaceTexture surfaceTexture, BoxTrackingListener boxTrackingListener) {
        this.activity = activity;
        this.context = context;
        this.viewGroup = viewGroup;
        this.trackingListener = boxTrackingListener;
        this.sceneformSurface = surface;
        this.sceneformSurfaceTexture = surfaceTexture;
        init();
    }
}