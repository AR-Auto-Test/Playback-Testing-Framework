package c.e.b;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
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
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SparseArray;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import c.e.b.hd;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.barcode.Barcode;
import com.google.android.gms.vision.barcode.BarcodeDetector;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.firebase.crashlytics.internal.common.CrashlyticsReportDataCapture;
import com.ibosoninnov.unitear.ARCoreSceneformActivity;
import com.ibosoninnov.unitear.AutoFitTextureView;
import com.ibosoninnov.unitear.CVLib;
import com.ibosoninnov.unitear.ImageTrackingActivity;
import com.ibosoninnov.unitear.NonARCoreActivitySceneform;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Objects;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/* compiled from: CamPreviewHelper.java */
/* loaded from: classes2.dex */
public class yb {

    /* renamed from: a  reason: collision with root package name */
    public static int f5438a;

    /* renamed from: b  reason: collision with root package name */
    public static float f5439b;

    /* renamed from: c  reason: collision with root package name */
    public static float f5440c;

    /* renamed from: d  reason: collision with root package name */
    public static final SparseIntArray f5441d;

    /* renamed from: e  reason: collision with root package name */
    public static int f5442e;

    /* renamed from: f  reason: collision with root package name */
    public static int f5443f;
    public final CameraDevice.StateCallback A;
    public final ImageReader.OnImageAvailableListener B;

    /* renamed from: g  reason: collision with root package name */
    public Activity f5444g;

    /* renamed from: h  reason: collision with root package name */
    public int f5445h;
    public AutoFitTextureView i;
    public String j;
    public CameraCaptureSession k;
    public CameraDevice l;
    public Size m;
    public ImageReader n;
    public CaptureRequest.Builder o;
    public CaptureRequest p;
    public Range<Integer> r;
    public int s;
    public Surface t;
    public Handler u;
    public HandlerThread v;
    public boolean w;
    public e y;
    public final TextureView.SurfaceTextureListener z;
    public Semaphore q = new Semaphore(1);
    public boolean x = false;

    /* compiled from: CamPreviewHelper.java */
    /* loaded from: classes2.dex */
    public class a implements TextureView.SurfaceTextureListener {
        public a() {
        }

        @Override // android.view.TextureView.SurfaceTextureListener
        public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i2) {
            Log.d("CCV2WithPreview", "onSurfaceTextureAvailable - opencamera");
            yb.this.e(i, i2);
        }

        @Override // android.view.TextureView.SurfaceTextureListener
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
            Log.d("CCV2WithPreview", "onSurfaceTextureDestroyed");
            yb.this.x = true;
            return false;
        }

        @Override // android.view.TextureView.SurfaceTextureListener
        public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i2) {
            yb.this.c(i, i2);
        }

        @Override // android.view.TextureView.SurfaceTextureListener
        public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
        }
    }

    /* compiled from: CamPreviewHelper.java */
    /* loaded from: classes2.dex */
    public class b extends CameraDevice.StateCallback {
        public b() {
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onDisconnected(CameraDevice cameraDevice) {
            yb.this.q.release();
            cameraDevice.close();
            yb.this.l = null;
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onError(CameraDevice cameraDevice, int i) {
            yb.this.q.release();
            cameraDevice.close();
            yb ybVar = yb.this;
            ybVar.l = null;
            Activity activity = ybVar.f5444g;
            if (activity != null) {
                activity.finish();
            }
        }

        @Override // android.hardware.camera2.CameraDevice.StateCallback
        public void onOpened(CameraDevice cameraDevice) {
            yb.this.q.release();
            yb ybVar = yb.this;
            ybVar.l = cameraDevice;
            Objects.requireNonNull(ybVar);
            Log.d("CCV2WithPreview", "createCameraPreviewSession");
            try {
                SurfaceTexture surfaceTexture = ybVar.i.getSurfaceTexture();
                if (surfaceTexture == null) {
                    Log.d("CCV2WithPreview", "texture null");
                    return;
                }
                surfaceTexture.setDefaultBufferSize(ybVar.m.getWidth(), ybVar.m.getHeight());
                if (ybVar.t == null || ybVar.x) {
                    ybVar.x = false;
                    ybVar.t = new Surface(surfaceTexture);
                }
                CaptureRequest.Builder createCaptureRequest = ybVar.l.createCaptureRequest(1);
                ybVar.o = createCaptureRequest;
                createCaptureRequest.addTarget(ybVar.n.getSurface());
                ybVar.l.createCaptureSession(Arrays.asList(ybVar.n.getSurface()), new zb(ybVar), null);
            } catch (CameraAccessException e2) {
                Log.e("CCV2WithPreview", e2.toString());
            }
        }
    }

    /* compiled from: CamPreviewHelper.java */
    /* loaded from: classes2.dex */
    public class c implements ImageReader.OnImageAvailableListener {
        public c() {
        }

        /* JADX DEBUG: Multi-variable search result rejected for r13v2, resolved type: int */
        /* JADX WARN: Multi-variable type inference failed */
        /* JADX WARN: Removed duplicated region for block: B:39:0x0241  */
        /* JADX WARN: Removed duplicated region for block: B:51:0x0270  */
        @Override // android.media.ImageReader.OnImageAvailableListener
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public void onImageAvailable(ImageReader imageReader) {
            boolean z;
            final String str;
            float f2;
            float f3;
            float f4;
            float f5;
            float f6;
            float f7;
            float f8;
            float f9;
            Node node;
            Image acquireLatestImage = imageReader.acquireLatestImage();
            if (acquireLatestImage != null) {
                e eVar = yb.this.y;
                int width = acquireLatestImage.getWidth();
                int height = acquireLatestImage.getHeight();
                final ImageTrackingActivity imageTrackingActivity = (ImageTrackingActivity) eVar;
                if (imageTrackingActivity.W) {
                    if (acquireLatestImage.getFormat() != 35) {
                        Log.e("ImageTrackingActivity", "camera image is in wrong format");
                    }
                    Image.Plane plane = acquireLatestImage.getPlanes()[0];
                    int rowStride = plane.getRowStride();
                    Image.Plane plane2 = acquireLatestImage.getPlanes()[1];
                    int rowStride2 = plane2.getRowStride();
                    Image.Plane plane3 = acquireLatestImage.getPlanes()[2];
                    int rowStride3 = plane3.getRowStride();
                    int pixelStride = plane3.getPixelStride();
                    if (!imageTrackingActivity.X) {
                        imageTrackingActivity.X = true;
                        float f10 = imageTrackingActivity.T < 2500 ? 2.0f : 1.5f;
                        StringBuilder x = c.b.a.a.a.x("CameraMatrix ");
                        x.append(acquireLatestImage.getHeight() / f10);
                        x.append("x");
                        x.append(acquireLatestImage.getWidth() / f10);
                        x.append(" FOV ");
                        Objects.requireNonNull(imageTrackingActivity.s);
                        x.append(yb.f5439b);
                        x.append(", ");
                        Objects.requireNonNull(imageTrackingActivity.s);
                        x.append(yb.f5440c);
                        Log.d("ImageTrackingActivity", x.toString());
                        CVLib cVLib = imageTrackingActivity.u0;
                        float height2 = acquireLatestImage.getHeight() / f10;
                        float width2 = acquireLatestImage.getWidth() / f10;
                        Objects.requireNonNull(imageTrackingActivity.s);
                        float f11 = yb.f5439b;
                        Objects.requireNonNull(imageTrackingActivity.s);
                        cVLib.patternDetectorSetCameraMatrixJNI(height2, width2, f11, yb.f5440c, imageTrackingActivity.Q, f10);
                        Objects.requireNonNull(imageTrackingActivity.s);
                        int i = yb.f5438a;
                        c.e.b.p000if.d dVar = imageTrackingActivity.C;
                        dVar.f4872b.putInt("sensorOrientation", i);
                        dVar.f4872b.apply();
                    }
                    imageTrackingActivity.u0.onImageAvailableJNI(acquireLatestImage.getWidth(), acquireLatestImage.getHeight(), pixelStride, rowStride, plane.getBuffer(), rowStride2, plane2.getBuffer(), rowStride3, plane3.getBuffer(), imageTrackingActivity.u.getSurface(), acquireLatestImage.getTimestamp(), false, imageTrackingActivity.j0);
                    if (imageTrackingActivity.g0) {
                        imageTrackingActivity.e0 = false;
                    } else {
                        if (imageTrackingActivity.u0.getTrackStatusJNI()) {
                            if (!imageTrackingActivity.e0) {
                                imageTrackingActivity.V = System.currentTimeMillis();
                                imageTrackingActivity.c0 = true;
                                imageTrackingActivity.Y = false;
                                imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.l2
                                    @Override // java.lang.Runnable
                                    public final void run() {
                                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                        imageTrackingActivity2.C0.setVisibility(8);
                                        imageTrackingActivity2.P0.setVisibility(8);
                                        imageTrackingActivity2.U0.setVisibility(8);
                                        if (imageTrackingActivity2.y.getChildren().size() > 0) {
                                            Node node2 = imageTrackingActivity2.A.j;
                                            if (node2 != null) {
                                                node2.setEnabled(true);
                                                imageTrackingActivity2.A.k();
                                                return;
                                            }
                                            imageTrackingActivity2.y.setEnabled(true);
                                            imageTrackingActivity2.A.v();
                                        }
                                    }
                                });
                            }
                            imageTrackingActivity.e0 = true;
                            float[] transformationMatrixJNI = imageTrackingActivity.u0.getTransformationMatrixJNI();
                            float[] fArr = {transformationMatrixJNI[3], transformationMatrixJNI[7], transformationMatrixJNI[11]};
                            imageTrackingActivity.w.getScene().getCamera().setWorldPosition(new Vector3((-fArr[0]) * 1.0f, fArr[1] * 1.0f, fArr[2] * 1.0f));
                            float f12 = transformationMatrixJNI[0];
                            float f13 = transformationMatrixJNI[4];
                            float f14 = transformationMatrixJNI[8];
                            float f15 = transformationMatrixJNI[1];
                            float f16 = transformationMatrixJNI[5];
                            float f17 = transformationMatrixJNI[9];
                            float f18 = transformationMatrixJNI[2];
                            float f19 = transformationMatrixJNI[6];
                            float f20 = transformationMatrixJNI[10];
                            float f21 = f12 + f16 + f20;
                            if (f21 >= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                                float sqrt = (float) Math.sqrt(f21 + 1.0f);
                                f3 = sqrt * 0.5f;
                                float f22 = 0.5f / sqrt;
                                f4 = (f19 - f17) * f22;
                                f5 = (f14 - f18) * f22;
                                f2 = (f15 - f13) * f22;
                            } else if (f12 > f16 && f12 > f20) {
                                float sqrt2 = (float) Math.sqrt(((f12 + 1.0f) - f16) - f20);
                                float f23 = 0.5f / sqrt2;
                                float f24 = (f14 + f18) * f23;
                                float f25 = (f19 - f17) * f23;
                                f4 = sqrt2 * 0.5f;
                                f3 = f25;
                                f5 = (f15 + f13) * f23;
                                f2 = f24;
                            } else if (f16 > f20) {
                                float sqrt3 = (float) Math.sqrt(((f16 + 1.0f) - f12) - f20);
                                f6 = sqrt3 * 0.5f;
                                float f26 = 0.5f / sqrt3;
                                f7 = (f15 + f13) * f26;
                                f8 = (f14 - f18) * f26;
                                f9 = (f19 + f17) * f26;
                                Quaternion quaternion = new Quaternion(f7, f6, f9, f8);
                                Quaternion quaternion2 = new Quaternion(quaternion.x, -quaternion.y, -quaternion.z, -quaternion.w);
                                node = imageTrackingActivity.y;
                                if (node != null) {
                                    node.setLocalRotation(quaternion2);
                                }
                            } else {
                                float sqrt4 = (float) Math.sqrt(((f20 + 1.0f) - f12) - f16);
                                float f27 = 0.5f / sqrt4;
                                float f28 = (f19 + f17) * f27;
                                float f29 = (f15 - f13) * f27;
                                f2 = sqrt4 * 0.5f;
                                f3 = f29;
                                f4 = (f14 + f18) * f27;
                                f5 = f28;
                            }
                            f9 = f2;
                            f7 = f4;
                            float f30 = f5;
                            f8 = f3;
                            f6 = f30;
                            Quaternion quaternion3 = new Quaternion(f7, f6, f9, f8);
                            Quaternion quaternion22 = new Quaternion(quaternion3.x, -quaternion3.y, -quaternion3.z, -quaternion3.w);
                            node = imageTrackingActivity.y;
                            if (node != null) {
                            }
                        } else {
                            imageTrackingActivity.e0 = false;
                            if (System.currentTimeMillis() - imageTrackingActivity.V > 1000 && imageTrackingActivity.c0 && !imageTrackingActivity.Y) {
                                z = true;
                                imageTrackingActivity.Y = true;
                                imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.e1
                                    @Override // java.lang.Runnable
                                    public final void run() {
                                        ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                        imageTrackingActivity2.L(imageTrackingActivity2.getResources().getString(R.string.target_lost), 2000);
                                        imageTrackingActivity2.P0.setVisibility(0);
                                        imageTrackingActivity2.U0.setVisibility(0);
                                        if (imageTrackingActivity2.y.getChildren().size() > 0) {
                                            Node node2 = imageTrackingActivity2.A.j;
                                            if (node2 != null) {
                                                node2.setEnabled(false);
                                                return;
                                            }
                                            imageTrackingActivity2.y.setEnabled(false);
                                            hd hdVar = imageTrackingActivity2.A;
                                            if (hdVar != null) {
                                                hdVar.r();
                                            }
                                        }
                                    }
                                });
                                int i2 = z;
                                if (!imageTrackingActivity.a0) {
                                    if (imageTrackingActivity.j0) {
                                        imageTrackingActivity.J = new Mat(height, width, CvType.CV_8UC1, acquireLatestImage.getPlanes()[0].getBuffer(), rowStride);
                                        imageTrackingActivity.I = imageTrackingActivity.z(width, height);
                                    }
                                    if (imageTrackingActivity.M != 0 && !imageTrackingActivity.a0 && imageTrackingActivity.j0) {
                                        if (!imageTrackingActivity.Z && !imageTrackingActivity.d0) {
                                            imageTrackingActivity.Z = i2;
                                            Mat z2 = imageTrackingActivity.z(width, height);
                                            String q = c.b.a.a.a.q(imageTrackingActivity.getCacheDir().getAbsolutePath() + "/", "ImgUpload.jpg");
                                            Log.d("ImageTrackingActivity", "saveFrame " + q);
                                            Mat mat = new Mat(z2, new Rect((int) ((((float) z2.width()) * 0.15f) / 2.0f), ((int) (((float) z2.height()) * 0.15f)) / 2, (int) (((float) z2.width()) * 0.85f), (int) (((float) z2.height()) * 0.7f)));
                                            float f31 = (float) ac.f4547a.f4551e;
                                            if (Math.max(mat.width(), mat.height()) > f31) {
                                                if (mat.width() > mat.height()) {
                                                    Imgproc.resize(mat, mat, new org.opencv.core.Size(f31, (int) ((f31 / mat.width()) * mat.height())));
                                                } else {
                                                    Imgproc.resize(mat, mat, new org.opencv.core.Size((int) ((f31 / mat.height()) * mat.width()), f31));
                                                }
                                            }
                                            StringBuilder x2 = c.b.a.a.a.x("resized Img ");
                                            x2.append(mat.size());
                                            Log.d("ImageTrackingActivity", x2.toString());
                                            Imgcodecs.imwrite(q, mat);
                                            gc gcVar = new gc(imageTrackingActivity);
                                            ac acVar = ac.f4547a;
                                            bf bfVar = new bf(gcVar, acVar.f4548b, acVar.f4550d);
                                            imageTrackingActivity.x0 = bfVar;
                                            Object[] objArr = new Object[i2];
                                            objArr[0] = q;
                                            bfVar.execute(objArr);
                                        }
                                        Mat mat2 = imageTrackingActivity.I;
                                        if (System.currentTimeMillis() - imageTrackingActivity.U > 200) {
                                            Bitmap createBitmap = Bitmap.createBitmap(mat2.width(), mat2.height(), Bitmap.Config.ARGB_8888);
                                            Utils.matToBitmap(mat2, createBitmap);
                                            BarcodeDetector build = new BarcodeDetector.Builder(imageTrackingActivity.getApplicationContext()).setBarcodeFormats(272).build();
                                            if (!build.isOperational()) {
                                                Log.e("ImageTrackingActivity", "Could not set up the detector!");
                                            }
                                            SparseArray<Barcode> detect = build.detect(new Frame.Builder().setBitmap(createBitmap).build());
                                            if (detect.size() > 0) {
                                                str = detect.valueAt(0).rawValue;
                                                Log.d("ImageTrackingActivity", "QR Google " + str);
                                            } else {
                                                str = "";
                                            }
                                            createBitmap.recycle();
                                            Log.d("ImageTrackingActivity", "QR " + str);
                                            if (!str.isEmpty()) {
                                                if (str.toLowerCase().contains("unitear_app/")) {
                                                    imageTrackingActivity.N();
                                                    imageTrackingActivity.d0 = i2;
                                                    imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.n2
                                                        @Override // java.lang.Runnable
                                                        public final void run() {
                                                            ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                                            int i3 = ImageTrackingActivity.r;
                                                            imageTrackingActivity2.A();
                                                        }
                                                    });
                                                    bf bfVar2 = imageTrackingActivity.x0;
                                                    if (bfVar2 != null) {
                                                        bfVar2.cancel(i2);
                                                    }
                                                    String[] split = str.split("/");
                                                    final String str2 = split[split.length - 2];
                                                    Log.d("ImageTrackingActivity", "qr alphaId = " + str2);
                                                    imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.a2
                                                        @Override // java.lang.Runnable
                                                        public final void run() {
                                                            ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                                            String str3 = str2;
                                                            b.v.u.c.z(imageTrackingActivity2.D);
                                                            imageTrackingActivity2.m0 = true;
                                                            imageTrackingActivity2.A = new hd(str3, ac.f4547a.f4549c, imageTrackingActivity2.y, imageTrackingActivity2.v, imageTrackingActivity2, imageTrackingActivity2);
                                                            imageTrackingActivity2.P();
                                                        }
                                                    });
                                                } else if (str.toLowerCase().contains("unitear") && str.toLowerCase().contains("campaign=")) {
                                                    imageTrackingActivity.N();
                                                    imageTrackingActivity.d0 = i2;
                                                    imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.n2
                                                        @Override // java.lang.Runnable
                                                        public final void run() {
                                                            ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                                            int i3 = ImageTrackingActivity.r;
                                                            imageTrackingActivity2.A();
                                                        }
                                                    });
                                                    bf bfVar3 = imageTrackingActivity.x0;
                                                    if (bfVar3 != null) {
                                                        bfVar3.cancel(i2);
                                                    }
                                                    String[] split2 = str.split("campaign=");
                                                    final String str3 = split2[split2.length - i2];
                                                    Log.d("ImageTrackingActivity", "qr Unitear ground alphaId = " + str);
                                                    imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.z1
                                                        @Override // java.lang.Runnable
                                                        public final void run() {
                                                            Intent intent;
                                                            ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                                            String str4 = str3;
                                                            Objects.requireNonNull(imageTrackingActivity2);
                                                            Log.d("ImageTrackingActivity", "loadARContentOnGroundFromAlphaIdUnitearGround alphaId = " + str4);
                                                            b.v.u.c.z(imageTrackingActivity2.D);
                                                            imageTrackingActivity2.m0 = true;
                                                            if (imageTrackingActivity2.i0) {
                                                                intent = new Intent(imageTrackingActivity2, NonARCoreActivitySceneform.class);
                                                            } else {
                                                                intent = new Intent(imageTrackingActivity2, ARCoreSceneformActivity.class);
                                                            }
                                                            intent.putExtra("groundContentId", str4);
                                                            intent.addFlags(536870912);
                                                            imageTrackingActivity2.startActivity(intent);
                                                        }
                                                    });
                                                } else {
                                                    imageTrackingActivity.runOnUiThread(new Runnable() { // from class: c.e.b.f1
                                                        @Override // java.lang.Runnable
                                                        public final void run() {
                                                            final ImageTrackingActivity imageTrackingActivity2 = ImageTrackingActivity.this;
                                                            String str4 = str;
                                                            imageTrackingActivity2.d0 = true;
                                                            imageTrackingActivity2.Z = false;
                                                            hd hdVar = new hd(str4, ac.f4547a.f4549c, imageTrackingActivity2.y, imageTrackingActivity2.v, imageTrackingActivity2, imageTrackingActivity2);
                                                            imageTrackingActivity2.A = hdVar;
                                                            hd.g gVar = new hd.g() { // from class: c.e.b.e2
                                                                @Override // c.e.b.hd.g
                                                                public final void a(final String str5) {
                                                                    final ImageTrackingActivity imageTrackingActivity3 = ImageTrackingActivity.this;
                                                                    Objects.requireNonNull(imageTrackingActivity3);
                                                                    if (str5.isEmpty()) {
                                                                        return;
                                                                    }
                                                                    imageTrackingActivity3.runOnUiThread(new Runnable() { // from class: c.e.b.v0
                                                                        @Override // java.lang.Runnable
                                                                        public final void run() {
                                                                            ImageTrackingActivity.this.L(str5, 5000);
                                                                        }
                                                                    });
                                                                }
                                                            };
                                                            c1 c1Var = new c1(imageTrackingActivity2);
                                                            hdVar.k = gVar;
                                                            String str5 = !hdVar.u ? "1" : CrashlyticsReportDataCapture.SIGNAL_DEFAULT;
                                                            hdVar.i();
                                                            hdVar.f4817h = new cc();
                                                            String str6 = hdVar.z + "app/get-ar-new-content/" + hdVar.y + "/" + hdVar.A + "/1/" + str5 + "/1/" + hdVar.C;
                                                            Log.d("LoaderARContent", str6);
                                                            hdVar.f4817h.a(str6, new jd(hdVar, gVar, c1Var));
                                                            hdVar.q = System.currentTimeMillis();
                                                            imageTrackingActivity2.G();
                                                            imageTrackingActivity2.O(str4);
                                                            imageTrackingActivity2.N();
                                                        }
                                                    });
                                                    bf bfVar4 = imageTrackingActivity.x0;
                                                    if (bfVar4 != null) {
                                                        bfVar4.cancel(i2);
                                                    }
                                                }
                                            }
                                            imageTrackingActivity.U = System.currentTimeMillis();
                                        }
                                    }
                                }
                            }
                        }
                        z = true;
                        int i22 = z;
                        if (!imageTrackingActivity.a0) {
                        }
                    }
                }
                acquireLatestImage.close();
            }
        }
    }

    /* compiled from: CamPreviewHelper.java */
    /* loaded from: classes2.dex */
    public static class d implements Comparator<Size> {
        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object, java.lang.Object] */
        @Override // java.util.Comparator
        public int compare(Size size, Size size2) {
            Size size3 = size;
            Size size4 = size2;
            return Long.signum((size3.getWidth() * size3.getHeight()) - (size4.getWidth() * size4.getHeight()));
        }
    }

    /* compiled from: CamPreviewHelper.java */
    /* loaded from: classes2.dex */
    public interface e {
    }

    static {
        SparseIntArray sparseIntArray = new SparseIntArray();
        f5441d = sparseIntArray;
        sparseIntArray.append(0, 90);
        sparseIntArray.append(1, 0);
        sparseIntArray.append(2, 270);
        sparseIntArray.append(3, BaseTransientBottomBar.ANIMATION_FADE_DURATION);
        f5442e = 640;
        f5443f = 480;
    }

    public yb(Activity activity, AutoFitTextureView autoFitTextureView, e eVar, boolean z) {
        this.w = false;
        a aVar = new a();
        this.z = aVar;
        this.A = new b();
        this.B = new c();
        this.f5444g = activity;
        this.i = autoFitTextureView;
        autoFitTextureView.setSurfaceTextureListener(aVar);
        this.y = eVar;
        this.w = z;
    }

    public static Size a(Size[] sizeArr, int i, int i2, int i3, int i4, Size size) {
        ArrayList arrayList = new ArrayList();
        ArrayList arrayList2 = new ArrayList();
        int width = size.getWidth();
        int height = size.getHeight();
        for (Size size2 : sizeArr) {
            if (size2.getWidth() <= i3 && size2.getHeight() <= i4 && size2.getHeight() != size2.getWidth()) {
                StringBuilder x = c.b.a.a.a.x("Available Preview Size ");
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
            return (Size) Collections.min(arrayList, new d());
        }
        if (arrayList2.size() > 0) {
            return (Size) Collections.max(arrayList2, new d());
        }
        Log.e("CCV2WithPreview", "Couldn't find any suitable preview size");
        return sizeArr[0];
    }

    public static Range<Integer> d(CameraCharacteristics cameraCharacteristics) {
        Range[] rangeArr = (Range[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        Range<Integer> range = null;
        if (rangeArr == null) {
            Log.e("CCV2WithPreview", "Failed to get FPS ranges.");
            return null;
        } else if (rangeArr.length == 0) {
            Log.e("CCV2WithPreview", "Failed to get FPS ranges.");
            return null;
        } else {
            for (Range range2 : rangeArr) {
                int intValue = ((Integer) range2.getLower()).intValue();
                int intValue2 = ((Integer) range2.getUpper()).intValue();
                if (intValue2 > 1000) {
                    Log.w("CCV2WithPreview", "Device uses FPS range in a 1000 scale. Normalizing. MaxFPS=" + intValue2 + " MinFPS=" + intValue);
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

    public void b() {
        try {
            try {
                this.q.acquire();
                CameraCaptureSession cameraCaptureSession = this.k;
                if (cameraCaptureSession != null) {
                    cameraCaptureSession.close();
                    this.k = null;
                }
                CameraDevice cameraDevice = this.l;
                if (cameraDevice != null) {
                    cameraDevice.close();
                    this.l = null;
                }
                ImageReader imageReader = this.n;
                if (imageReader != null) {
                    imageReader.close();
                    this.n = null;
                }
                HandlerThread handlerThread = this.v;
                if (handlerThread != null) {
                    handlerThread.quitSafely();
                    this.v.join();
                }
                this.v = null;
                this.u = null;
            } catch (InterruptedException e2) {
                throw new RuntimeException("Interrupted while trying to lock camera closing.", e2);
            }
        } finally {
            this.q.release();
        }
    }

    public final void c(int i, int i2) {
        if (this.i == null || this.m == null) {
            return;
        }
        int rotation = this.f5444g.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        float f2 = i;
        float f3 = i2;
        RectF rectF = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, f2, f3);
        RectF rectF2 = new RectF(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD, this.m.getHeight(), this.m.getWidth());
        float centerX = rectF.centerX();
        float centerY = rectF.centerY();
        if (1 == rotation || 3 == rotation) {
            rectF2.offset(centerX - rectF2.centerX(), centerY - rectF2.centerY());
            matrix.setRectToRect(rectF, rectF2, Matrix.ScaleToFit.FILL);
            float max = Math.max(f3 / this.m.getHeight(), f2 / this.m.getWidth());
            matrix.postScale(max, max, centerX, centerY);
            matrix.postRotate((rotation - 2) * 90, centerX, centerY);
        } else if (2 == rotation) {
            matrix.postRotate(180.0f, centerX, centerY);
        }
        this.i.setTransform(matrix);
    }

    /* JADX WARN: Code restructure failed: missing block: B:21:0x0067, code lost:
        r16 = (android.util.Size) java.util.Collections.max(java.util.Arrays.asList(r10.getOutputSizes(256)), new c.e.b.yb.d());
        r0 = r18.f5444g.getWindowManager().getDefaultDisplay().getRotation();
        r4 = ((java.lang.Integer) r9.get(android.hardware.camera2.CameraCharacteristics.SENSOR_ORIENTATION)).intValue();
        c.e.b.yb.f5438a = r4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:22:0x009a, code lost:
        r11 = true;
     */
    /* JADX WARN: Code restructure failed: missing block: B:23:0x009e, code lost:
        if (r0 == 0) goto L107;
     */
    /* JADX WARN: Code restructure failed: missing block: B:24:0x00a0, code lost:
        if (r0 == 1) goto L103;
     */
    /* JADX WARN: Code restructure failed: missing block: B:25:0x00a2, code lost:
        if (r0 == 2) goto L107;
     */
    /* JADX WARN: Code restructure failed: missing block: B:27:0x00a5, code lost:
        if (r0 == 3) goto L103;
     */
    /* JADX WARN: Code restructure failed: missing block: B:28:0x00a7, code lost:
        android.util.Log.e("CCV2WithPreview", "Display rotation is invalid: " + r0);
     */
    /* JADX WARN: Code restructure failed: missing block: B:29:0x00bc, code lost:
        if (r4 == 0) goto L33;
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x00c0, code lost:
        if (r4 != 180) goto L32;
     */
    /* JADX WARN: Code restructure failed: missing block: B:34:0x00c5, code lost:
        if (r4 == 90) goto L33;
     */
    /* JADX WARN: Code restructure failed: missing block: B:36:0x00c9, code lost:
        if (r4 != 270) goto L32;
     */
    /* JADX WARN: Code restructure failed: missing block: B:38:0x00cc, code lost:
        r11 = false;
     */
    /* JADX WARN: Code restructure failed: missing block: B:39:0x00cd, code lost:
        r0 = new android.graphics.Point();
        r18.f5444g.getWindowManager().getDefaultDisplay().getSize(r0);
        r4 = r0.x;
        r0 = r0.y;
     */
    /* JADX WARN: Code restructure failed: missing block: B:40:0x00e3, code lost:
        if (r11 == false) goto L102;
     */
    /* JADX WARN: Code restructure failed: missing block: B:41:0x00e5, code lost:
        r13 = r19;
        r12 = r20;
        r4 = r0;
        r0 = r4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:42:0x00ef, code lost:
        r12 = r19;
        r13 = r20;
     */
    /* JADX WARN: Code restructure failed: missing block: B:43:0x00f3, code lost:
        r11 = c.e.b.yb.f5442e;
     */
    /* JADX WARN: Code restructure failed: missing block: B:44:0x00f5, code lost:
        if (r4 <= r11) goto L101;
     */
    /* JADX WARN: Code restructure failed: missing block: B:45:0x00f7, code lost:
        r14 = r11;
     */
    /* JADX WARN: Code restructure failed: missing block: B:46:0x00f9, code lost:
        r14 = r4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:47:0x00fa, code lost:
        r4 = c.e.b.yb.f5443f;
     */
    /* JADX WARN: Code restructure failed: missing block: B:48:0x00fc, code lost:
        if (r0 <= r4) goto L100;
     */
    /* JADX WARN: Code restructure failed: missing block: B:49:0x00fe, code lost:
        r15 = r4;
     */
    /* JADX WARN: Code restructure failed: missing block: B:50:0x0100, code lost:
        r15 = r0;
     */
    /* JADX WARN: Code restructure failed: missing block: B:51:0x0101, code lost:
        r18.m = a(r10.getOutputSizes(android.graphics.SurfaceTexture.class), r12, r13, r14, r15, r16);
        android.util.Log.d("CamPreiviewHelper", "Preview Size " + r18.m.getWidth() + " x " + r18.m.getHeight());
        r0 = android.media.ImageReader.newInstance(r18.m.getWidth(), r18.m.getHeight(), 35, 2);
        r18.n = r0;
        r0.setOnImageAvailableListener(r18.B, r18.u);
     */
    /* JADX WARN: Code restructure failed: missing block: B:52:0x015e, code lost:
        if (r18.f5444g.getResources().getConfiguration().orientation != 2) goto L99;
     */
    /* JADX WARN: Code restructure failed: missing block: B:53:0x0160, code lost:
        r18.i.a(r18.m.getWidth(), r18.m.getHeight());
     */
    /* JADX WARN: Code restructure failed: missing block: B:54:0x0172, code lost:
        r18.i.a(r18.m.getHeight(), r18.m.getWidth());
     */
    /* JADX WARN: Code restructure failed: missing block: B:55:0x0183, code lost:
        r0 = (java.lang.Boolean) r9.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE);
     */
    /* JADX WARN: Code restructure failed: missing block: B:56:0x018b, code lost:
        if (r0 != null) goto L98;
     */
    /* JADX WARN: Code restructure failed: missing block: B:58:0x018e, code lost:
        r0.booleanValue();
     */
    /* JADX WARN: Code restructure failed: missing block: B:59:0x0191, code lost:
        r0 = (android.util.SizeF) r9.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
     */
    /* JADX WARN: Code restructure failed: missing block: B:60:0x01a1, code lost:
        if (((float[]) r9.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)) == null) goto L50;
     */
    /* JADX WARN: Code restructure failed: missing block: B:61:0x01a3, code lost:
        c.e.b.yb.f5439b = ((((float) java.lang.Math.atan(r0.getWidth() / (r4[0] * 2.0f))) * 2.0f) * 180.0f) / 3.14f;
        c.e.b.yb.f5440c = ((((float) java.lang.Math.atan(r0.getHeight() / (r4[0] * 2.0f))) * 2.0f) * 180.0f) / 3.14f;
     */
    /* JADX WARN: Code restructure failed: missing block: B:63:0x01d1, code lost:
        r0 = move-exception;
     */
    /* JADX WARN: Code restructure failed: missing block: B:64:0x01d2, code lost:
        android.util.Log.e("CCV2WithPreview", r0.toString());
     */
    /* JADX WARN: Removed duplicated region for block: B:79:0x0257 A[Catch: NullPointerException -> 0x0275, CameraAccessException -> 0x027a, TryCatch #3 {NullPointerException -> 0x0275, blocks: (B:11:0x003a, B:13:0x0043, B:15:0x0053, B:20:0x0064, B:18:0x005a, B:21:0x0067, B:28:0x00a7, B:39:0x00cd, B:43:0x00f3, B:47:0x00fa, B:51:0x0101, B:53:0x0160, B:55:0x0183, B:59:0x0191, B:65:0x01d9, B:67:0x01e3, B:69:0x01e7, B:71:0x01ec, B:76:0x0217, B:77:0x021e, B:79:0x0257, B:81:0x025b, B:82:0x0260, B:84:0x026a, B:86:0x026d, B:87:0x0272, B:64:0x01d2, B:58:0x018e, B:54:0x0172), top: B:109:0x003a }] */
    /* JADX WARN: Removed duplicated region for block: B:84:0x026a A[Catch: NullPointerException -> 0x0275, CameraAccessException -> 0x027a, TryCatch #3 {NullPointerException -> 0x0275, blocks: (B:11:0x003a, B:13:0x0043, B:15:0x0053, B:20:0x0064, B:18:0x005a, B:21:0x0067, B:28:0x00a7, B:39:0x00cd, B:43:0x00f3, B:47:0x00fa, B:51:0x0101, B:53:0x0160, B:55:0x0183, B:59:0x0191, B:65:0x01d9, B:67:0x01e3, B:69:0x01e7, B:71:0x01ec, B:76:0x0217, B:77:0x021e, B:79:0x0257, B:81:0x025b, B:82:0x0260, B:84:0x026a, B:86:0x026d, B:87:0x0272, B:64:0x01d2, B:58:0x018e, B:54:0x0172), top: B:109:0x003a }] */
    /* JADX WARN: Removed duplicated region for block: B:96:0x0295 A[Catch: InterruptedException -> 0x02a7, CameraAccessException -> 0x02b0, TryCatch #6 {CameraAccessException -> 0x02b0, InterruptedException -> 0x02a7, blocks: (B:94:0x0289, B:96:0x0295, B:97:0x029f, B:98:0x02a6), top: B:111:0x0289 }] */
    /* JADX WARN: Removed duplicated region for block: B:97:0x029f A[Catch: InterruptedException -> 0x02a7, CameraAccessException -> 0x02b0, TryCatch #6 {CameraAccessException -> 0x02b0, InterruptedException -> 0x02a7, blocks: (B:94:0x0289, B:96:0x0295, B:97:0x029f, B:98:0x02a6), top: B:111:0x0289 }] */
    @SuppressLint({"MissingPermission"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void e(int i, int i2) {
        int i3;
        String str;
        CameraCharacteristics cameraCharacteristics;
        int[] iArr;
        int[] iArr2;
        if (b.j.c.a.a(this.f5444g, "android.permission.CAMERA") != 0) {
            return;
        }
        if (this.w && this.v == null) {
            HandlerThread handlerThread = new HandlerThread("Camera2Thread");
            this.v = handlerThread;
            handlerThread.start();
            this.u = new Handler(this.v.getLooper());
        }
        CameraManager cameraManager = (CameraManager) this.f5444g.getSystemService("camera");
        try {
            try {
                String[] cameraIdList = cameraManager.getCameraIdList();
                int length = cameraIdList.length;
                int i4 = 0;
                while (true) {
                    if (i4 >= length) {
                        break;
                    }
                    str = cameraIdList[i4];
                    cameraCharacteristics = cameraManager.getCameraCharacteristics(str);
                    Integer num = (Integer) cameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
                    if ((num == null || num.intValue() != 0) && (r10 = (StreamConfigurationMap) cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)) != null) {
                        break;
                    }
                    i4++;
                }
            } catch (CameraAccessException e2) {
                e2.printStackTrace();
            }
        } catch (NullPointerException e3) {
            e3.printStackTrace();
        }
        c(i, i2);
        CameraManager cameraManager2 = (CameraManager) this.f5444g.getSystemService("camera");
        try {
            if (this.q.tryAcquire(2500L, TimeUnit.MILLISECONDS)) {
                cameraManager2.openCamera(this.j, this.A, this.u);
                return;
            }
            throw new RuntimeException("Time out waiting to lock camera opening.");
        } catch (CameraAccessException e4) {
            e4.printStackTrace();
            return;
        } catch (InterruptedException e5) {
            throw new RuntimeException("Interrupted while trying to lock camera opening.", e5);
        }
        int[] iArr3 = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_SCENE_MODES);
        if (iArr3 != null) {
            for (int i5 : iArr3) {
                if (i5 == 5) {
                    this.f5445h = i5;
                    Log.d("CCV2WithPreview", "SceneModeNight " + this.f5445h);
                }
            }
        }
        try {
            this.s = ((Integer) cameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)).intValue();
        } catch (NullPointerException e6) {
            Log.e("JavaCamera2View", e6.toString());
        }
        Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.s);
        this.r = d(cameraCharacteristics);
        android.graphics.Rect rect = (android.graphics.Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        ((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue();
        iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
        if (iArr != null) {
            for (int i6 : iArr) {
            }
        }
        iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
        if (iArr2 != null) {
            for (int i7 : iArr2) {
            }
        }
        this.j = str;
        c(i, i2);
        CameraManager cameraManager22 = (CameraManager) this.f5444g.getSystemService("camera");
        if (this.q.tryAcquire(2500L, TimeUnit.MILLISECONDS)) {
        }
        Log.d("JavaCamera2View", "INFO_SUPPORTED_HARDWARE_LEVEL " + this.s);
        this.r = d(cameraCharacteristics);
        android.graphics.Rect rect2 = (android.graphics.Rect) cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        ((Integer) cameraCharacteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF)).intValue();
        iArr = (int[]) cameraCharacteristics.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES);
        if (iArr != null) {
        }
        iArr2 = (int[]) cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
        if (iArr2 != null) {
        }
        this.j = str;
        c(i, i2);
        CameraManager cameraManager222 = (CameraManager) this.f5444g.getSystemService("camera");
        if (this.q.tryAcquire(2500L, TimeUnit.MILLISECONDS)) {
        }
    }
}