package com.google.android.gms.vision;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.SystemClock;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.WindowManager;
import androidx.annotation.RecentlyNonNull;
import c.b.a.a.a;
import com.google.android.gms.common.images.Size;
import com.google.android.gms.common.internal.Preconditions;
import com.google.android.gms.common.util.VisibleForTesting;
import com.google.android.gms.vision.Frame;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.snackbar.BaseTransientBottomBar;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import org.opencv.ml.DTrees;

/* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
/* loaded from: classes.dex */
public class CameraSource {
    @SuppressLint({"InlinedApi"})
    public static final int CAMERA_FACING_BACK = 0;
    @SuppressLint({"InlinedApi"})
    public static final int CAMERA_FACING_FRONT = 1;
    private Context zza;
    private final Object zzb;
    private Camera zzc;
    private int zzd;
    private int zze;
    private Size zzf;
    private float zzg;
    private int zzh;
    private int zzi;
    private boolean zzj;
    private String zzk;
    private SurfaceTexture zzl;
    private Thread zzm;
    private zza zzn;
    private final IdentityHashMap<byte[], ByteBuffer> zzo;

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class Builder {
        private final Detector<?> zza;
        private CameraSource zzb;

        public Builder(@RecentlyNonNull Context context, @RecentlyNonNull Detector<?> detector) {
            CameraSource cameraSource = new CameraSource();
            this.zzb = cameraSource;
            if (context == null) {
                throw new IllegalArgumentException("No context supplied.");
            }
            if (detector != null) {
                this.zza = detector;
                cameraSource.zza = context;
                return;
            }
            throw new IllegalArgumentException("No detector supplied.");
        }

        @RecentlyNonNull
        public CameraSource build() {
            CameraSource cameraSource = this.zzb;
            cameraSource.getClass();
            cameraSource.zzn = new zza(this.zza);
            return this.zzb;
        }

        @RecentlyNonNull
        public Builder setAutoFocusEnabled(boolean z) {
            this.zzb.zzj = z;
            return this;
        }

        @RecentlyNonNull
        public Builder setFacing(int i) {
            if (i == 0 || i == 1) {
                this.zzb.zzd = i;
                return this;
            }
            throw new IllegalArgumentException(a.g(27, "Invalid camera: ", i));
        }

        @RecentlyNonNull
        public Builder setFocusMode(@RecentlyNonNull String str) {
            if (!str.equals("continuous-video") && !str.equals("continuous-picture")) {
                Log.w("CameraSource", String.format("FocusMode %s is not supported for now.", str));
                this.zzb.zzk = null;
            } else {
                this.zzb.zzk = str;
            }
            return this;
        }

        @RecentlyNonNull
        public Builder setRequestedFps(float f2) {
            if (f2 <= StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD) {
                StringBuilder sb = new StringBuilder(28);
                sb.append("Invalid fps: ");
                sb.append(f2);
                throw new IllegalArgumentException(sb.toString());
            }
            this.zzb.zzg = f2;
            return this;
        }

        @RecentlyNonNull
        public Builder setRequestedPreviewSize(int i, int i2) {
            if (i <= 0 || i > 1000000 || i2 <= 0 || i2 > 1000000) {
                throw new IllegalArgumentException(a.h(45, "Invalid preview size: ", i, "x", i2));
            }
            this.zzb.zzh = i;
            this.zzb.zzi = i2;
            return this;
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public interface PictureCallback {
        void onPictureTaken(@RecentlyNonNull byte[] bArr);
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public interface ShutterCallback {
        void onShutter();
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public class zzb implements Camera.PreviewCallback {
        private zzb() {
        }

        @Override // android.hardware.Camera.PreviewCallback
        public final void onPreviewFrame(byte[] bArr, Camera camera) {
            CameraSource.this.zzn.zza(bArr, camera);
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public class zzc implements Camera.PictureCallback {
        private PictureCallback zza;

        private zzc() {
        }

        @Override // android.hardware.Camera.PictureCallback
        public final void onPictureTaken(byte[] bArr, Camera camera) {
            PictureCallback pictureCallback = this.zza;
            if (pictureCallback != null) {
                pictureCallback.onPictureTaken(bArr);
            }
            synchronized (CameraSource.this.zzb) {
                if (CameraSource.this.zzc != null) {
                    CameraSource.this.zzc.startPreview();
                }
            }
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public static class zzd implements Camera.ShutterCallback {
        private ShutterCallback zza;

        private zzd() {
        }

        @Override // android.hardware.Camera.ShutterCallback
        public final void onShutter() {
            ShutterCallback shutterCallback = this.zza;
            if (shutterCallback != null) {
                shutterCallback.onShutter();
            }
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    @VisibleForTesting
    /* loaded from: classes.dex */
    public static class zze {
        private Size zza;
        private Size zzb;

        public zze(Camera.Size size, Camera.Size size2) {
            this.zza = new Size(size.width, size.height);
            if (size2 != null) {
                this.zzb = new Size(size2.width, size2.height);
            }
        }

        public final Size zza() {
            return this.zza;
        }

        public final Size zzb() {
            return this.zzb;
        }
    }

    private CameraSource() {
        this.zzb = new Object();
        this.zzd = 0;
        this.zzg = 30.0f;
        this.zzh = 1024;
        this.zzi = DTrees.PREDICT_MASK;
        this.zzj = false;
        this.zzo = new IdentityHashMap<>();
    }

    /* JADX WARN: Removed duplicated region for block: B:58:0x01a1  */
    /* JADX WARN: Removed duplicated region for block: B:59:0x01ab  */
    /* JADX WARN: Removed duplicated region for block: B:62:0x01c1  */
    /* JADX WARN: Removed duplicated region for block: B:72:0x01fe  */
    /* JADX WARN: Removed duplicated region for block: B:73:0x0204  */
    @SuppressLint({"InlinedApi"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private final Camera zza() {
        int i;
        Camera.CameraInfo cameraInfo;
        int i2;
        int i3;
        int i4 = this.zzd;
        Camera.CameraInfo cameraInfo2 = new Camera.CameraInfo();
        int i5 = 0;
        while (true) {
            if (i5 >= Camera.getNumberOfCameras()) {
                i5 = -1;
                break;
            }
            Camera.getCameraInfo(i5, cameraInfo2);
            if (cameraInfo2.facing == i4) {
                break;
            }
            i5++;
        }
        if (i5 != -1) {
            Camera open = Camera.open(i5);
            int i6 = this.zzh;
            int i7 = this.zzi;
            Camera.Parameters parameters = open.getParameters();
            List<Camera.Size> supportedPreviewSizes = parameters.getSupportedPreviewSizes();
            List<Camera.Size> supportedPictureSizes = parameters.getSupportedPictureSizes();
            ArrayList arrayList = new ArrayList();
            for (Camera.Size size : supportedPreviewSizes) {
                float f2 = size.width / size.height;
                Iterator<Camera.Size> it = supportedPictureSizes.iterator();
                while (true) {
                    if (it.hasNext()) {
                        Camera.Size next = it.next();
                        if (Math.abs(f2 - (next.width / next.height)) < 0.01f) {
                            arrayList.add(new zze(size, next));
                            break;
                        }
                    }
                }
            }
            if (arrayList.size() == 0) {
                Log.w("CameraSource", "No preview sizes have a corresponding same-aspect-ratio picture size");
                for (Camera.Size size2 : supportedPreviewSizes) {
                    arrayList.add(new zze(size2, null));
                }
            }
            int size3 = arrayList.size();
            int i8 = Integer.MAX_VALUE;
            int i9 = 0;
            int i10 = Integer.MAX_VALUE;
            zze zzeVar = null;
            while (i9 < size3) {
                Object obj = arrayList.get(i9);
                i9++;
                zze zzeVar2 = (zze) obj;
                Size zza2 = zzeVar2.zza();
                int abs = Math.abs(zza2.getHeight() - i7) + Math.abs(zza2.getWidth() - i6);
                if (abs < i10) {
                    zzeVar = zzeVar2;
                    i10 = abs;
                }
            }
            zze zzeVar3 = (zze) Preconditions.checkNotNull(zzeVar);
            if (zzeVar3 != null) {
                Size zzb2 = zzeVar3.zzb();
                this.zzf = zzeVar3.zza();
                int i11 = (int) (this.zzg * 1000.0f);
                int[] iArr = null;
                for (int[] iArr2 : open.getParameters().getSupportedPreviewFpsRange()) {
                    int abs2 = Math.abs(i11 - iArr2[1]) + Math.abs(i11 - iArr2[0]);
                    if (abs2 < i8) {
                        iArr = iArr2;
                        i8 = abs2;
                    }
                }
                int[] iArr3 = (int[]) Preconditions.checkNotNull(iArr);
                if (iArr3 != null) {
                    Camera.Parameters parameters2 = open.getParameters();
                    if (zzb2 != null) {
                        parameters2.setPictureSize(zzb2.getWidth(), zzb2.getHeight());
                    }
                    parameters2.setPreviewSize(this.zzf.getWidth(), this.zzf.getHeight());
                    parameters2.setPreviewFpsRange(iArr3[0], iArr3[1]);
                    parameters2.setPreviewFormat(17);
                    int rotation = ((WindowManager) Preconditions.checkNotNull((WindowManager) this.zza.getSystemService("window"))).getDefaultDisplay().getRotation();
                    if (rotation != 0) {
                        if (rotation == 1) {
                            i = 90;
                        } else if (rotation == 2) {
                            i = BaseTransientBottomBar.ANIMATION_FADE_DURATION;
                        } else if (rotation != 3) {
                            StringBuilder sb = new StringBuilder(31);
                            sb.append("Bad rotation value: ");
                            sb.append(rotation);
                            Log.e("CameraSource", sb.toString());
                        } else {
                            i = 270;
                        }
                        cameraInfo = new Camera.CameraInfo();
                        Camera.getCameraInfo(i5, cameraInfo);
                        if (cameraInfo.facing != 1) {
                            i2 = (cameraInfo.orientation + i) % 360;
                            i3 = (360 - i2) % 360;
                        } else {
                            i2 = ((cameraInfo.orientation - i) + 360) % 360;
                            i3 = i2;
                        }
                        this.zze = i2 / 90;
                        open.setDisplayOrientation(i3);
                        parameters2.setRotation(i2);
                        if (this.zzk != null) {
                            if (parameters2.getSupportedFocusModes().contains(this.zzk)) {
                                parameters2.setFocusMode((String) Preconditions.checkNotNull(this.zzk));
                            } else {
                                Log.w("CameraSource", String.format("FocusMode %s is not supported on this device.", this.zzk));
                                this.zzk = null;
                            }
                        }
                        if (this.zzk == null && this.zzj) {
                            if (!parameters2.getSupportedFocusModes().contains("continuous-video")) {
                                parameters2.setFocusMode("continuous-video");
                                this.zzk = "continuous-video";
                            } else {
                                Log.i("CameraSource", "Camera auto focus is not supported on this device.");
                            }
                        }
                        open.setParameters(parameters2);
                        open.setPreviewCallbackWithBuffer(new zzb());
                        open.addCallbackBuffer(zza(this.zzf));
                        open.addCallbackBuffer(zza(this.zzf));
                        open.addCallbackBuffer(zza(this.zzf));
                        open.addCallbackBuffer(zza(this.zzf));
                        return open;
                    }
                    i = 0;
                    cameraInfo = new Camera.CameraInfo();
                    Camera.getCameraInfo(i5, cameraInfo);
                    if (cameraInfo.facing != 1) {
                    }
                    this.zze = i2 / 90;
                    open.setDisplayOrientation(i3);
                    parameters2.setRotation(i2);
                    if (this.zzk != null) {
                    }
                    if (this.zzk == null) {
                        if (!parameters2.getSupportedFocusModes().contains("continuous-video")) {
                        }
                    }
                    open.setParameters(parameters2);
                    open.setPreviewCallbackWithBuffer(new zzb());
                    open.addCallbackBuffer(zza(this.zzf));
                    open.addCallbackBuffer(zza(this.zzf));
                    open.addCallbackBuffer(zza(this.zzf));
                    open.addCallbackBuffer(zza(this.zzf));
                    return open;
                }
                throw new IOException("Could not find suitable preview frames per second range.");
            }
            throw new IOException("Could not find suitable preview size.");
        }
        throw new IOException("Could not find requested camera.");
    }

    public int getCameraFacing() {
        return this.zzd;
    }

    @RecentlyNonNull
    public Size getPreviewSize() {
        return this.zzf;
    }

    public void release() {
        synchronized (this.zzb) {
            stop();
            this.zzn.zza();
        }
    }

    @RecentlyNonNull
    public CameraSource start() {
        synchronized (this.zzb) {
            if (this.zzc != null) {
                return this;
            }
            this.zzc = zza();
            SurfaceTexture surfaceTexture = new SurfaceTexture(100);
            this.zzl = surfaceTexture;
            this.zzc.setPreviewTexture(surfaceTexture);
            this.zzc.startPreview();
            Thread thread = new Thread(this.zzn);
            this.zzm = thread;
            thread.setName("gms.vision.CameraSource");
            this.zzn.zza(true);
            Thread thread2 = this.zzm;
            if (thread2 != null) {
                thread2.start();
            }
            return this;
        }
    }

    public void stop() {
        synchronized (this.zzb) {
            this.zzn.zza(false);
            Thread thread = this.zzm;
            if (thread != null) {
                try {
                    thread.join();
                } catch (InterruptedException unused) {
                    Log.d("CameraSource", "Frame processing thread interrupted on release.");
                }
                this.zzm = null;
            }
            Camera camera = this.zzc;
            if (camera != null) {
                camera.stopPreview();
                this.zzc.setPreviewCallbackWithBuffer(null);
                try {
                    this.zzc.setPreviewTexture(null);
                    this.zzl = null;
                    this.zzc.setPreviewDisplay(null);
                } catch (Exception e2) {
                    String valueOf = String.valueOf(e2);
                    StringBuilder sb = new StringBuilder(valueOf.length() + 32);
                    sb.append("Failed to clear camera preview: ");
                    sb.append(valueOf);
                    Log.e("CameraSource", sb.toString());
                }
                ((Camera) Preconditions.checkNotNull(this.zzc)).release();
                this.zzc = null;
            }
            this.zzo.clear();
        }
    }

    public void takePicture(ShutterCallback shutterCallback, PictureCallback pictureCallback) {
        synchronized (this.zzb) {
            if (this.zzc != null) {
                zzd zzdVar = new zzd();
                zzdVar.zza = shutterCallback;
                zzc zzcVar = new zzc();
                zzcVar.zza = pictureCallback;
                this.zzc.takePicture(zzdVar, null, null, zzcVar);
            }
        }
    }

    /* compiled from: com.google.android.gms:play-services-vision-common@@19.1.3 */
    /* loaded from: classes.dex */
    public class zza implements Runnable {
        private Detector<?> zza;
        private long zze;
        private ByteBuffer zzg;
        private long zzb = SystemClock.elapsedRealtime();
        private final Object zzc = new Object();
        private boolean zzd = true;
        private int zzf = 0;

        public zza(Detector<?> detector) {
            this.zza = detector;
        }

        @Override // java.lang.Runnable
        @SuppressLint({"InlinedApi"})
        public final void run() {
            boolean z;
            Frame build;
            ByteBuffer byteBuffer;
            while (true) {
                synchronized (this.zzc) {
                    while (true) {
                        z = this.zzd;
                        if (!z || this.zzg != null) {
                            break;
                        }
                        try {
                            this.zzc.wait();
                        } catch (InterruptedException e2) {
                            Log.d("CameraSource", "Frame processing loop terminated.", e2);
                            return;
                        }
                    }
                    if (!z) {
                        return;
                    }
                    build = new Frame.Builder().setImageData((ByteBuffer) Preconditions.checkNotNull(this.zzg), CameraSource.this.zzf.getWidth(), CameraSource.this.zzf.getHeight(), 17).setId(this.zzf).setTimestampMillis(this.zze).setRotation(CameraSource.this.zze).build();
                    byteBuffer = this.zzg;
                    this.zzg = null;
                }
                try {
                    ((Detector) Preconditions.checkNotNull(this.zza)).receiveFrame(build);
                } catch (Exception e3) {
                    Log.e("CameraSource", "Exception thrown from receiver.", e3);
                } finally {
                    ((Camera) Preconditions.checkNotNull(CameraSource.this.zzc)).addCallbackBuffer(((ByteBuffer) Preconditions.checkNotNull(byteBuffer)).array());
                }
            }
        }

        @SuppressLint({"Assert"})
        public final void zza() {
            Detector<?> detector = this.zza;
            if (detector != null) {
                detector.release();
                this.zza = null;
            }
        }

        public final void zza(boolean z) {
            synchronized (this.zzc) {
                this.zzd = z;
                this.zzc.notifyAll();
            }
        }

        public final void zza(byte[] bArr, Camera camera) {
            synchronized (this.zzc) {
                ByteBuffer byteBuffer = this.zzg;
                if (byteBuffer != null) {
                    camera.addCallbackBuffer(byteBuffer.array());
                    this.zzg = null;
                }
                if (!CameraSource.this.zzo.containsKey(bArr)) {
                    Log.d("CameraSource", "Skipping frame. Could not find ByteBuffer associated with the image data from the camera.");
                    return;
                }
                this.zze = SystemClock.elapsedRealtime() - this.zzb;
                this.zzf++;
                this.zzg = (ByteBuffer) CameraSource.this.zzo.get(bArr);
                this.zzc.notifyAll();
            }
        }
    }

    @RecentlyNonNull
    public CameraSource start(@RecentlyNonNull SurfaceHolder surfaceHolder) {
        synchronized (this.zzb) {
            if (this.zzc != null) {
                return this;
            }
            Camera zza2 = zza();
            this.zzc = zza2;
            zza2.setPreviewDisplay(surfaceHolder);
            this.zzc.startPreview();
            this.zzm = new Thread(this.zzn);
            this.zzn.zza(true);
            Thread thread = this.zzm;
            if (thread != null) {
                thread.start();
            }
            return this;
        }
    }

    @SuppressLint({"InlinedApi"})
    private final byte[] zza(Size size) {
        byte[] bArr = new byte[((int) Math.ceil(((size.getHeight() * size.getWidth()) * ImageFormat.getBitsPerPixel(17)) / 8.0d)) + 1];
        ByteBuffer wrap = ByteBuffer.wrap(bArr);
        if (wrap.hasArray() && wrap.array() == bArr) {
            this.zzo.put(bArr, wrap);
            return bArr;
        }
        throw new IllegalStateException("Failed to create valid buffer for camera source.");
    }
}