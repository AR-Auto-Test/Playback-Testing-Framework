package com.google.mediapipe.components;

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.util.Size;
import javax.annotation.Nullable;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/CameraHelper.class */
public abstract class CameraHelper {
    protected static final String TAG = "CameraHelper";
    protected OnCameraStartedListener onCameraStartedListener;
    protected CameraFacing cameraFacing;

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/CameraHelper$CameraFacing.class */
    public enum CameraFacing {
        FRONT,
        BACK
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/components/CameraHelper$OnCameraStartedListener.class */
    public interface OnCameraStartedListener {
        void onCameraStarted(@Nullable SurfaceTexture surfaceTexture);
    }

    public abstract void startCamera(Activity context, CameraFacing cameraFacing, @Nullable SurfaceTexture surfaceTexture);

    public abstract Size computeDisplaySizeFromViewSize(Size viewSize);

    public abstract boolean isCameraRotated();

    public void setOnCameraStartedListener(@Nullable OnCameraStartedListener listener) {
        this.onCameraStartedListener = listener;
    }
}