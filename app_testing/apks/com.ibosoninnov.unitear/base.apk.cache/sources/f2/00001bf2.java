package com.google.ar.core;

import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraDevice;
import android.os.Handler;
import android.view.Surface;

/* compiled from: SharedCamera.java */
/* loaded from: classes.dex */
public final class ah extends CameraDevice.StateCallback {

    /* renamed from: d  reason: collision with root package name */
    public static final /* synthetic */ int f5563d = 0;

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Handler f5564a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ CameraDevice.StateCallback f5565b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SharedCamera f5566c;

    public ah(SharedCamera sharedCamera, Handler handler, CameraDevice.StateCallback stateCallback) {
        this.f5566c = sharedCamera;
        this.f5564a = handler;
        this.f5565b = stateCallback;
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public final void onClosed(final CameraDevice cameraDevice) {
        Handler handler = this.f5564a;
        final CameraDevice.StateCallback stateCallback = this.f5565b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$1$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                CameraDevice.StateCallback stateCallback2 = stateCallback;
                CameraDevice cameraDevice2 = cameraDevice;
                int i = ah.f5563d;
                stateCallback2.onClosed(cameraDevice2);
            }
        });
        this.f5566c.onDeviceClosed(cameraDevice);
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public final void onDisconnected(final CameraDevice cameraDevice) {
        Handler handler = this.f5564a;
        final CameraDevice.StateCallback stateCallback = this.f5565b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$1$$ExternalSyntheticLambda1
            @Override // java.lang.Runnable
            public final void run() {
                CameraDevice.StateCallback stateCallback2 = stateCallback;
                CameraDevice cameraDevice2 = cameraDevice;
                int i = ah.f5563d;
                stateCallback2.onDisconnected(cameraDevice2);
            }
        });
        this.f5566c.onDeviceDisconnected(cameraDevice);
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public final void onError(final CameraDevice cameraDevice, final int i) {
        Handler handler = this.f5564a;
        final CameraDevice.StateCallback stateCallback = this.f5565b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$1$$ExternalSyntheticLambda3
            @Override // java.lang.Runnable
            public final void run() {
                CameraDevice.StateCallback stateCallback2 = stateCallback;
                CameraDevice cameraDevice2 = cameraDevice;
                int i2 = i;
                int i3 = ah.f5563d;
                stateCallback2.onError(cameraDevice2, i2);
            }
        });
        this.f5566c.close();
    }

    @Override // android.hardware.camera2.CameraDevice.StateCallback
    public final void onOpened(final CameraDevice cameraDevice) {
        aj ajVar;
        aj ajVar2;
        SurfaceTexture gpuSurfaceTexture;
        aj ajVar3;
        Surface gpuSurface;
        ajVar = this.f5566c.sharedCameraInfo;
        ajVar.d(cameraDevice);
        Handler handler = this.f5564a;
        final CameraDevice.StateCallback stateCallback = this.f5565b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$1$$ExternalSyntheticLambda2
            @Override // java.lang.Runnable
            public final void run() {
                CameraDevice.StateCallback stateCallback2 = stateCallback;
                CameraDevice cameraDevice2 = cameraDevice;
                int i = ah.f5563d;
                stateCallback2.onOpened(cameraDevice2);
            }
        });
        this.f5566c.onDeviceOpened(cameraDevice);
        ajVar2 = this.f5566c.sharedCameraInfo;
        gpuSurfaceTexture = this.f5566c.getGpuSurfaceTexture();
        ajVar2.g(gpuSurfaceTexture);
        ajVar3 = this.f5566c.sharedCameraInfo;
        gpuSurface = this.f5566c.getGpuSurface();
        ajVar3.f(gpuSurface);
    }
}