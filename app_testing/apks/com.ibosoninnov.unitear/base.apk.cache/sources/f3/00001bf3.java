package com.google.ar.core;

import android.hardware.camera2.CameraCaptureSession;
import android.os.Handler;

/* compiled from: SharedCamera.java */
/* loaded from: classes.dex */
public final class ai extends CameraCaptureSession.StateCallback {

    /* renamed from: d  reason: collision with root package name */
    public static final /* synthetic */ int f5567d = 0;

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ Handler f5568a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ CameraCaptureSession.StateCallback f5569b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ SharedCamera f5570c;

    public ai(SharedCamera sharedCamera, Handler handler, CameraCaptureSession.StateCallback stateCallback) {
        this.f5570c = sharedCamera;
        this.f5568a = handler;
        this.f5569b = stateCallback;
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public final void onActive(final CameraCaptureSession cameraCaptureSession) {
        Handler handler = this.f5568a;
        final CameraCaptureSession.StateCallback stateCallback = this.f5569b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$2$$ExternalSyntheticLambda0
            @Override // java.lang.Runnable
            public final void run() {
                CameraCaptureSession.StateCallback stateCallback2 = stateCallback;
                CameraCaptureSession cameraCaptureSession2 = cameraCaptureSession;
                int i = ai.f5567d;
                stateCallback2.onActive(cameraCaptureSession2);
            }
        });
        this.f5570c.onCaptureSessionActive(cameraCaptureSession);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public final void onClosed(final CameraCaptureSession cameraCaptureSession) {
        Handler handler = this.f5568a;
        final CameraCaptureSession.StateCallback stateCallback = this.f5569b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$2$$ExternalSyntheticLambda1
            @Override // java.lang.Runnable
            public final void run() {
                CameraCaptureSession.StateCallback stateCallback2 = stateCallback;
                CameraCaptureSession cameraCaptureSession2 = cameraCaptureSession;
                int i = ai.f5567d;
                stateCallback2.onClosed(cameraCaptureSession2);
            }
        });
        this.f5570c.onCaptureSessionClosed(cameraCaptureSession);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public final void onConfigureFailed(final CameraCaptureSession cameraCaptureSession) {
        Handler handler = this.f5568a;
        final CameraCaptureSession.StateCallback stateCallback = this.f5569b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$2$$ExternalSyntheticLambda2
            @Override // java.lang.Runnable
            public final void run() {
                CameraCaptureSession.StateCallback stateCallback2 = stateCallback;
                CameraCaptureSession cameraCaptureSession2 = cameraCaptureSession;
                int i = ai.f5567d;
                stateCallback2.onConfigureFailed(cameraCaptureSession2);
            }
        });
        this.f5570c.onCaptureSessionConfigureFailed(cameraCaptureSession);
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public final void onConfigured(final CameraCaptureSession cameraCaptureSession) {
        aj ajVar;
        aj unused;
        unused = this.f5570c.sharedCameraInfo;
        Handler handler = this.f5568a;
        final CameraCaptureSession.StateCallback stateCallback = this.f5569b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$2$$ExternalSyntheticLambda3
            @Override // java.lang.Runnable
            public final void run() {
                CameraCaptureSession.StateCallback stateCallback2 = stateCallback;
                CameraCaptureSession cameraCaptureSession2 = cameraCaptureSession;
                int i = ai.f5567d;
                stateCallback2.onConfigured(cameraCaptureSession2);
            }
        });
        this.f5570c.onCaptureSessionConfigured(cameraCaptureSession);
        ajVar = this.f5570c.sharedCameraInfo;
        if (ajVar.b() != null) {
            this.f5570c.setDummyListenerToAvoidImageBufferStarvation();
        }
    }

    @Override // android.hardware.camera2.CameraCaptureSession.StateCallback
    public final void onReady(final CameraCaptureSession cameraCaptureSession) {
        Handler handler = this.f5568a;
        final CameraCaptureSession.StateCallback stateCallback = this.f5569b;
        handler.post(new Runnable() { // from class: com.google.ar.core.SharedCamera$2$$ExternalSyntheticLambda4
            @Override // java.lang.Runnable
            public final void run() {
                CameraCaptureSession.StateCallback stateCallback2 = stateCallback;
                CameraCaptureSession cameraCaptureSession2 = cameraCaptureSession;
                int i = ai.f5567d;
                stateCallback2.onReady(cameraCaptureSession2);
            }
        });
        this.f5570c.onCaptureSessionReady(cameraCaptureSession);
    }
}