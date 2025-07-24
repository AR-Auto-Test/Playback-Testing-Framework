package com.google.ar.core;

import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraDevice;
import android.view.Surface;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* compiled from: SharedCamera.java */
/* loaded from: classes.dex */
public final class aj {

    /* renamed from: a  reason: collision with root package name */
    private CameraDevice f5571a = null;

    /* renamed from: b  reason: collision with root package name */
    private final Map<String, List<Surface>> f5572b = new HashMap();

    /* renamed from: c  reason: collision with root package name */
    private SurfaceTexture f5573c = null;

    /* renamed from: d  reason: collision with root package name */
    private Surface f5574d = null;

    private aj() {
    }

    public final SurfaceTexture a() {
        return this.f5573c;
    }

    public final CameraDevice b() {
        return this.f5571a;
    }

    public final Surface c() {
        return this.f5574d;
    }

    public final void d(CameraDevice cameraDevice) {
        this.f5571a = cameraDevice;
    }

    public final void e(String str, List<Surface> list) {
        this.f5572b.put(str, list);
    }

    public final void f(Surface surface) {
        this.f5574d = surface;
    }

    public final void g(SurfaceTexture surfaceTexture) {
        this.f5573c = surfaceTexture;
    }

    public /* synthetic */ aj(byte[] bArr) {
    }
}