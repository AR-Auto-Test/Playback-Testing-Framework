package com.google.mediapipe.glutil;

import android.opengl.GLES20;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLSurface;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/glutil/GlThread.class */
public class GlThread extends Thread {
    private static final String TAG = "GlThread";
    private static final String THREAD_NAME = "mediapipe.glutil.GlThread";
    private boolean doneStarting;
    private boolean startedSuccessfully;
    private final Object startLock;
    protected volatile EglManager eglManager;
    protected EGLSurface eglSurface;
    protected Handler handler;
    protected Looper looper;
    protected int framebuffer;

    public GlThread(@Nullable Object parentContext) {
        this(parentContext, null);
    }

    public GlThread(@Nullable Object parentContext, @Nullable int[] additionalConfigAttributes) {
        this.startLock = new Object();
        this.eglSurface = null;
        this.handler = null;
        this.looper = null;
        this.framebuffer = 0;
        this.eglManager = new EglManager(parentContext, additionalConfigAttributes);
        setName("mediapipe.glutil.GlThread");
    }

    public Handler getHandler() {
        return this.handler;
    }

    public Looper getLooper() {
        return this.looper;
    }

    public EglManager getEglManager() {
        return this.eglManager;
    }

    public EGLContext getEGLContext() {
        return this.eglManager.getContext();
    }

    public int getFramebuffer() {
        return this.framebuffer;
    }

    public void bindFramebuffer(int texture, int width, int height) {
        GLES20.glBindFramebuffer(36160, this.framebuffer);
        GLES20.glFramebufferTexture2D(36160, 36064, 3553, texture, 0);
        int status = GLES20.glCheckFramebufferStatus(36160);
        if (status != 36053) {
            throw new RuntimeException("Framebuffer not complete, status=" + status);
        }
        GLES20.glViewport(0, 0, width, height);
        ShaderUtil.checkGlError("glViewport");
    }

    /* JADX DEBUG: Finally have unexpected throw blocks count: 2, expect 1 */
    @Override // java.lang.Thread, java.lang.Runnable
    public void run() {
        try {
            Looper.prepare();
            this.handler = createHandler();
            this.looper = Looper.myLooper();
            Log.d("GlThread", String.format("Starting GL thread %s", getName()));
            prepareGl();
            this.startedSuccessfully = true;
            synchronized (this.startLock) {
                this.doneStarting = true;
                this.startLock.notify();
            }
            try {
                Looper.loop();
                this.looper = null;
                releaseGl();
                this.eglManager.release();
                Log.d("GlThread", String.format("Stopping GL thread %s", getName()));
            } catch (Throwable th) {
                this.looper = null;
                releaseGl();
                this.eglManager.release();
                Log.d("GlThread", String.format("Stopping GL thread %s", getName()));
                throw th;
            }
        } catch (Throwable th2) {
            synchronized (this.startLock) {
                this.doneStarting = true;
                this.startLock.notify();
                throw th2;
            }
        }
    }

    public boolean quitSafely() {
        if (this.looper == null) {
            return false;
        }
        this.looper.quitSafely();
        return true;
    }

    public boolean waitUntilReady() throws InterruptedException {
        synchronized (this.startLock) {
            while (!this.doneStarting) {
                this.startLock.wait();
            }
        }
        return this.startedSuccessfully;
    }

    public void prepareGl() {
        this.eglSurface = createEglSurface();
        this.eglManager.makeCurrent(this.eglSurface, this.eglSurface);
        GLES20.glDisable(2929);
        GLES20.glDisable(2884);
        int[] values = new int[1];
        GLES20.glGenFramebuffers(1, values, 0);
        this.framebuffer = values[0];
    }

    public void releaseGl() {
        if (this.framebuffer != 0) {
            int[] values = {this.framebuffer};
            GLES20.glDeleteFramebuffers(1, values, 0);
            this.framebuffer = 0;
        }
        this.eglManager.makeNothingCurrent();
        if (this.eglSurface != null) {
            this.eglManager.releaseSurface(this.eglSurface);
            this.eglSurface = null;
        }
    }

    protected Handler createHandler() {
        return new Handler();
    }

    protected EGLSurface createEglSurface() {
        return this.eglManager.createOffscreenSurface(1, 1);
    }
}