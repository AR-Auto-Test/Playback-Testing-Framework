package com.google.ar.sceneform.rendering;

import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES30;

/* loaded from: classes.dex */
public class GLHelper {
    private static final int EGL_OPENGL_ES3_BIT = 64;
    private static final String TAG = "GLHelper";

    public static int createCameraTexture() {
        int[] iArr = new int[1];
        GLES30.glGenTextures(1, iArr, 0);
        int i = iArr[0];
        GLES30.glBindTexture(36197, i);
        GLES30.glTexParameteri(36197, 10242, 33071);
        GLES30.glTexParameteri(36197, 10243, 33071);
        GLES30.glTexParameteri(36197, 10241, 9728);
        GLES30.glTexParameteri(36197, 10240, 9728);
        return i;
    }

    public static void destroyContext(EGLContext eGLContext) {
        if (!EGL14.eglDestroyContext(EGL14.eglGetDisplay(0), eGLContext)) {
            throw new IllegalStateException("Error destroying GL context.");
        }
    }

    public static EGLContext makeContext() {
        return makeContext(EGL14.EGL_NO_CONTEXT);
    }

    public static EGLContext makeContext(EGLContext eGLContext) {
        EGLDisplay eglGetDisplay = EGL14.eglGetDisplay(0);
        EGL14.eglInitialize(eglGetDisplay, null, 0, null, 0);
        EGLConfig[] eGLConfigArr = new EGLConfig[1];
        EGL14.eglChooseConfig(eglGetDisplay, new int[]{12352, 64, 12344}, 0, eGLConfigArr, 0, 1, new int[]{0}, 0);
        EGLContext eglCreateContext = EGL14.eglCreateContext(eglGetDisplay, eGLConfigArr[0], eGLContext, new int[]{12440, 3, 12344}, 0);
        EGLSurface eglCreatePbufferSurface = EGL14.eglCreatePbufferSurface(eglGetDisplay, eGLConfigArr[0], new int[]{12375, 1, 12374, 1, 12344}, 0);
        if (EGL14.eglMakeCurrent(eglGetDisplay, eglCreatePbufferSurface, eglCreatePbufferSurface, eglCreateContext)) {
            return eglCreateContext;
        }
        throw new IllegalStateException("Error making GL context.");
    }
}