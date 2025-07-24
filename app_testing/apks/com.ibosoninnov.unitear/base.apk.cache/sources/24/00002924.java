package com.google.mediapipe.glutil;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.os.Build;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import com.google.mediapipe.framework.Compat;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/glutil/EglManager.class */
public class EglManager {
    private static final String TAG = "EglManager";
    public static final int EGL_CONTEXT_CLIENT_VERSION = 12440;
    public static final int EGL_OPENGL_ES2_BIT = 4;
    public static final int EGL_OPENGL_ES3_BIT_KHR = 64;
    public static final int EGL_DRAW = 12377;
    public static final int EGL_READ = 12378;
    public static final int EGL14_API_LEVEL = 17;
    private EGL10 egl;
    private EGLDisplay eglDisplay;
    private EGLConfig eglConfig;
    private EGLContext eglContext;
    private int[] singleIntArray;
    private int glVersion;
    private long nativeEglContext;
    private android.opengl.EGLContext egl14Context;

    public EglManager(@Nullable Object parentContext) {
        this(parentContext, null);
    }

    public EglManager(@Nullable Object parentContext, @Nullable int[] additionalConfigAttributes) {
        EGLContext realParentContext;
        this.eglDisplay = EGL10.EGL_NO_DISPLAY;
        this.eglConfig = null;
        this.eglContext = EGL10.EGL_NO_CONTEXT;
        this.nativeEglContext = 0L;
        this.egl14Context = null;
        this.singleIntArray = new int[1];
        this.egl = (EGL10) EGLContext.getEGL();
        this.eglDisplay = this.egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
        if (this.eglDisplay == EGL10.EGL_NO_DISPLAY) {
            throw new RuntimeException("eglGetDisplay failed");
        }
        int[] version = new int[2];
        if (!this.egl.eglInitialize(this.eglDisplay, version)) {
            throw new RuntimeException("eglInitialize failed");
        }
        if (parentContext == null) {
            realParentContext = EGL10.EGL_NO_CONTEXT;
        } else if (parentContext instanceof EGLContext) {
            realParentContext = (EGLContext) parentContext;
        } else if (Build.VERSION.SDK_INT >= 17 && (parentContext instanceof android.opengl.EGLContext)) {
            if (parentContext == EGL14.EGL_NO_CONTEXT) {
                realParentContext = EGL10.EGL_NO_CONTEXT;
            } else {
                realParentContext = egl10ContextFromEgl14Context((android.opengl.EGLContext) parentContext);
            }
        } else {
            throw new RuntimeException("invalid parent context: " + parentContext);
        }
        try {
            createContext(realParentContext, 3, additionalConfigAttributes);
            this.glVersion = 3;
        } catch (RuntimeException e2) {
            Log.w("EglManager", "could not create GLES 3 context: " + e2);
            createContext(realParentContext, 2, additionalConfigAttributes);
            this.glVersion = 2;
        }
    }

    public EGLContext getContext() {
        return this.eglContext;
    }

    public long getNativeContext() {
        if (this.nativeEglContext == 0) {
            grabContextVariants();
        }
        return this.nativeEglContext;
    }

    public android.opengl.EGLContext getEgl14Context() {
        if (Build.VERSION.SDK_INT < 17) {
            throw new RuntimeException("cannot use EGL14 on API level < 17");
        }
        if (this.egl14Context == null) {
            grabContextVariants();
        }
        return this.egl14Context;
    }

    public int getGlMajorVersion() {
        return this.glVersion;
    }

    public void makeCurrent(EGLSurface drawSurface, EGLSurface readSurface) {
        if (!this.egl.eglMakeCurrent(this.eglDisplay, drawSurface, readSurface, this.eglContext)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }
    }

    public void makeNothingCurrent() {
        if (!this.egl.eglMakeCurrent(this.eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }
    }

    public EGLSurface createWindowSurface(Object surface) {
        if (!(surface instanceof Surface) && !(surface instanceof SurfaceTexture) && !(surface instanceof SurfaceHolder) && !(surface instanceof SurfaceView)) {
            throw new RuntimeException("invalid surface: " + surface);
        }
        int[] surfaceAttribs = {12344};
        EGLSurface eglSurface = this.egl.eglCreateWindowSurface(this.eglDisplay, this.eglConfig, surface, surfaceAttribs);
        checkEglError("eglCreateWindowSurface");
        if (eglSurface == null) {
            throw new RuntimeException("surface was null");
        }
        return eglSurface;
    }

    public EGLSurface createOffscreenSurface(int width, int height) {
        int[] surfaceAttribs = {12375, width, 12374, height, 12344};
        EGLSurface eglSurface = this.egl.eglCreatePbufferSurface(this.eglDisplay, this.eglConfig, surfaceAttribs);
        checkEglError("eglCreatePbufferSurface");
        if (eglSurface == null) {
            throw new RuntimeException("surface was null");
        }
        return eglSurface;
    }

    public void release() {
        if (this.eglDisplay != EGL10.EGL_NO_DISPLAY) {
            this.egl.eglMakeCurrent(this.eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT);
            this.egl.eglDestroyContext(this.eglDisplay, this.eglContext);
            this.egl.eglTerminate(this.eglDisplay);
        }
        this.eglDisplay = EGL10.EGL_NO_DISPLAY;
        this.eglContext = EGL10.EGL_NO_CONTEXT;
        this.eglConfig = null;
    }

    public void releaseSurface(EGLSurface eglSurface) {
        this.egl.eglDestroySurface(this.eglDisplay, eglSurface);
    }

    private void createContext(EGLContext parentContext, int glVersion, @Nullable int[] additionalConfigAttributes) {
        String str;
        this.eglConfig = getConfig(glVersion, additionalConfigAttributes);
        if (this.eglConfig == null) {
            throw new RuntimeException("Unable to find a suitable EGLConfig");
        }
        int[] contextAttrs = {12440, glVersion, 12344};
        this.eglContext = this.egl.eglCreateContext(this.eglDisplay, this.eglConfig, parentContext, contextAttrs);
        if (this.eglContext == null || this.eglContext == EGL10.EGL_NO_CONTEXT) {
            int error = this.egl.eglGetError();
            StringBuilder append = new StringBuilder().append("Could not create GL context: EGL error: 0x").append(Integer.toHexString(error));
            if (error == 12294) {
                str = ": parent context uses a different version of OpenGL";
            } else {
                str = "";
            }
            throw new RuntimeException(append.append(str).toString());
        }
    }

    private void grabContextVariants() {
        EGLContext previousContext = this.egl.eglGetCurrentContext();
        EGLDisplay previousDisplay = this.egl.eglGetCurrentDisplay();
        EGLSurface previousDrawSurface = this.egl.eglGetCurrentSurface(12377);
        EGLSurface previousReadSurface = this.egl.eglGetCurrentSurface(12378);
        EGLSurface tempEglSurface = null;
        if (previousContext != this.eglContext) {
            tempEglSurface = createOffscreenSurface(1, 1);
            makeCurrent(tempEglSurface, tempEglSurface);
        }
        this.nativeEglContext = Compat.getCurrentNativeEGLContext();
        if (Build.VERSION.SDK_INT >= 17) {
            this.egl14Context = EGL14.eglGetCurrentContext();
        }
        if (previousContext != this.eglContext) {
            this.egl.eglMakeCurrent(previousDisplay, previousDrawSurface, previousReadSurface, previousContext);
            releaseSurface(tempEglSurface);
        }
    }

    private EGLContext egl10ContextFromEgl14Context(android.opengl.EGLContext context) {
        android.opengl.EGLContext previousContext = EGL14.eglGetCurrentContext();
        android.opengl.EGLDisplay previousDisplay = EGL14.eglGetCurrentDisplay();
        android.opengl.EGLSurface previousDrawSurface = EGL14.eglGetCurrentSurface(12377);
        android.opengl.EGLSurface previousReadSurface = EGL14.eglGetCurrentSurface(12378);
        android.opengl.EGLDisplay defaultDisplay = EGL14.eglGetDisplay(0);
        android.opengl.EGLSurface tempEglSurface = null;
        if (!previousContext.equals(context)) {
            int[] surfaceAttribs = {12375, 1, 12374, 1, 12344};
            android.opengl.EGLConfig tempConfig = getThrowawayConfig(defaultDisplay);
            tempEglSurface = EGL14.eglCreatePbufferSurface(previousDisplay, tempConfig, surfaceAttribs, 0);
            EGL14.eglMakeCurrent(defaultDisplay, tempEglSurface, tempEglSurface, context);
        }
        EGLContext egl10Context = this.egl.eglGetCurrentContext();
        if (!previousContext.equals(context)) {
            EGL14.eglMakeCurrent(previousDisplay, previousDrawSurface, previousReadSurface, previousContext);
            EGL14.eglDestroySurface(defaultDisplay, tempEglSurface);
        }
        return egl10Context;
    }

    private android.opengl.EGLConfig getThrowawayConfig(android.opengl.EGLDisplay display) {
        int[] attribList = {12339, 5, 12344};
        android.opengl.EGLConfig[] configs = new android.opengl.EGLConfig[1];
        int[] numConfigs = this.singleIntArray;
        if (!EGL14.eglChooseConfig(display, attribList, 0, configs, 0, 1, numConfigs, 0)) {
            throw new IllegalArgumentException("eglChooseConfig failed");
        }
        if (numConfigs[0] <= 0) {
            throw new IllegalArgumentException("No configs match requested attributes");
        }
        return configs[0];
    }

    /* JADX DEBUG: Multi-variable search result rejected for r0v27, resolved type: java.lang.Object[] */
    /* JADX DEBUG: Multi-variable search result rejected for r0v3, resolved type: int[] */
    /* JADX DEBUG: Multi-variable search result rejected for r0v31, resolved type: byte */
    /* JADX DEBUG: Multi-variable search result rejected for r0v33, resolved type: byte */
    /* JADX WARN: Multi-variable type inference failed */
    private int[] mergeAttribLists(int[] list1, @Nullable int[] list2) {
        int[] iArr;
        if (list2 == null) {
            return list1;
        }
        HashMap<Integer, Integer> attribMap = new HashMap<>();
        for (Object[] objArr : new int[]{list1, list2}) {
            for (int i = 0; i < objArr.length / 2; i++) {
                byte b2 = objArr[2 * i];
                byte b3 = objArr[(2 * i) + 1];
                if (b2 == 12344) {
                    break;
                }
                attribMap.put(Integer.valueOf(b2), Integer.valueOf(b3));
            }
        }
        int[] merged = new int[(attribMap.size() * 2) + 1];
        int i2 = 0;
        for (Map.Entry<Integer, Integer> e2 : attribMap.entrySet()) {
            int i3 = i2;
            int i4 = i2 + 1;
            merged[i3] = e2.getKey().intValue();
            i2 = i4 + 1;
            merged[i4] = e2.getValue().intValue();
        }
        merged[i2] = 12344;
        return merged;
    }

    private EGLConfig getConfig(int glVersion, @Nullable int[] additionalConfigAttributes) {
        int[] baseAttribList = new int[15];
        baseAttribList[0] = 12324;
        baseAttribList[1] = 8;
        baseAttribList[2] = 12323;
        baseAttribList[3] = 8;
        baseAttribList[4] = 12322;
        baseAttribList[5] = 8;
        baseAttribList[6] = 12321;
        baseAttribList[7] = 8;
        baseAttribList[8] = 12325;
        baseAttribList[9] = 16;
        baseAttribList[10] = 12352;
        baseAttribList[11] = glVersion == 3 ? 64 : 4;
        baseAttribList[12] = 12339;
        baseAttribList[13] = 5;
        baseAttribList[14] = 12344;
        int[] attribList = mergeAttribLists(baseAttribList, additionalConfigAttributes);
        int[] numConfigs = this.singleIntArray;
        if (!this.egl.eglChooseConfig(this.eglDisplay, attribList, null, 0, numConfigs)) {
            throw new IllegalArgumentException("eglChooseConfig failed");
        }
        if (numConfigs[0] <= 0) {
            throw new IllegalArgumentException("No configs match requested attributes");
        }
        EGLConfig[] configs = new EGLConfig[numConfigs[0]];
        if (!this.egl.eglChooseConfig(this.eglDisplay, attribList, configs, configs.length, numConfigs)) {
            throw new IllegalArgumentException("eglChooseConfig#2 failed");
        }
        EGLConfig bestConfig = null;
        int length = configs.length;
        int i = 0;
        while (true) {
            if (i >= length) {
                break;
            }
            EGLConfig config = configs[i];
            int r = findConfigAttrib(config, 12324, 0);
            int g2 = findConfigAttrib(config, 12323, 0);
            int b2 = findConfigAttrib(config, 12322, 0);
            int a2 = findConfigAttrib(config, 12321, 0);
            if (r != 8 || g2 != 8 || b2 != 8 || a2 != 8) {
                i++;
            } else {
                bestConfig = config;
                break;
            }
        }
        if (bestConfig == null) {
            bestConfig = configs[0];
        }
        return bestConfig;
    }

    private void checkEglError(String msg) {
        int error = this.egl.eglGetError();
        if (error != 12288) {
            throw new RuntimeException(msg + ": EGL error: 0x" + Integer.toHexString(error));
        }
    }

    private int findConfigAttrib(EGLConfig config, int attribute, int defaultValue) {
        if (this.egl.eglGetConfigAttrib(this.eglDisplay, config, attribute, this.singleIntArray)) {
            return this.singleIntArray[0];
        }
        return defaultValue;
    }
}