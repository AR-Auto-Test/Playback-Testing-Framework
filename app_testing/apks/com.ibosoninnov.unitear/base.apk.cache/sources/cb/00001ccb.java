package com.google.ar.sceneform.rendering;

import com.google.android.filament.Engine;
import com.google.android.filament.NativeSurface;
import com.google.android.filament.SwapChain;
import com.google.ar.sceneform.utilities.Preconditions;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

/* loaded from: classes.dex */
public class HeadlessEngineWrapper extends FilamentEngineWrapper {
    public static final String TAG = "com.google.ar.sceneform.rendering.HeadlessEngineWrapper";
    private static final Constructor<Engine> engineInit;
    private static final Method getNativeEngineMethod;
    private static final Method getNativeSwapChainMethod;
    private static final Constructor<SwapChain> swapChainInit;
    public long nativeHandle;

    static {
        try {
            Method declaredMethod = SwapChain.class.getDeclaredMethod("getNativeObject", new Class[0]);
            getNativeSwapChainMethod = declaredMethod;
            Class cls = Long.TYPE;
            Constructor<SwapChain> declaredConstructor = SwapChain.class.getDeclaredConstructor(cls, Object.class);
            swapChainInit = declaredConstructor;
            Method declaredMethod2 = Engine.class.getDeclaredMethod("getNativeObject", new Class[0]);
            getNativeEngineMethod = declaredMethod2;
            Constructor<Engine> declaredConstructor2 = Engine.class.getDeclaredConstructor(cls);
            engineInit = declaredConstructor2;
            declaredMethod.setAccessible(true);
            declaredConstructor.setAccessible(true);
            declaredMethod2.setAccessible(true);
            declaredConstructor2.setAccessible(true);
        } catch (Exception e2) {
            throw new IllegalStateException("Couldn't get native getters", e2);
        }
    }

    public HeadlessEngineWrapper() {
        super(engineInit.newInstance(Long.valueOf(nCreateSwiftShaderEngine())));
    }

    private static native long nCreateSwiftShaderEngine();

    private static native long nCreateSwiftShaderSwapChain(long j, long j2);

    private static native void nDestroySwiftShaderEngine(long j);

    private static native void nDestroySwiftShaderSwapChain(long j, long j2);

    @Override // com.google.ar.sceneform.rendering.FilamentEngineWrapper, com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChain(Object obj) {
        try {
            return swapChainInit.newInstance(Long.valueOf(nCreateSwiftShaderSwapChain(((Long) Preconditions.checkNotNull((Long) getNativeEngineMethod.invoke(this.engine, new Object[0]))).longValue(), 0L)), null);
        } catch (ReflectiveOperationException e2) {
            throw new RuntimeException(e2);
        }
    }

    @Override // com.google.ar.sceneform.rendering.FilamentEngineWrapper, com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChainFromNativeSurface(NativeSurface nativeSurface, long j) {
        try {
            return swapChainInit.newInstance(Long.valueOf(nCreateSwiftShaderSwapChain(((Long) Preconditions.checkNotNull((Long) getNativeEngineMethod.invoke(this.engine, new Object[0]))).longValue(), j)), null);
        } catch (ReflectiveOperationException e2) {
            throw new RuntimeException(e2);
        }
    }

    @Override // com.google.ar.sceneform.rendering.FilamentEngineWrapper, com.google.ar.sceneform.rendering.IEngine
    public void destroy() {
        try {
            nDestroySwiftShaderEngine(((Long) Preconditions.checkNotNull((Long) getNativeEngineMethod.invoke(this.engine, new Object[0]))).longValue());
        } catch (ReflectiveOperationException e2) {
            throw new RuntimeException(e2);
        }
    }

    @Override // com.google.ar.sceneform.rendering.FilamentEngineWrapper, com.google.ar.sceneform.rendering.IEngine
    public void destroySwapChain(SwapChain swapChain) {
        try {
            nDestroySwiftShaderSwapChain(((Long) Preconditions.checkNotNull((Long) getNativeEngineMethod.invoke(this.engine, new Object[0]))).longValue(), ((Long) Preconditions.checkNotNull((Long) getNativeSwapChainMethod.invoke(swapChain, new Object[0]))).longValue());
        } catch (ReflectiveOperationException e2) {
            throw new RuntimeException(e2);
        }
    }

    @Override // com.google.ar.sceneform.rendering.FilamentEngineWrapper, com.google.ar.sceneform.rendering.IEngine
    public SwapChain createSwapChain(Object obj, long j) {
        try {
            return swapChainInit.newInstance(Long.valueOf(nCreateSwiftShaderSwapChain(((Long) Preconditions.checkNotNull((Long) getNativeEngineMethod.invoke(this.engine, new Object[0]))).longValue(), j)), null);
        } catch (ReflectiveOperationException e2) {
            throw new RuntimeException(e2);
        }
    }
}