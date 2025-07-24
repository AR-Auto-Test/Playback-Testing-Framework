package com.google.ar.sceneform.rendering;

import android.opengl.EGLContext;
import com.google.android.filament.Engine;
import com.google.android.filament.Filament;
import com.google.android.filament.gltfio.Gltfio;
import com.google.ar.sceneform.utilities.Preconditions;

/* loaded from: classes.dex */
public class EngineInstance {
    private static IEngine engine = null;
    private static boolean filamentInitialized = false;
    private static EGLContext glContext = null;
    private static boolean headlessEngine = false;

    private static void createEngine() {
        if (engine == null) {
            if (!filamentInitialized) {
                try {
                    gltfioInit();
                } catch (UnsatisfiedLinkError unused) {
                }
            }
            if (!filamentInitialized) {
                try {
                    Filament.init();
                    filamentInitialized = true;
                } catch (UnsatisfiedLinkError e2) {
                    if (loadUnifiedJni()) {
                        filamentInitialized = true;
                    } else {
                        throw e2;
                    }
                }
            }
            FilamentEngineWrapper filamentEngineWrapper = new FilamentEngineWrapper(createFilamentEngine());
            engine = filamentEngineWrapper;
            if (filamentEngineWrapper == null) {
                throw new IllegalStateException("Filament Engine creation has failed.");
            }
        }
    }

    private static Engine createFilamentEngine() {
        Engine createSharedFilamentEngine = createSharedFilamentEngine();
        if (createSharedFilamentEngine == null) {
            EGLContext makeContext = GLHelper.makeContext();
            glContext = makeContext;
            return Engine.create(makeContext);
        }
        return createSharedFilamentEngine;
    }

    private static void createHeadlessEngine() {
        if (engine == null) {
            try {
                HeadlessEngineWrapper headlessEngineWrapper = new HeadlessEngineWrapper();
                engine = headlessEngineWrapper;
                if (headlessEngineWrapper == null) {
                    throw new IllegalStateException("Filament Engine creation has failed.");
                }
            } catch (ReflectiveOperationException e2) {
                throw new RuntimeException("Filament Engine creation failed due to reflection error", e2);
            }
        }
    }

    private static Engine createSharedFilamentEngine() {
        return null;
    }

    public static void destroyEngine() {
        destroyFilamentEngine();
    }

    private static void destroyFilamentEngine() {
        if (engine != null) {
            if (headlessEngine || !destroySharedFilamentEngine()) {
                EGLContext eGLContext = glContext;
                if (eGLContext != null) {
                    GLHelper.destroyContext(eGLContext);
                    glContext = null;
                }
                ((IEngine) Preconditions.checkNotNull(engine)).destroy();
            }
            engine = null;
        }
    }

    private static boolean destroySharedFilamentEngine() {
        return false;
    }

    public static void disableHeadlessEngine() {
        headlessEngine = false;
    }

    public static void enableHeadlessEngine() {
        headlessEngine = true;
    }

    public static IEngine getEngine() {
        if (!headlessEngine) {
            createEngine();
        } else {
            createHeadlessEngine();
        }
        IEngine iEngine = engine;
        if (iEngine != null) {
            return iEngine;
        }
        throw new IllegalStateException("Filament Engine creation has failed.");
    }

    private static void gltfioInit() {
        Gltfio.init();
        filamentInitialized = true;
    }

    public static boolean isEngineDestroyed() {
        return engine == null;
    }

    public static boolean isHeadlessMode() {
        return headlessEngine;
    }

    private static boolean loadUnifiedJni() {
        return false;
    }

    private static native Object nCreateEngine();

    private static native void nDestroyEngine();
}