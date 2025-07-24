package com.google.android.filament;

/* loaded from: classes.dex */
public class Skybox {
    private long mNativeObject;

    /* loaded from: classes.dex */
    public static class Builder {
        private final BuilderFinalizer mFinalizer;
        private final long mNativeBuilder;

        /* loaded from: classes.dex */
        public static class BuilderFinalizer {
            private final long mNativeObject;

            public BuilderFinalizer(long j) {
                this.mNativeObject = j;
            }

            public void finalize() {
                try {
                    super.finalize();
                } catch (Throwable unused) {
                }
                Skybox.nDestroyBuilder(this.mNativeObject);
            }
        }

        public Builder() {
            long nCreateBuilder = Skybox.nCreateBuilder();
            this.mNativeBuilder = nCreateBuilder;
            this.mFinalizer = new BuilderFinalizer(nCreateBuilder);
        }

        public Skybox build(Engine engine) {
            long nBuilderBuild = Skybox.nBuilderBuild(this.mNativeBuilder, engine.getNativeObject());
            if (nBuilderBuild != 0) {
                return new Skybox(nBuilderBuild);
            }
            throw new IllegalStateException("Couldn't create Skybox");
        }

        public Builder color(float f2, float f3, float f4, float f5) {
            Skybox.nBuilderColor(this.mNativeBuilder, f2, f3, f4, f5);
            return this;
        }

        public Builder environment(Texture texture) {
            Skybox.nBuilderEnvironment(this.mNativeBuilder, texture.getNativeObject());
            return this;
        }

        public Builder intensity(float f2) {
            Skybox.nBuilderIntensity(this.mNativeBuilder, f2);
            return this;
        }

        public Builder showSun(boolean z) {
            Skybox.nBuilderShowSun(this.mNativeBuilder, z);
            return this;
        }

        public Builder color(float[] fArr) {
            Skybox.nBuilderColor(this.mNativeBuilder, fArr[0], fArr[1], fArr[2], fArr[3]);
            return this;
        }
    }

    public Skybox(Engine engine, long j) {
        this.mNativeObject = j;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nBuilderBuild(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderColor(long j, float f2, float f3, float f4, float f5);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderEnvironment(long j, long j2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderIntensity(long j, float f2);

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nBuilderShowSun(long j, boolean z);

    /* JADX INFO: Access modifiers changed from: private */
    public static native long nCreateBuilder();

    /* JADX INFO: Access modifiers changed from: private */
    public static native void nDestroyBuilder(long j);

    private static native float nGetIntensity(long j);

    private static native int nGetLayerMask(long j);

    private static native void nSetColor(long j, float f2, float f3, float f4, float f5);

    private static native void nSetLayerMask(long j, int i, int i2);

    public void clearNativeObject() {
        this.mNativeObject = 0L;
    }

    public float getIntensity() {
        return nGetIntensity(getNativeObject());
    }

    public int getLayerMask() {
        return nGetLayerMask(getNativeObject());
    }

    public long getNativeObject() {
        long j = this.mNativeObject;
        if (j != 0) {
            return j;
        }
        throw new IllegalStateException("Calling method on destroyed Skybox");
    }

    public void setColor(float f2, float f3, float f4, float f5) {
        nSetColor(getNativeObject(), f2, f3, f4, f5);
    }

    public void setLayerMask(int i, int i2) {
        nSetLayerMask(getNativeObject(), i & 255, i2 & 255);
    }

    public void setColor(float[] fArr) {
        nSetColor(getNativeObject(), fArr[0], fArr[1], fArr[2], fArr[3]);
    }

    public Skybox(long j) {
        this.mNativeObject = j;
    }
}