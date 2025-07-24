package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public class Config {
    private static final String TAG = "ARCore-Config";
    public long nativeHandle;
    public final long nativeSymbolTableHandle;
    public final Session session;

    /* loaded from: classes.dex */
    public enum AugmentedFaceMode {
        DISABLED(0),
        MESH3D(2);
        
        public final int nativeCode;

        AugmentedFaceMode(int i) {
            this.nativeCode = i;
        }

        public static AugmentedFaceMode forNumber(int i) {
            AugmentedFaceMode[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                AugmentedFaceMode augmentedFaceMode = values[i2];
                if (augmentedFaceMode.nativeCode == i) {
                    return augmentedFaceMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(64, "Unexpected value for native AugmentedFaceMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum CloudAnchorMode {
        DISABLED(0),
        ENABLED(1);
        
        public final int nativeCode;

        CloudAnchorMode(int i) {
            this.nativeCode = i;
        }

        public static CloudAnchorMode forNumber(int i) {
            CloudAnchorMode[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                CloudAnchorMode cloudAnchorMode = values[i2];
                if (cloudAnchorMode.nativeCode == i) {
                    return cloudAnchorMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(64, "Unexpected value for native AnchorHostingMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum DepthMode {
        DISABLED(0),
        AUTOMATIC(1),
        RAW_DEPTH_ONLY(3);
        
        public final int nativeCode;

        DepthMode(int i) {
            this.nativeCode = i;
        }

        public static DepthMode forNumber(int i) {
            DepthMode[] values = values();
            for (int i2 = 0; i2 < 3; i2++) {
                DepthMode depthMode = values[i2];
                if (depthMode.nativeCode == i) {
                    return depthMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(56, "Unexpected value for native DepthMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum FocusMode {
        FIXED(0),
        AUTO(1);
        
        public final int nativeCode;

        FocusMode(int i) {
            this.nativeCode = i;
        }

        public static FocusMode forNumber(int i) {
            FocusMode[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                FocusMode focusMode = values[i2];
                if (focusMode.nativeCode == i) {
                    return focusMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(56, "Unexpected value for native FocusMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum InstantPlacementMode {
        DISABLED(0),
        LOCAL_Y_UP(2);
        
        public final int nativeCode;

        InstantPlacementMode(int i) {
            this.nativeCode = i;
        }

        public static InstantPlacementMode forNumber(int i) {
            InstantPlacementMode[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                InstantPlacementMode instantPlacementMode = values[i2];
                if (instantPlacementMode.nativeCode == i) {
                    return instantPlacementMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(67, "Unexpected value for native InstantPlacementMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum LightEstimationMode {
        DISABLED(0),
        AMBIENT_INTENSITY(1),
        ENVIRONMENTAL_HDR(2);
        
        public final int nativeCode;

        LightEstimationMode(int i) {
            this.nativeCode = i;
        }

        public static LightEstimationMode forNumber(int i) {
            LightEstimationMode[] values = values();
            for (int i2 = 0; i2 < 3; i2++) {
                LightEstimationMode lightEstimationMode = values[i2];
                if (lightEstimationMode.nativeCode == i) {
                    return lightEstimationMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(66, "Unexpected value for native LightEstimationMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum PlaneFindingMode {
        DISABLED(0),
        HORIZONTAL(1),
        VERTICAL(2),
        HORIZONTAL_AND_VERTICAL(3);
        
        public final int nativeCode;

        PlaneFindingMode(int i) {
            this.nativeCode = i;
        }

        public static PlaneFindingMode forNumber(int i) {
            PlaneFindingMode[] values = values();
            for (int i2 = 0; i2 < 4; i2++) {
                PlaneFindingMode planeFindingMode = values[i2];
                if (planeFindingMode.nativeCode == i) {
                    return planeFindingMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(63, "Unexpected value for native PlaneFindingMode, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum UpdateMode {
        BLOCKING(0),
        LATEST_CAMERA_IMAGE(1);
        
        public final int nativeCode;

        UpdateMode(int i) {
            this.nativeCode = i;
        }

        public static UpdateMode forNumber(int i) {
            UpdateMode[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                UpdateMode updateMode = values[i2];
                if (updateMode.nativeCode == i) {
                    return updateMode;
                }
            }
            throw new FatalException(c.b.a.a.a.g(57, "Unexpected value for native UpdateMode, value=", i));
        }
    }

    public Config() {
        this.session = null;
        this.nativeHandle = 0L;
        this.nativeSymbolTableHandle = 0L;
    }

    public Config(Session session) {
        this.session = session;
        this.nativeHandle = nativeCreate(session.nativeWrapperHandle);
        this.nativeSymbolTableHandle = session.nativeSymbolTableHandle;
    }

    private static native long nativeCreate(long j);

    private static native void nativeDestroy(long j, long j2);

    private native int nativeGetAugmentedFaceMode(long j, long j2);

    private native long nativeGetAugmentedImageDatabase(long j, long j2);

    private native int nativeGetCloudAnchorMode(long j, long j2);

    private native int nativeGetDepthMode(long j, long j2);

    private native int nativeGetFocusMode(long j, long j2);

    private native int nativeGetInstantPlacementMode(long j, long j2);

    private native int nativeGetLightEstimationMode(long j, long j2);

    private native int nativeGetPlaneFindingMode(long j, long j2);

    private native int nativeGetUpdateMode(long j, long j2);

    private native void nativeSetAugmentedFaceMode(long j, long j2, int i);

    private native void nativeSetAugmentedImageDatabase(long j, long j2, long j3);

    private native void nativeSetCloudAnchorMode(long j, long j2, int i);

    private native void nativeSetDepthMode(long j, long j2, int i);

    private native void nativeSetFocusMode(long j, long j2, int i);

    private native void nativeSetInstantPlacementMode(long j, long j2, int i);

    private native void nativeSetLightEstimationMode(long j, long j2, int i);

    private native void nativeSetPlaneFindingMode(long j, long j2, int i);

    private native void nativeSetUpdateMode(long j, long j2, int i);

    public void finalize() {
        long j = this.nativeHandle;
        if (j != 0) {
            nativeDestroy(this.nativeSymbolTableHandle, j);
        }
        super.finalize();
    }

    public AugmentedFaceMode getAugmentedFaceMode() {
        return AugmentedFaceMode.forNumber(nativeGetAugmentedFaceMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public AugmentedImageDatabase getAugmentedImageDatabase() {
        return new AugmentedImageDatabase(this.session, nativeGetAugmentedImageDatabase(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public CloudAnchorMode getCloudAnchorMode() {
        return CloudAnchorMode.forNumber(nativeGetCloudAnchorMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public DepthMode getDepthMode() {
        return DepthMode.forNumber(nativeGetDepthMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public FocusMode getFocusMode() {
        return FocusMode.forNumber(nativeGetFocusMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public InstantPlacementMode getInstantPlacementMode() {
        return InstantPlacementMode.forNumber(nativeGetInstantPlacementMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public LightEstimationMode getLightEstimationMode() {
        return LightEstimationMode.forNumber(nativeGetLightEstimationMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public PlaneFindingMode getPlaneFindingMode() {
        return PlaneFindingMode.forNumber(nativeGetPlaneFindingMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public UpdateMode getUpdateMode() {
        return UpdateMode.forNumber(nativeGetUpdateMode(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public Config setAugmentedFaceMode(AugmentedFaceMode augmentedFaceMode) {
        nativeSetAugmentedFaceMode(this.session.nativeWrapperHandle, this.nativeHandle, augmentedFaceMode.nativeCode);
        return this;
    }

    public Config setAugmentedImageDatabase(AugmentedImageDatabase augmentedImageDatabase) {
        nativeSetAugmentedImageDatabase(this.session.nativeWrapperHandle, this.nativeHandle, augmentedImageDatabase == null ? 0L : augmentedImageDatabase.nativeHandle);
        return this;
    }

    public Config setCloudAnchorMode(CloudAnchorMode cloudAnchorMode) {
        nativeSetCloudAnchorMode(this.session.nativeWrapperHandle, this.nativeHandle, cloudAnchorMode.nativeCode);
        return this;
    }

    public Config setDepthMode(DepthMode depthMode) {
        nativeSetDepthMode(this.session.nativeWrapperHandle, this.nativeHandle, depthMode.nativeCode);
        return this;
    }

    public Config setFocusMode(FocusMode focusMode) {
        nativeSetFocusMode(this.session.nativeWrapperHandle, this.nativeHandle, focusMode.nativeCode);
        return this;
    }

    public Config setInstantPlacementMode(InstantPlacementMode instantPlacementMode) {
        nativeSetInstantPlacementMode(this.session.nativeWrapperHandle, this.nativeHandle, instantPlacementMode.nativeCode);
        return this;
    }

    public Config setLightEstimationMode(LightEstimationMode lightEstimationMode) {
        nativeSetLightEstimationMode(this.session.nativeWrapperHandle, this.nativeHandle, lightEstimationMode.nativeCode);
        return this;
    }

    public Config setPlaneFindingMode(PlaneFindingMode planeFindingMode) {
        nativeSetPlaneFindingMode(this.session.nativeWrapperHandle, this.nativeHandle, planeFindingMode.nativeCode);
        return this;
    }

    public Config setUpdateMode(UpdateMode updateMode) {
        nativeSetUpdateMode(this.session.nativeWrapperHandle, this.nativeHandle, updateMode.nativeCode);
        return this;
    }

    public Config(Session session, long j) {
        this.session = session;
        this.nativeHandle = j;
        this.nativeSymbolTableHandle = session.nativeSymbolTableHandle;
    }
}