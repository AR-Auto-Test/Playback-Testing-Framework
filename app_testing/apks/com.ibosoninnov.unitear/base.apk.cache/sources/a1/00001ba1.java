package com.google.ar.core;

import android.util.Range;
import android.util.Size;
import com.google.ar.core.exceptions.FatalException;
import java.util.EnumSet;

/* loaded from: classes.dex */
public class CameraConfig {
    public long nativeHandle;
    private final long nativeSymbolTableHandle;
    private final Session session;

    /* loaded from: classes.dex */
    public enum DepthSensorUsage {
        REQUIRE_AND_USE(1),
        DO_NOT_USE(2);
        
        public final int nativeCode;

        DepthSensorUsage(int i) {
            this.nativeCode = i;
        }

        public static EnumSet<DepthSensorUsage> forBitFlags(int i) {
            EnumSet<DepthSensorUsage> noneOf = EnumSet.noneOf(DepthSensorUsage.class);
            DepthSensorUsage[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                DepthSensorUsage depthSensorUsage = values[i2];
                if ((depthSensorUsage.nativeCode & i) != 0) {
                    noneOf.add(depthSensorUsage);
                }
            }
            return noneOf;
        }

        public static DepthSensorUsage forNumber(int i) {
            DepthSensorUsage[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                DepthSensorUsage depthSensorUsage = values[i2];
                if (depthSensorUsage.nativeCode == i) {
                    return depthSensorUsage;
                }
            }
            throw new FatalException(c.b.a.a.a.g(63, "Unexpected value for native DepthSensorUsage, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum FacingDirection {
        BACK(0),
        FRONT(1);
        
        public final int nativeCode;

        FacingDirection(int i) {
            this.nativeCode = i;
        }

        public static FacingDirection forNumber(int i) {
            FacingDirection[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                FacingDirection facingDirection = values[i2];
                if (facingDirection.nativeCode == i) {
                    return facingDirection;
                }
            }
            throw new FatalException(c.b.a.a.a.g(62, "Unexpected value for native FacingDirection, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum StereoCameraUsage {
        REQUIRE_AND_USE(1),
        DO_NOT_USE(2);
        
        public final int nativeCode;

        StereoCameraUsage(int i) {
            this.nativeCode = i;
        }

        public static EnumSet<StereoCameraUsage> forBitFlags(int i) {
            EnumSet<StereoCameraUsage> noneOf = EnumSet.noneOf(StereoCameraUsage.class);
            StereoCameraUsage[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                StereoCameraUsage stereoCameraUsage = values[i2];
                if ((stereoCameraUsage.nativeCode & i) != 0) {
                    noneOf.add(stereoCameraUsage);
                }
            }
            return noneOf;
        }

        public static StereoCameraUsage forNumber(int i) {
            StereoCameraUsage[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                StereoCameraUsage stereoCameraUsage = values[i2];
                if (stereoCameraUsage.nativeCode == i) {
                    return stereoCameraUsage;
                }
            }
            throw new FatalException(c.b.a.a.a.g(64, "Unexpected value for native StereoCameraUsage, value=", i));
        }
    }

    /* loaded from: classes.dex */
    public enum TargetFps {
        TARGET_FPS_30(1),
        TARGET_FPS_60(2);
        
        public final int nativeCode;

        TargetFps(int i) {
            this.nativeCode = i;
        }

        public static EnumSet<TargetFps> forBitFlags(int i) {
            EnumSet<TargetFps> noneOf = EnumSet.noneOf(TargetFps.class);
            TargetFps[] values = values();
            for (int i2 = 0; i2 < 2; i2++) {
                TargetFps targetFps = values[i2];
                if ((targetFps.nativeCode & i) != 0) {
                    noneOf.add(targetFps);
                }
            }
            return noneOf;
        }
    }

    public CameraConfig() {
        this.session = null;
        this.nativeHandle = 0L;
        this.nativeSymbolTableHandle = 0L;
    }

    public CameraConfig(Session session, long j) {
        this.session = session;
        this.nativeHandle = j;
        this.nativeSymbolTableHandle = session.nativeSymbolTableHandle;
    }

    private static native void nativeDestroyCameraConfig(long j, long j2);

    private native String nativeGetCameraId(long j, long j2);

    private native int nativeGetDepthSensorUsage(long j, long j2);

    private native int nativeGetFacingDirection(long j, long j2);

    private native void nativeGetFpsRange(long j, long j2, int[] iArr);

    private native void nativeGetImageDimensions(long j, long j2, int[] iArr);

    private native int nativeGetStereoCameraUsage(long j, long j2);

    private native void nativeGetTextureDimensions(long j, long j2, int[] iArr);

    public void finalize() {
        long j = this.nativeHandle;
        if (j != 0) {
            nativeDestroyCameraConfig(this.nativeSymbolTableHandle, j);
            this.nativeHandle = 0L;
        }
        super.finalize();
    }

    public String getCameraId() {
        return nativeGetCameraId(this.session.nativeWrapperHandle, this.nativeHandle);
    }

    public DepthSensorUsage getDepthSensorUsage() {
        return DepthSensorUsage.forNumber(nativeGetDepthSensorUsage(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public FacingDirection getFacingDirection() {
        return FacingDirection.forNumber(nativeGetFacingDirection(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public Range<Integer> getFpsRange() {
        int[] iArr = new int[2];
        nativeGetFpsRange(this.session.nativeWrapperHandle, this.nativeHandle, iArr);
        return new Range<>(Integer.valueOf(iArr[0]), Integer.valueOf(iArr[1]));
    }

    public Size getImageSize() {
        int[] iArr = new int[2];
        nativeGetImageDimensions(this.session.nativeWrapperHandle, this.nativeHandle, iArr);
        return new Size(iArr[0], iArr[1]);
    }

    public StereoCameraUsage getStereoCameraUsage() {
        return StereoCameraUsage.forNumber(nativeGetStereoCameraUsage(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public Size getTextureSize() {
        int[] iArr = new int[2];
        nativeGetTextureDimensions(this.session.nativeWrapperHandle, this.nativeHandle, iArr);
        return new Size(iArr[0], iArr[1]);
    }
}