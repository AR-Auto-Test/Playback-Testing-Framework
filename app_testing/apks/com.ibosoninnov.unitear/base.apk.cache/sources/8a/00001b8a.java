package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;

/* loaded from: classes.dex */
public class Anchor {
    public long nativeHandle;
    public final long nativeSymbolTableHandle;
    private final Session session;

    /* loaded from: classes.dex */
    public enum CloudAnchorState {
        NONE(0),
        TASK_IN_PROGRESS(1),
        SUCCESS(2),
        ERROR_INTERNAL(-1),
        ERROR_NOT_AUTHORIZED(-2),
        ERROR_SERVICE_UNAVAILABLE(-3),
        ERROR_RESOURCE_EXHAUSTED(-4),
        ERROR_HOSTING_DATASET_PROCESSING_FAILED(-5),
        ERROR_CLOUD_ID_NOT_FOUND(-6),
        ERROR_RESOLVING_LOCALIZATION_NO_MATCH(-7),
        ERROR_RESOLVING_SDK_VERSION_TOO_OLD(-8),
        ERROR_RESOLVING_SDK_VERSION_TOO_NEW(-9),
        ERROR_HOSTING_SERVICE_UNAVAILABLE(-10);
        
        public final int nativeCode;

        CloudAnchorState(int i) {
            this.nativeCode = i;
        }

        public static CloudAnchorState forNumber(int i) {
            CloudAnchorState[] values = values();
            for (int i2 = 0; i2 < 13; i2++) {
                CloudAnchorState cloudAnchorState = values[i2];
                if (cloudAnchorState.nativeCode == i) {
                    return cloudAnchorState;
                }
            }
            throw new FatalException(c.b.a.a.a.g(63, "Unexpected value for native CloudAnchorState, value=", i));
        }

        public boolean isError() {
            return this.nativeCode < 0;
        }
    }

    public Anchor() {
        this.session = null;
        this.nativeHandle = 0L;
        this.nativeSymbolTableHandle = 0L;
    }

    public Anchor(long j, Session session) {
        this.session = session;
        this.nativeHandle = j;
        this.nativeSymbolTableHandle = session.nativeSymbolTableHandle;
    }

    private native void nativeDetach(long j, long j2);

    private native String nativeGetCloudAnchorId(long j, long j2);

    private native int nativeGetCloudAnchorState(long j, long j2);

    private native Pose nativeGetPose(long j, long j2);

    private native int nativeGetTrackingState(long j, long j2);

    private static native void nativeReleaseAnchor(long j, long j2);

    public void detach() {
        nativeDetach(this.session.nativeWrapperHandle, this.nativeHandle);
    }

    public boolean equals(Object obj) {
        return (obj instanceof Anchor) && ((Anchor) obj).nativeHandle == this.nativeHandle;
    }

    public void finalize() {
        long j = this.nativeHandle;
        if (j != 0) {
            nativeReleaseAnchor(this.nativeSymbolTableHandle, j);
        }
        super.finalize();
    }

    public String getCloudAnchorId() {
        return nativeGetCloudAnchorId(this.session.nativeWrapperHandle, this.nativeHandle);
    }

    public CloudAnchorState getCloudAnchorState() {
        return CloudAnchorState.forNumber(nativeGetCloudAnchorState(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public Pose getPose() {
        return nativeGetPose(this.session.nativeWrapperHandle, this.nativeHandle);
    }

    public TrackingState getTrackingState() {
        return TrackingState.forNumber(nativeGetTrackingState(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    public int hashCode() {
        return Long.valueOf(this.nativeHandle).hashCode();
    }
}