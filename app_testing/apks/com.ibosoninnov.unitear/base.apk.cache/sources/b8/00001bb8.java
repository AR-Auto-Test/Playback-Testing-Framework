package com.google.ar.core;

import com.google.ar.core.exceptions.FatalException;
import java.util.Collection;

/* loaded from: classes.dex */
public class InstantPlacementPoint extends TrackableBase {

    /* loaded from: classes.dex */
    public enum TrackingMethod {
        NOT_TRACKING(0),
        SCREENSPACE_WITH_APPROXIMATE_DISTANCE(1),
        FULL_TRACKING(2);
        
        public final int nativeCode;

        TrackingMethod(int i) {
            this.nativeCode = i;
        }

        public static TrackingMethod fromNumber(int i) {
            TrackingMethod[] values = values();
            for (int i2 = 0; i2 < 3; i2++) {
                TrackingMethod trackingMethod = values[i2];
                if (trackingMethod.nativeCode == i) {
                    return trackingMethod;
                }
            }
            throw new FatalException(c.b.a.a.a.g(88, "Unexpected value for native InstantPlacementPoint TrackingMethod Mode, value=", i));
        }
    }

    public InstantPlacementPoint(long j, Session session) {
        super(j, session);
    }

    private native Pose nativeGetPose(long j, long j2);

    private native int nativeGetTrackingMethod(long j, long j2);

    @Override // com.google.ar.core.TrackableBase, com.google.ar.core.Trackable
    public /* bridge */ /* synthetic */ Anchor createAnchor(Pose pose) {
        return super.createAnchor(pose);
    }

    @Override // com.google.ar.core.TrackableBase
    public /* bridge */ /* synthetic */ boolean equals(Object obj) {
        return super.equals(obj);
    }

    @Override // com.google.ar.core.TrackableBase, com.google.ar.core.Trackable
    public /* bridge */ /* synthetic */ Collection getAnchors() {
        return super.getAnchors();
    }

    public Pose getPose() {
        return nativeGetPose(this.session.nativeWrapperHandle, this.nativeHandle);
    }

    public TrackingMethod getTrackingMethod() {
        return TrackingMethod.fromNumber(nativeGetTrackingMethod(this.session.nativeWrapperHandle, this.nativeHandle));
    }

    @Override // com.google.ar.core.TrackableBase, com.google.ar.core.Trackable
    public /* bridge */ /* synthetic */ TrackingState getTrackingState() {
        return super.getTrackingState();
    }

    @Override // com.google.ar.core.TrackableBase
    public /* bridge */ /* synthetic */ int hashCode() {
        return super.hashCode();
    }
}