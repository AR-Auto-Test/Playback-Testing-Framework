package com.google.mediapipe.framework;

import com.google.common.base.Preconditions;
import com.google.mediapipe.proto.CalculatorProfileProto;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.ArrayList;
import java.util.List;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/GraphProfiler.class */
public class GraphProfiler {
    private final long nativeProfilerHandle;
    private final Graph mediapipeGraph;

    private native void nativeReset(long profilingContextHandle);

    private native void nativeResume(long profilingContextHandle);

    private native void nativePause(long profilingContextHandle);

    private native byte[][] nativeGetCalculatorProfiles(long profilingContextHandle);

    /* JADX INFO: Access modifiers changed from: package-private */
    public GraphProfiler(long nativeProfilerHandle, Graph mediapipeGraph) {
        Preconditions.checkState(nativeProfilerHandle != 0, "Invalid profiler, tearDown() might have been called already.");
        this.nativeProfilerHandle = nativeProfilerHandle;
        this.mediapipeGraph = mediapipeGraph;
    }

    public void reset() {
        synchronized (this.mediapipeGraph) {
            checkContext();
            nativeReset(this.nativeProfilerHandle);
        }
    }

    public void resume() {
        synchronized (this.mediapipeGraph) {
            checkContext();
            nativeResume(this.nativeProfilerHandle);
        }
    }

    public void pause() {
        synchronized (this.mediapipeGraph) {
            checkContext();
            nativePause(this.nativeProfilerHandle);
        }
    }

    public List<CalculatorProfileProto.CalculatorProfile> getCalculatorProfiles() {
        List<CalculatorProfileProto.CalculatorProfile> profileList;
        synchronized (this.mediapipeGraph) {
            checkContext();
            byte[][] profileBytes = nativeGetCalculatorProfiles(this.nativeProfilerHandle);
            profileList = new ArrayList<>();
            for (byte[] element : profileBytes) {
                try {
                    CalculatorProfileProto.CalculatorProfile profile = CalculatorProfileProto.CalculatorProfile.parseFrom(element);
                    profileList.add(profile);
                } catch (InvalidProtocolBufferException e2) {
                    throw new RuntimeException(e2);
                }
            }
        }
        return profileList;
    }

    private void checkContext() {
        Preconditions.checkState(this.mediapipeGraph.getNativeHandle() != 0, "Invalid context, tearDown() might have been called already.");
    }
}