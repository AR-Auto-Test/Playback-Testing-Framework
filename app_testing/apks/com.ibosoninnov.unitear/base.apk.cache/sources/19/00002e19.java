package org.opencv.features2d;

/* loaded from: classes2.dex */
public class Params {
    public final long nativeObj;

    public Params(long j) {
        this.nativeObj = j;
    }

    private static native long Params_0();

    public static Params __fromPtr__(long j) {
        return new Params(j);
    }

    private static native void delete(long j);

    private static native boolean get_filterByArea_0(long j);

    private static native boolean get_filterByCircularity_0(long j);

    private static native boolean get_filterByColor_0(long j);

    private static native boolean get_filterByConvexity_0(long j);

    private static native boolean get_filterByInertia_0(long j);

    private static native float get_maxArea_0(long j);

    private static native float get_maxCircularity_0(long j);

    private static native float get_maxConvexity_0(long j);

    private static native float get_maxInertiaRatio_0(long j);

    private static native float get_maxThreshold_0(long j);

    private static native float get_minArea_0(long j);

    private static native float get_minCircularity_0(long j);

    private static native float get_minConvexity_0(long j);

    private static native float get_minDistBetweenBlobs_0(long j);

    private static native float get_minInertiaRatio_0(long j);

    private static native long get_minRepeatability_0(long j);

    private static native float get_minThreshold_0(long j);

    private static native float get_thresholdStep_0(long j);

    private static native void set_filterByArea_0(long j, boolean z);

    private static native void set_filterByCircularity_0(long j, boolean z);

    private static native void set_filterByColor_0(long j, boolean z);

    private static native void set_filterByConvexity_0(long j, boolean z);

    private static native void set_filterByInertia_0(long j, boolean z);

    private static native void set_maxArea_0(long j, float f2);

    private static native void set_maxCircularity_0(long j, float f2);

    private static native void set_maxConvexity_0(long j, float f2);

    private static native void set_maxInertiaRatio_0(long j, float f2);

    private static native void set_maxThreshold_0(long j, float f2);

    private static native void set_minArea_0(long j, float f2);

    private static native void set_minCircularity_0(long j, float f2);

    private static native void set_minConvexity_0(long j, float f2);

    private static native void set_minDistBetweenBlobs_0(long j, float f2);

    private static native void set_minInertiaRatio_0(long j, float f2);

    private static native void set_minRepeatability_0(long j, long j2);

    private static native void set_minThreshold_0(long j, float f2);

    private static native void set_thresholdStep_0(long j, float f2);

    public void finalize() {
        delete(this.nativeObj);
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public boolean get_filterByArea() {
        return get_filterByArea_0(this.nativeObj);
    }

    public boolean get_filterByCircularity() {
        return get_filterByCircularity_0(this.nativeObj);
    }

    public boolean get_filterByColor() {
        return get_filterByColor_0(this.nativeObj);
    }

    public boolean get_filterByConvexity() {
        return get_filterByConvexity_0(this.nativeObj);
    }

    public boolean get_filterByInertia() {
        return get_filterByInertia_0(this.nativeObj);
    }

    public float get_maxArea() {
        return get_maxArea_0(this.nativeObj);
    }

    public float get_maxCircularity() {
        return get_maxCircularity_0(this.nativeObj);
    }

    public float get_maxConvexity() {
        return get_maxConvexity_0(this.nativeObj);
    }

    public float get_maxInertiaRatio() {
        return get_maxInertiaRatio_0(this.nativeObj);
    }

    public float get_maxThreshold() {
        return get_maxThreshold_0(this.nativeObj);
    }

    public float get_minArea() {
        return get_minArea_0(this.nativeObj);
    }

    public float get_minCircularity() {
        return get_minCircularity_0(this.nativeObj);
    }

    public float get_minConvexity() {
        return get_minConvexity_0(this.nativeObj);
    }

    public float get_minDistBetweenBlobs() {
        return get_minDistBetweenBlobs_0(this.nativeObj);
    }

    public float get_minInertiaRatio() {
        return get_minInertiaRatio_0(this.nativeObj);
    }

    public long get_minRepeatability() {
        return get_minRepeatability_0(this.nativeObj);
    }

    public float get_minThreshold() {
        return get_minThreshold_0(this.nativeObj);
    }

    public float get_thresholdStep() {
        return get_thresholdStep_0(this.nativeObj);
    }

    public void set_filterByArea(boolean z) {
        set_filterByArea_0(this.nativeObj, z);
    }

    public void set_filterByCircularity(boolean z) {
        set_filterByCircularity_0(this.nativeObj, z);
    }

    public void set_filterByColor(boolean z) {
        set_filterByColor_0(this.nativeObj, z);
    }

    public void set_filterByConvexity(boolean z) {
        set_filterByConvexity_0(this.nativeObj, z);
    }

    public void set_filterByInertia(boolean z) {
        set_filterByInertia_0(this.nativeObj, z);
    }

    public void set_maxArea(float f2) {
        set_maxArea_0(this.nativeObj, f2);
    }

    public void set_maxCircularity(float f2) {
        set_maxCircularity_0(this.nativeObj, f2);
    }

    public void set_maxConvexity(float f2) {
        set_maxConvexity_0(this.nativeObj, f2);
    }

    public void set_maxInertiaRatio(float f2) {
        set_maxInertiaRatio_0(this.nativeObj, f2);
    }

    public void set_maxThreshold(float f2) {
        set_maxThreshold_0(this.nativeObj, f2);
    }

    public void set_minArea(float f2) {
        set_minArea_0(this.nativeObj, f2);
    }

    public void set_minCircularity(float f2) {
        set_minCircularity_0(this.nativeObj, f2);
    }

    public void set_minConvexity(float f2) {
        set_minConvexity_0(this.nativeObj, f2);
    }

    public void set_minDistBetweenBlobs(float f2) {
        set_minDistBetweenBlobs_0(this.nativeObj, f2);
    }

    public void set_minInertiaRatio(float f2) {
        set_minInertiaRatio_0(this.nativeObj, f2);
    }

    public void set_minRepeatability(long j) {
        set_minRepeatability_0(this.nativeObj, j);
    }

    public void set_minThreshold(float f2) {
        set_minThreshold_0(this.nativeObj, f2);
    }

    public void set_thresholdStep(float f2) {
        set_thresholdStep_0(this.nativeObj, f2);
    }

    public Params() {
        this.nativeObj = Params_0();
    }
}