package org.opencv.imgproc;

/* loaded from: classes2.dex */
public class GeneralizedHoughGuil extends GeneralizedHough {
    public GeneralizedHoughGuil(long j) {
        super(j);
    }

    public static GeneralizedHoughGuil __fromPtr__(long j) {
        return new GeneralizedHoughGuil(j);
    }

    private static native void delete(long j);

    private static native double getAngleEpsilon_0(long j);

    private static native double getAngleStep_0(long j);

    private static native int getAngleThresh_0(long j);

    private static native int getLevels_0(long j);

    private static native double getMaxAngle_0(long j);

    private static native double getMaxScale_0(long j);

    private static native double getMinAngle_0(long j);

    private static native double getMinScale_0(long j);

    private static native int getPosThresh_0(long j);

    private static native double getScaleStep_0(long j);

    private static native int getScaleThresh_0(long j);

    private static native double getXi_0(long j);

    private static native void setAngleEpsilon_0(long j, double d2);

    private static native void setAngleStep_0(long j, double d2);

    private static native void setAngleThresh_0(long j, int i);

    private static native void setLevels_0(long j, int i);

    private static native void setMaxAngle_0(long j, double d2);

    private static native void setMaxScale_0(long j, double d2);

    private static native void setMinAngle_0(long j, double d2);

    private static native void setMinScale_0(long j, double d2);

    private static native void setPosThresh_0(long j, int i);

    private static native void setScaleStep_0(long j, double d2);

    private static native void setScaleThresh_0(long j, int i);

    private static native void setXi_0(long j, double d2);

    @Override // org.opencv.imgproc.GeneralizedHough, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public double getAngleEpsilon() {
        return getAngleEpsilon_0(this.nativeObj);
    }

    public double getAngleStep() {
        return getAngleStep_0(this.nativeObj);
    }

    public int getAngleThresh() {
        return getAngleThresh_0(this.nativeObj);
    }

    public int getLevels() {
        return getLevels_0(this.nativeObj);
    }

    public double getMaxAngle() {
        return getMaxAngle_0(this.nativeObj);
    }

    public double getMaxScale() {
        return getMaxScale_0(this.nativeObj);
    }

    public double getMinAngle() {
        return getMinAngle_0(this.nativeObj);
    }

    public double getMinScale() {
        return getMinScale_0(this.nativeObj);
    }

    public int getPosThresh() {
        return getPosThresh_0(this.nativeObj);
    }

    public double getScaleStep() {
        return getScaleStep_0(this.nativeObj);
    }

    public int getScaleThresh() {
        return getScaleThresh_0(this.nativeObj);
    }

    public double getXi() {
        return getXi_0(this.nativeObj);
    }

    public void setAngleEpsilon(double d2) {
        setAngleEpsilon_0(this.nativeObj, d2);
    }

    public void setAngleStep(double d2) {
        setAngleStep_0(this.nativeObj, d2);
    }

    public void setAngleThresh(int i) {
        setAngleThresh_0(this.nativeObj, i);
    }

    public void setLevels(int i) {
        setLevels_0(this.nativeObj, i);
    }

    public void setMaxAngle(double d2) {
        setMaxAngle_0(this.nativeObj, d2);
    }

    public void setMaxScale(double d2) {
        setMaxScale_0(this.nativeObj, d2);
    }

    public void setMinAngle(double d2) {
        setMinAngle_0(this.nativeObj, d2);
    }

    public void setMinScale(double d2) {
        setMinScale_0(this.nativeObj, d2);
    }

    public void setPosThresh(int i) {
        setPosThresh_0(this.nativeObj, i);
    }

    public void setScaleStep(double d2) {
        setScaleStep_0(this.nativeObj, d2);
    }

    public void setScaleThresh(int i) {
        setScaleThresh_0(this.nativeObj, i);
    }

    public void setXi(double d2) {
        setXi_0(this.nativeObj, d2);
    }
}