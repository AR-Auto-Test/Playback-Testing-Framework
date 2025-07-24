package org.opencv.ml;

/* loaded from: classes2.dex */
public class ParamGrid {
    public final long nativeObj;

    public ParamGrid(long j) {
        this.nativeObj = j;
    }

    public static ParamGrid __fromPtr__(long j) {
        return new ParamGrid(j);
    }

    public static ParamGrid create(double d2, double d3, double d4) {
        return __fromPtr__(create_0(d2, d3, d4));
    }

    private static native long create_0(double d2, double d3, double d4);

    private static native long create_1(double d2, double d3);

    private static native long create_2(double d2);

    private static native long create_3();

    private static native void delete(long j);

    private static native double get_logStep_0(long j);

    private static native double get_maxVal_0(long j);

    private static native double get_minVal_0(long j);

    private static native void set_logStep_0(long j, double d2);

    private static native void set_maxVal_0(long j, double d2);

    private static native void set_minVal_0(long j, double d2);

    public void finalize() {
        delete(this.nativeObj);
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public double get_logStep() {
        return get_logStep_0(this.nativeObj);
    }

    public double get_maxVal() {
        return get_maxVal_0(this.nativeObj);
    }

    public double get_minVal() {
        return get_minVal_0(this.nativeObj);
    }

    public void set_logStep(double d2) {
        set_logStep_0(this.nativeObj, d2);
    }

    public void set_maxVal(double d2) {
        set_maxVal_0(this.nativeObj, d2);
    }

    public void set_minVal(double d2) {
        set_minVal_0(this.nativeObj, d2);
    }

    public static ParamGrid create(double d2, double d3) {
        return __fromPtr__(create_1(d2, d3));
    }

    public static ParamGrid create(double d2) {
        return __fromPtr__(create_2(d2));
    }

    public static ParamGrid create() {
        return __fromPtr__(create_3());
    }
}