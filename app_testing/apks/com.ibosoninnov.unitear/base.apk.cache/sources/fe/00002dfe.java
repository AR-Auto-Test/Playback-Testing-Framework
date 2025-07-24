package org.opencv.core;

/* loaded from: classes2.dex */
public class TickMeter {
    public final long nativeObj;

    public TickMeter(long j) {
        this.nativeObj = j;
    }

    private static native long TickMeter_0();

    public static TickMeter __fromPtr__(long j) {
        return new TickMeter(j);
    }

    private static native void delete(long j);

    private static native long getCounter_0(long j);

    private static native double getTimeMicro_0(long j);

    private static native double getTimeMilli_0(long j);

    private static native double getTimeSec_0(long j);

    private static native long getTimeTicks_0(long j);

    private static native void reset_0(long j);

    private static native void start_0(long j);

    private static native void stop_0(long j);

    public void finalize() {
        delete(this.nativeObj);
    }

    public long getCounter() {
        return getCounter_0(this.nativeObj);
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public double getTimeMicro() {
        return getTimeMicro_0(this.nativeObj);
    }

    public double getTimeMilli() {
        return getTimeMilli_0(this.nativeObj);
    }

    public double getTimeSec() {
        return getTimeSec_0(this.nativeObj);
    }

    public long getTimeTicks() {
        return getTimeTicks_0(this.nativeObj);
    }

    public void reset() {
        reset_0(this.nativeObj);
    }

    public void start() {
        start_0(this.nativeObj);
    }

    public void stop() {
        stop_0(this.nativeObj);
    }

    public TickMeter() {
        this.nativeObj = TickMeter_0();
    }
}