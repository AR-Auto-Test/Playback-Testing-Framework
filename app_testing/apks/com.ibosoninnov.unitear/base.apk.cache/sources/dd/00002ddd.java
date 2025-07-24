package org.opencv.core;

/* loaded from: classes2.dex */
public class Algorithm {
    public final long nativeObj;

    public Algorithm(long j) {
        this.nativeObj = j;
    }

    public static Algorithm __fromPtr__(long j) {
        return new Algorithm(j);
    }

    private static native void clear_0(long j);

    private static native void delete(long j);

    private static native boolean empty_0(long j);

    private static native String getDefaultName_0(long j);

    private static native void save_0(long j, String str);

    public void clear() {
        clear_0(this.nativeObj);
    }

    public boolean empty() {
        return empty_0(this.nativeObj);
    }

    public void finalize() {
        delete(this.nativeObj);
    }

    public String getDefaultName() {
        return getDefaultName_0(this.nativeObj);
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public void save(String str) {
        save_0(this.nativeObj, str);
    }
}