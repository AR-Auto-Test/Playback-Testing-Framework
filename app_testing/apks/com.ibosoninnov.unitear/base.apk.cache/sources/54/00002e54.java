package org.opencv.videoio;

import org.opencv.core.Mat;
import org.opencv.core.Size;

/* loaded from: classes2.dex */
public class VideoWriter {
    public final long nativeObj;

    public VideoWriter(long j) {
        this.nativeObj = j;
    }

    private static native long VideoWriter_0(String str, int i, int i2, double d2, double d3, double d4, boolean z);

    private static native long VideoWriter_1(String str, int i, int i2, double d2, double d3, double d4);

    private static native long VideoWriter_2(String str, int i, double d2, double d3, double d4, boolean z);

    private static native long VideoWriter_3(String str, int i, double d2, double d3, double d4);

    private static native long VideoWriter_4();

    public static VideoWriter __fromPtr__(long j) {
        return new VideoWriter(j);
    }

    private static native void delete(long j);

    public static int fourcc(char c2, char c3, char c4, char c5) {
        return fourcc_0(c2, c3, c4, c5);
    }

    private static native int fourcc_0(char c2, char c3, char c4, char c5);

    private static native String getBackendName_0(long j);

    private static native double get_0(long j, int i);

    private static native boolean isOpened_0(long j);

    private static native boolean open_0(long j, String str, int i, int i2, double d2, double d3, double d4, boolean z);

    private static native boolean open_1(long j, String str, int i, int i2, double d2, double d3, double d4);

    private static native boolean open_2(long j, String str, int i, double d2, double d3, double d4, boolean z);

    private static native boolean open_3(long j, String str, int i, double d2, double d3, double d4);

    private static native void release_0(long j);

    private static native boolean set_0(long j, int i, double d2);

    private static native void write_0(long j, long j2);

    public void finalize() {
        delete(this.nativeObj);
    }

    public double get(int i) {
        return get_0(this.nativeObj, i);
    }

    public String getBackendName() {
        return getBackendName_0(this.nativeObj);
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public boolean isOpened() {
        return isOpened_0(this.nativeObj);
    }

    public boolean open(String str, int i, int i2, double d2, Size size, boolean z) {
        return open_0(this.nativeObj, str, i, i2, d2, size.width, size.height, z);
    }

    public void release() {
        release_0(this.nativeObj);
    }

    public boolean set(int i, double d2) {
        return set_0(this.nativeObj, i, d2);
    }

    public void write(Mat mat) {
        write_0(this.nativeObj, mat.nativeObj);
    }

    public VideoWriter(String str, int i, int i2, double d2, Size size, boolean z) {
        this.nativeObj = VideoWriter_0(str, i, i2, d2, size.width, size.height, z);
    }

    public boolean open(String str, int i, int i2, double d2, Size size) {
        return open_1(this.nativeObj, str, i, i2, d2, size.width, size.height);
    }

    public boolean open(String str, int i, double d2, Size size, boolean z) {
        return open_2(this.nativeObj, str, i, d2, size.width, size.height, z);
    }

    public VideoWriter(String str, int i, int i2, double d2, Size size) {
        this.nativeObj = VideoWriter_1(str, i, i2, d2, size.width, size.height);
    }

    public boolean open(String str, int i, double d2, Size size) {
        return open_3(this.nativeObj, str, i, d2, size.width, size.height);
    }

    public VideoWriter(String str, int i, double d2, Size size, boolean z) {
        this.nativeObj = VideoWriter_2(str, i, d2, size.width, size.height, z);
    }

    public VideoWriter(String str, int i, double d2, Size size) {
        this.nativeObj = VideoWriter_3(str, i, d2, size.width, size.height);
    }

    public VideoWriter() {
        this.nativeObj = VideoWriter_4();
    }
}