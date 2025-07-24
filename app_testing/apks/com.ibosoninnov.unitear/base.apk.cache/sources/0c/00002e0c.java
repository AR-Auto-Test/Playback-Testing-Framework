package org.opencv.features2d;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

/* loaded from: classes2.dex */
public class BOWTrainer {
    public final long nativeObj;

    public BOWTrainer(long j) {
        this.nativeObj = j;
    }

    public static BOWTrainer __fromPtr__(long j) {
        return new BOWTrainer(j);
    }

    private static native void add_0(long j, long j2);

    private static native void clear_0(long j);

    private static native long cluster_0(long j, long j2);

    private static native long cluster_1(long j);

    private static native void delete(long j);

    private static native int descriptorsCount_0(long j);

    private static native long getDescriptors_0(long j);

    public void add(Mat mat) {
        add_0(this.nativeObj, mat.nativeObj);
    }

    public void clear() {
        clear_0(this.nativeObj);
    }

    public Mat cluster(Mat mat) {
        return new Mat(cluster_0(this.nativeObj, mat.nativeObj));
    }

    public int descriptorsCount() {
        return descriptorsCount_0(this.nativeObj);
    }

    public void finalize() {
        delete(this.nativeObj);
    }

    public List<Mat> getDescriptors() {
        ArrayList arrayList = new ArrayList();
        Converters.Mat_to_vector_Mat(new Mat(getDescriptors_0(this.nativeObj)), arrayList);
        return arrayList;
    }

    public long getNativeObjAddr() {
        return this.nativeObj;
    }

    public Mat cluster() {
        return new Mat(cluster_1(this.nativeObj));
    }
}