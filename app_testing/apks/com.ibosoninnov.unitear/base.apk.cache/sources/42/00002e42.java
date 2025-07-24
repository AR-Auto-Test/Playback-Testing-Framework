package org.opencv.photo;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.utils.Converters;

/* loaded from: classes2.dex */
public class MergeRobertson extends MergeExposures {
    public MergeRobertson(long j) {
        super(j);
    }

    public static MergeRobertson __fromPtr__(long j) {
        return new MergeRobertson(j);
    }

    private static native void delete(long j);

    private static native void process_0(long j, long j2, long j3, long j4, long j5);

    private static native void process_1(long j, long j2, long j3, long j4);

    @Override // org.opencv.photo.MergeExposures, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    @Override // org.opencv.photo.MergeExposures
    public void process(List<Mat> list, Mat mat, Mat mat2, Mat mat3) {
        process_0(this.nativeObj, Converters.vector_Mat_to_Mat(list).nativeObj, mat.nativeObj, mat2.nativeObj, mat3.nativeObj);
    }

    public void process(List<Mat> list, Mat mat, Mat mat2) {
        process_1(this.nativeObj, Converters.vector_Mat_to_Mat(list).nativeObj, mat.nativeObj, mat2.nativeObj);
    }
}