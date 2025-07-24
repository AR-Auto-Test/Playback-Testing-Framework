package org.opencv.features2d;

import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;

/* loaded from: classes2.dex */
public class BOWKMeansTrainer extends BOWTrainer {
    public BOWKMeansTrainer(long j) {
        super(j);
    }

    private static native long BOWKMeansTrainer_0(int i, int i2, int i3, double d2, int i4, int i5);

    private static native long BOWKMeansTrainer_1(int i, int i2, int i3, double d2, int i4);

    private static native long BOWKMeansTrainer_2(int i, int i2, int i3, double d2);

    private static native long BOWKMeansTrainer_3(int i);

    public static BOWKMeansTrainer __fromPtr__(long j) {
        return new BOWKMeansTrainer(j);
    }

    private static native long cluster_0(long j, long j2);

    private static native long cluster_1(long j);

    private static native void delete(long j);

    @Override // org.opencv.features2d.BOWTrainer
    public Mat cluster(Mat mat) {
        return new Mat(cluster_0(this.nativeObj, mat.nativeObj));
    }

    @Override // org.opencv.features2d.BOWTrainer
    public void finalize() {
        delete(this.nativeObj);
    }

    public BOWKMeansTrainer(int i, TermCriteria termCriteria, int i2, int i3) {
        super(BOWKMeansTrainer_0(i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon, i2, i3));
    }

    @Override // org.opencv.features2d.BOWTrainer
    public Mat cluster() {
        return new Mat(cluster_1(this.nativeObj));
    }

    public BOWKMeansTrainer(int i, TermCriteria termCriteria, int i2) {
        super(BOWKMeansTrainer_1(i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon, i2));
    }

    public BOWKMeansTrainer(int i, TermCriteria termCriteria) {
        super(BOWKMeansTrainer_2(i, termCriteria.type, termCriteria.maxCount, termCriteria.epsilon));
    }

    public BOWKMeansTrainer(int i) {
        super(BOWKMeansTrainer_3(i));
    }
}