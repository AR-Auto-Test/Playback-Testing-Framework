package org.opencv.objdetect;

import org.opencv.core.Algorithm;

/* loaded from: classes2.dex */
public class BaseCascadeClassifier extends Algorithm {
    public BaseCascadeClassifier(long j) {
        super(j);
    }

    public static BaseCascadeClassifier __fromPtr__(long j) {
        return new BaseCascadeClassifier(j);
    }

    private static native void delete(long j);

    @Override // org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }
}